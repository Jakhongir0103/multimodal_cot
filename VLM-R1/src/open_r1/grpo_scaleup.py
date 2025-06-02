# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


import os
import json
import pathlib
from typing import Optional
from dataclasses import dataclass, field

import torch
from transformers import set_seed

from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from open_r1.vlm_modules import Qwen2VLModule, InvernVLModule
from open_r1.rewards.rewards_scaleup import accuracy_reward, format_reward, bbox_reward
from open_r1.utils.format_prompt_scaleup import format_prompt

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from deepspeed.runtime.fp16.loss_scaler import LossScaler
torch.serialization.add_safe_globals([LossScaler])

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    explanation_type: str = field(
        default=None,
        metadata={"help": "Possible values: [original, bbox]"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "bbox"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'bbox'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    explanation_type: str = field(
        default='bbox',metadata={"help": "Possible values: [original, bbox]"}
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = True

# Format into conversation
def make_conversation(sample, max_pixels, min_pixels):
    """
    {'question': 'what is the contact person name mentioned in letter?',
    'answer': 'P. Carter',
    'possible_answers': ['P. Carter', 'p. carter'],
    'image': 'xnbl0037_1.png',
    'width': 1695,
    'height': 2025,
    'bboxs': [[429, 511, 666, 578], [429, 511, 666, 578]],
    'dataset': 'docvqa',
    'split': 'train'
    'image_path': <image_path>}
    """
    # https://github.com/QwenLM/Qwen2.5-VL/blob/fe0d43a3b74d70b40d28062c8b44d05978a0ed98/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L112C1-L113C1

    sample_formatted = format_prompt(sample=sample, max_pixels=max_pixels, min_pixels=min_pixels)

    return {
        'image_path': sample['image_path'],
        'bboxes': sample_formatted['bboxes'],
        'possible_answers': sample_formatted['possible_answers'],
        'prompt': [{
            'role': 'user',
            'content': [
                {'type': 'image', 'text': None},
                {'type': 'text', 'text': sample_formatted['prompt']}
            ]
        }]
    }

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "bbox": bbox_reward
}

def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def load_dataset(data_dir:str = '/lid/home/saydalie/multimodal_cot/VLM-R1/data/scaleup'):

    image_paths = {
        'coco': "", # images/coco/train2017
        'flickr30k': "", # images/flickr30k/flickr30k-images
        'gqa': 'images/gqa/images', 
        'ocr_vqa': "", # images/ocr_vqa/images
        'textvqa': "images/textvqa/train_images", 
        'v7w': "images/v7w/images", 
        'vg': "", # images/vg/VG_100K 
        'cub': "images/cot/cub/CUB_200_2011", 
        'docvqa': "images/cot/docvqa", 
        'dude': "images/cot/dude/DUDE_train-val-test_binaries/images/train", 
        'infographicsvqa': "images/cot/infographicsvqa", 
        'sroie': "images/cot/sroie/0325updated.task1train(626p)", 
        'vsr': "images/cot/vsr/images",
        'textcap': "images/textcap/train_images",
        'openimages': ""
    }    
    
    data_all = []

    for file_name in os.listdir(f"{data_dir}/data"):
        if file_name.endswith('.jsonl'):
            file_paht = os.path.join(f"{data_dir}/data", file_name)
            with open(file_paht, 'r', encoding='utf-8') as f:
                data_all.extend([json.loads(line) for line in f])

    # keep those with valid image path
    data_valid = []
    for data in data_all:
        image_path = f"{data_dir}/{image_paths[data['dataset']]}/{data['image']}"
        if data['dataset'] not in ['openimages', 'flickr30k'] and os.path.exists(image_path):
            data['image_path'] = image_path
            data_valid.append(data)

    return data_valid

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    # `dataset_name`: path to `scaleup` dir
    data = load_dataset(script_args.dataset_name)
    
    # Map the conversations
    data = [make_conversation(sample, script_args.max_pixels, script_args.min_pixels) for sample in data]
    print(f'Data Size: {len(data)}')
    print('Sample:')
    print(data[0])

    os.environ["WANDB_RUN_ID"] = training_args.run_name
    os.environ["WANDB_RESUME"] = "allow"

    # Initialize the GRPO trainer
    trainer = VLMGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=data,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and save the model
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    torch.cuda.synchronize()
    trainer.save_model(os.path.join(training_args.output_dir, 'final'))
    if training_args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
