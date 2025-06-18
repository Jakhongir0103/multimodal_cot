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
import random
import pathlib
from typing import Optional
from dataclasses import dataclass, field

import torch
from transformers import set_seed

from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from open_r1.vlm_modules import Qwen2VLModule, InvernVLModule
from open_r1.rewards.rewards import accuracy_reward, format_reward, bbox_reward
from open_r1.utils.format_prompt import format_prompt

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from deepspeed.runtime.fp16.loss_scaler import LossScaler
torch.serialization.add_safe_globals([LossScaler])

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_dir: str = field(
        default=None,
        metadata={"help": "Path to data dir"},
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
def make_conversation(sample, data_dir, explanation_type, dataset_name):
    # https://github.com/QwenLM/Qwen2.5-VL/blob/fe0d43a3b74d70b40d28062c8b44d05978a0ed98/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L112C1-L113C1

    if dataset_name == "drivingvqa":
        image_path = os.path.join(data_dir, sample['img_filename'])
    elif dataset_name == "aokvqa":
        image_path = os.path.join(data_dir, f"aokvqa/images/train2017/{sample['img_filename']}")
    sample_formatted = format_prompt(sample, explanation_type=explanation_type)

    return {
        'image_path': image_path,
        'solution': f"{sample_formatted['response']}",
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
    if script_args.dataset_name == 'drivingvqa':
        with open(script_args.data_dir + '/DrivingVQA/train.json', "r") as f:
            data = list(json.load(f).values())

        # Remove double questions: 3142 -> 2248
        data = [d for d in data if not d['has_multiple_questions']]

        # Filter out large images: 2248 -> 1885
        data = [d for d in data if d['img_size'][0] * d['img_size'][1] <= 3686400]

    elif script_args.dataset_name == 'aokvqa': 
        with open(script_args.data_dir + '/aokvqa/train.json', "r") as f:
            data = list(json.load(f).values())

    # Map the conversations
    data = [make_conversation(sample, script_args.data_dir, script_args.explanation_type, script_args.dataset_name) for sample in data]

    print(f'Data size: {len(data)}')
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
