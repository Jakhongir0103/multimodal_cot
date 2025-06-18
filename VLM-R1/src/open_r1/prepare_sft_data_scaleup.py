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

"""
Implementation is adapted from: https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl
"""

import os
import json
import argparse
import random

from transformers import set_seed

from open_r1.utils.format_prompt import format_prompt, format_okvqa_prompt_train

# Format into conversation
def make_conversation(sample):
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

    messages = [
        {
            'role': 'user',
            'content': "<image>" + sample['question']
        },
        {
            "role": "assistant",
            "content": f"<answer>{sample['answer']}</answer>"
        }
    ]
    images = [sample['image_path']]

    return {'messages': messages, 'images': images}

def load_dataset(data_dir:str):

    image_paths = {
        'coco': "images/coco/train2017",
        'flickr30k': "images/flickr30k/flickr30k-images",
        'gqa': 'images/gqa/images', 
        'ocr_vqa': "images/ocr_vqa/images", 
        'textvqa': "images/textvqa/train_images", 
        'v7w': "images/v7w/images", 
        'vg': "images/vg/VG_100K", 
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

def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Load the dataset
    # `dataset_name`: path to `scaleup` dir
    data = load_dataset(args.data_dir)
    
    # Map the conversations
    data = [make_conversation(sample) for sample in data]
    print(f'Data Size: {len(data)}')
    print('Sample:')
    print(data[0])

    random.shuffle(data)

    # Save
    with open(args.data_dir + f"/sft_no_reasoning/{args.output_name}.json", "w+") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)