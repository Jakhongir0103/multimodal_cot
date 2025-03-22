# -*- coding: utf-8 -*-

import os
import json
import argparse

from tqdm import tqdm

from PIL import Image
import torch

import sys
sys.path.append("/lid/home/saydalie/multimodal_cot/Emu3")

from emu3.tokenizer import Emu3VisionVQModel, Emu3VisionVQImageProcessor

MODEL_PATH = '/lid/home/saydalie/multimodal_cot/Emu3-models/Emu3-VisionTokenizer/snapshots/c81f916ad371289e205310a7539255e8a9396488/'
DATA_PATH = '/lid/home/saydalie/multimodal_cot/LLM-PuzzleTest/PuzzleVQA/data/'
DATA_OUTPUT_PATH = '/lid/home/saydalie/multimodal_cot/Emu3/data/'

def prepare_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model-path', type=str, help='vision tokenizer path')
    # parser.add_argument('--data-path', type=str, help='data path')
    # parser.add_argument('--output-path', type=str, help='tokenized data save path')
    parser.add_argument('--image-area', type=int, default=512 * 512)

    args = parser.parse_args()
    return args

def smart_resize(image, image_area: int = 512 * 512):
    w, h = image.size
    current_area = h * w
    target_ratio = (image_area / current_area) ** 0.5

    th = int(round(h * target_ratio))
    tw = int(round(w * target_ratio))

    image = image.resize((tw, th))
    return image

def build_puzzle_train_prompt(
    question: str,
    options: list[str],
    pattern: str,
    options_reasonings: list[str],
    final_answer: str
):
    map_option_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    prompt = """# Question:
<image>{question}

# Options:
{options}

# Pattern:
{pattern}

{options_reasonings}

# Final Answer:
{final_answer}"""
    prompt = prompt.format(
        question = question,
        options = '\n'.join([f"({map_option_to_letter[option_idx]}) {option}" for option_idx, option in enumerate(options)]),
        pattern = pattern,
        final_answer = final_answer,
        options_reasonings = '\n\n'.join([f"# Option {map_option_to_letter[option_idx]}\nReplacing '?' with {option}: <image>\nReasoning: {options_reasonings[option_idx]}" for option_idx, option in enumerate(options)])
    )
    return prompt

def build_puzzle_eval_prompt(
    question: str,
    options: list[str]
):
    map_option_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    prompt = """# Question:
<image>{question}

# Options:
{options}

# Pattern:
"""
    prompt = prompt.format(
        question = question,
        options = '\n'.join([f"({map_option_to_letter[option_idx]}) {option}" for option_idx, option in enumerate(options)])
    )
    return prompt

def main_train(pattern_names, data_cap=3000):
    args = prepare_args()

    image_processor = Emu3VisionVQImageProcessor.from_pretrained(MODEL_PATH)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(MODEL_PATH, device_map="cuda:0")
    image_tokenizer.eval()

    os.makedirs(f"{DATA_OUTPUT_PATH}/train/feature", exist_ok=True)
    os.makedirs(f"{DATA_OUTPUT_PATH}/train/list", exist_ok=True)

    datalist = {
        "prefix": f"{DATA_OUTPUT_PATH}/train/feature",
        "path_list": []
    }

    for pattern_name in pattern_names:

        with open(DATA_PATH + f'train/{pattern_name}.json', 'r') as f:
            input_data = [json.loads(line) for line in f]

        with open(DATA_PATH + f'train/{pattern_name}_options.json', 'r') as f:
            input_data_options = [json.loads(line) for line in f]

        for data_idx, (inp, inp_options) in tqdm(enumerate(zip(input_data, input_data_options)), desc="Tokenize train dataset", total=data_cap):
            
            if data_idx >= data_cap:
                break

            output_file_path = f"{DATA_OUTPUT_PATH}/train/feature/{pattern_name}_{data_idx:06}.pth"

            if os.path.exists(output_file_path):
                datalist["path_list"].append(f"{pattern_name}_{data_idx:06}.pth")
                continue

            input_text = build_puzzle_train_prompt(
                question = inp['question'],
                options = inp['options'],
                pattern = inp_options['pattern'],
                options_reasonings = inp_options['options_reasonings'],
                final_answer = inp_options['final_answer']
            )

            images_paths = [inp['image']] + inp_options['image_options_path']
            image_token_ids = []
            for image_path in images_paths:
                image = Image.open(DATA_PATH + f'train/{image_path}').convert("RGB")
                image = smart_resize(image, args.image_area)

                image = image_processor(image, return_tensors="pt")["pixel_values"]
                with torch.no_grad():
                    image = image.cuda()
                    token_ids = image_tokenizer.encode(image)

                image_token_ids.append(token_ids.cpu())

            image_token_ids = torch.cat(image_token_ids, dim=0).numpy()
            data = {
                "name": pattern_name,
                "images": image_token_ids,
                "text": input_text
            }

            torch.save(data, output_file_path)
            datalist["path_list"].append(f"{pattern_name}_{data_idx:06}.pth")

    with open(f"{DATA_OUTPUT_PATH}/train/list/train.json", 'w') as f:
        json.dump(datalist, f)

def main_eval(pattern_names, data_cap=100):
    args = prepare_args()

    image_processor = Emu3VisionVQImageProcessor.from_pretrained(MODEL_PATH)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(MODEL_PATH, device_map="cuda:0")
    image_tokenizer.eval()

    os.makedirs(f"{DATA_OUTPUT_PATH}/eval/feature", exist_ok=True)
    os.makedirs(f"{DATA_OUTPUT_PATH}/eval/list", exist_ok=True)

    datalist = {
        "prefix": f"{DATA_OUTPUT_PATH}/eval/feature",
        "path_list": []
    }

    for pattern_name in pattern_names:

        with open(DATA_PATH + f'eval/{pattern_name}.json', 'r') as f:
            input_data = [json.loads(line) for line in f]

        for data_idx, inp in tqdm(enumerate(input_data), desc="Tokenize eval dataset", total=data_cap):
            
            if data_idx >= data_cap:
                break

            output_file_path = f"{DATA_OUTPUT_PATH}/eval/feature/{pattern_name}_{data_idx:06}.pth"

            if os.path.exists(output_file_path):
                datalist["path_list"].append(f"{pattern_name}_{data_idx:06}.pth")
                continue

            input_text = build_puzzle_eval_prompt(
                question = inp['question'],
                options = inp['options'],
            )

            images_paths = [inp['image']]
            image_token_ids = []
            for image_path in images_paths:
                image = Image.open(DATA_PATH + f'eval/{image_path}').convert("RGB")
                image = smart_resize(image, args.image_area)

                image = image_processor(image, return_tensors="pt")["pixel_values"]
                with torch.no_grad():
                    image = image.cuda()
                    token_ids = image_tokenizer.encode(image)

                image_token_ids.append(token_ids.cpu())

            image_token_ids = torch.cat(image_token_ids, dim=0).numpy()
            data = {
                "name": pattern_name,
                "images": image_token_ids,
                "text": input_text
            }

            torch.save(data, output_file_path)
            datalist["path_list"].append(f"{pattern_name}_{data_idx:06}.pth")

    with open(f"{DATA_OUTPUT_PATH}/eval/list/train.json", 'w') as f:
        json.dump(datalist, f)

if __name__ == "__main__":
    # # train
    # pattern_names = ['color_number_hexagon', 'grid_number_color', 'rectangle_height_color', 'polygon_sides_number', 'triangle']
    # main_train(pattern_names)

    # eval
    pattern_names = ["circle_size_number", "color_number_hexagon", "grid_number", "polygon_sides_number", "shape_morph", "shape_size_hexagon", "triangle", 'color_grid', 'color_overlap_squares', 'grid_number_color', 'rectangle_height_color', 'shape_reflect', "size_cycle", 'venn', 'color_hexagon', 'color_size_circle', 'polygon_sides_color', 'rectangle_height_number', 'shape_size_grid', "size_grid"]
    # pattern_names = ["color_number_hexagon"]
    main_eval(pattern_names, data_cap=1)