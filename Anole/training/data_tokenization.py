import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append("/lid/home/saydalie/multimodal_cot")
sys.path.append("/lid/home/saydalie/multimodal_cot/anole")

from tqdm import tqdm
from PIL import Image
from copy import deepcopy

from chameleon.inference.chameleon import TokenManager
from training.constants_training import (
    DATASET_RAW_DIR,
    DATASET_TOKENIZED_DIR
)
from anole.constants import (
    TOKENIZER_TEXT_PATH,
    TOKENIZER_IMAGE_CFG_PATH,
    TOKENIZER_IMAGE_PATH,
)

token_manager = TokenManager(
    tokenizer_path = TOKENIZER_TEXT_PATH.as_posix(),
    vqgan_cfg_path = TOKENIZER_IMAGE_CFG_PATH.as_posix(),
    vqgan_ckpt_path = TOKENIZER_IMAGE_PATH.as_posix(),
    device = 'cuda'
)

def build_puzzle_train_prompt_txt(
    question: str,
    options: list[int|str],
    answer: int|str,
    caption: str,
    explanation: str,
    deduction: str
):
    try:
        answer = int(answer)
        options = [int(option) for option in options]
    except:
        answer = str(answer)
        options = [str(option) for option in options]

    map_option_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    answer_id = options.index(answer)

    prompt = """# Question:
<image>{question}

# Options:
{options}

# Pattern:
{caption} {explanation}

# Final Answer:
{deduction} Therefore, among {option_letters}, the answer is: ({answer_letter})."""
    prompt = prompt.format(
        question = question,
        options = '\n'.join([f"({map_option_to_letter[option_idx]}) {option}" for option_idx, option in enumerate(options)]),
        caption = caption,
        explanation = explanation,
        option_letters = "(A) (B) (C) (D)" if len(options) == 4 else "(A) (B) (C)",
        deduction = deduction,
        answer_letter = map_option_to_letter[answer_id]
    )
    return prompt

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
    options: list[str],
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

def sft_puzzle_train_tokenization_txt(data_dir: str, pattern_name: str) -> list[dict]:
    data_questions_path = os.path.join(data_dir, f'{pattern_name}.json')
    with open(data_questions_path, 'r') as input_file:
        data_questions = [json.loads(line) for line in input_file]

    output_data = []
    for data_idx, data_question in tqdm(enumerate(data_questions), desc="Tokenize train dataset"):
        # if data_idx >= 100:
        #     # for OOD evaluation
        #     break

        input_text = build_puzzle_train_prompt_txt(
            question = data_question['question'],
            options = data_question['options'],
            answer = data_question['answer'],
            caption = data_question['caption'],
            explanation = data_question['explanation'],
            deduction = data_question['deduction']
        )
        entry = {'input_text': input_text}
        
        texts = input_text.split('<image>')
        input_text_ids = []

        for text_id, text in enumerate(texts):
            text_tokens = token_manager.tokenize_text(text)

            image_path = None
            if text_id == 0:
                image_path = os.path.join(data_dir, data_question['image'])
                image_tokens = token_manager.tokenize_image(Image.open(image_path))
                input_text_ids.extend(text_tokens + image_tokens)
            else:
                input_text_ids.extend(text_tokens)
            
        entry['input_text_ids'] = deepcopy(input_text_ids)
        output_data.append(entry)
    
    return output_data

def sft_puzzle_train_tokenization(data_dir: str, pattern_name: str) -> list[dict]:
    data_questions_path = os.path.join(data_dir, f'{pattern_name}.json')
    with open(data_questions_path, 'r') as input_file:
        data_questions = [json.loads(line) for line in input_file]

    data_options_path = os.path.join(data_dir, f'{pattern_name}_options.json')
    with open(data_options_path, 'r') as input_file:
        data_options = [json.loads(line) for line in input_file]

    output_data = []
    for data_idx, (data_question, data_option) in tqdm(enumerate(zip(data_questions, data_options)), desc="Tokenize train dataset"):
        # if data_idx >= 100:
        #     # for OOD evaluation
        #     break

        input_text = build_puzzle_train_prompt(
            question = data_question['question'],
            options = data_question['options'],
            pattern = data_option['pattern'],
            options_reasonings = data_option['options_reasonings'],
            final_answer = data_option['final_answer']
        )
        entry = {'input_text': input_text}
        
        texts = input_text.split('<image>')
        input_text_ids = []

        for text_id, text in enumerate(texts):
            text_tokens = token_manager.tokenize_text(text)

            image_path = None
            if text_id == 0:
                image_path = os.path.join(data_dir, data_question['image'])
            elif text_id < len(texts) - 1:
                image_path = os.path.join(data_dir, data_option['image_options_path'][text_id-1])

            if image_path:
                image_tokens = token_manager.tokenize_image(Image.open(image_path))
                input_text_ids.extend(text_tokens + image_tokens)
            else:
                input_text_ids.extend(text_tokens)
            
        entry['input_text_ids'] = deepcopy(input_text_ids)
        output_data.append(entry)
    
    return output_data

def sft_puzzle_eval_tokenization(data_dir: str, pattern_name: str) -> list[dict]:
    data_questions_path = os.path.join(data_dir, f'{pattern_name}.json')
    with open(data_questions_path, 'r') as input_file:
        data_questions = [json.loads(line) for line in input_file]

    output_data = []
    for data_question in tqdm(data_questions, desc="Tokenize eval dataset"):
        input_text = build_puzzle_eval_prompt(
            question = data_question['question'],
            options = data_question['options'],
        )
        entry = {'input_text': input_text}
        
        texts = input_text.split('<image>')
        input_text_ids = []

        for text_id, text in enumerate(texts):
            text_tokens = token_manager.tokenize_text(text)

            if text_id == 0:
                image_path = os.path.join(data_dir, data_question['image'])
                image_tokens = token_manager.tokenize_image(Image.open(image_path))
                input_text_ids.extend(text_tokens + image_tokens)
            else:
                input_text_ids.extend(text_tokens)
            
        entry['input_text_ids'] = deepcopy(input_text_ids)
        output_data.append(entry)
    
    return output_data

if __name__ == "__main__":
        
    # pattern_names_train = ['color_number_hexagon', 'grid_number_color', 'rectangle_height_color', 'polygon_sides_number', 'triangle']
    # pattern_names_ood = ['color_grid', 'color_overlap_squares', 'rectangle_height_number', 'polygon_sides_color', 'shape_reflect']

    # for pattern_name in pattern_names_train:
    #     print(f'Processing {pattern_name} (train)')

    #     output_file_path = os.path.join(DATASET_TOKENIZED_DIR, f"train/{pattern_name}_txt.jsonl")

    #     if os.path.exists(output_file_path):
    #         print(f"File '{output_file_path}' already exists. Exiting script.")
    #         continue

    #     output_data = sft_puzzle_train_tokenization_txt(os.path.join(DATASET_RAW_DIR, "train"), pattern_name)
    #     # output_data = sft_puzzle_train_tokenization(os.path.join(DATASET_RAW_DIR, split), pattern_name)

    #     with open(output_file_path, 'w') as output_file:
    #         for entry in output_data:
    #             output_file.write(json.dumps(entry) + '\n')

    pattern_names_all = ["circle_size_number", "color_number_hexagon", "grid_number", "polygon_sides_number", "shape_morph", "shape_size_hexagon", "triangle", 'color_grid', 'color_overlap_squares', 'grid_number_color', 'rectangle_height_color', 'shape_reflect', "size_cycle", 'venn', 'color_hexagon', 'color_size_circle', 'polygon_sides_color', 'rectangle_height_number', 'shape_size_grid', "size_grid"]

    for pattern_name in pattern_names_all:
        print(f'Processing {pattern_name} (eval)')

        output_file_path = os.path.join(DATASET_TOKENIZED_DIR, f"eval/{pattern_name}.jsonl")

        if os.path.exists(output_file_path):
            print(f"File '{output_file_path}' already exists. Exiting script.")
            continue

        output_data = sft_puzzle_eval_tokenization(os.path.join(DATASET_RAW_DIR, "eval"), pattern_name)

        with open(output_file_path, 'w') as output_file:
            for entry in output_data:
                output_file.write(json.dumps(entry) + '\n')