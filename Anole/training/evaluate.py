import os
import re
import json
import torch
import jsonlines
import argparse

from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from copy import deepcopy

import sys
sys.path.append("/lid/home/saydalie/multimodal_cot")
sys.path.append("/lid/home/saydalie/multimodal_cot/anole")

import pandas as pd

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from transformers import ChameleonForCausalLM, ChameleonProcessor

from training.constants_training import (
    ANOLE_PATH_HF,
    ANOLE_PATH_HF_TRAINED,
    DATASET_TOKENIZED_DIR,
    RESULTS_DIR,
    DATASET_RAW_DIR
)

PAD_TOKEN_ID = 1

def extract_final_answer(text: str) -> str:
    # Normalize spaces and remove excessive newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Define patterns to capture various answer formats
    patterns = [
        r"answer\s*is[:\-]?\s*\(?([A-D])\)?",     # Standard: "answer is: (A)" or "answer is A"
        r"final\s*answer[:\-]?\s*\(?([A-D])\)?",   # "Final answer: A" or "Final answer (A)"
        r"correct\s*(?:option|choice|answer)[:\-]?\s*\(?([A-D])\)?",  # "Correct choice: A"
    ]

    # Try matching each pattern
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None

# Define the dataset class
class TokenizedDataset(Dataset):
    def __init__(self, data_dir, pattern_names, start_idx=0, end_idx=9999):
        self.tokenized_data = []

        for pattern_name in pattern_names:
            data_file_path = os.path.join(data_dir, f"{pattern_name}.jsonl")

            with jsonlines.open(data_file_path) as reader:
                for idx, obj in enumerate(reader):
                    if (idx < start_idx) or (idx > end_idx):
                        continue

                    self.tokenized_data.append(torch.tensor(obj['input_text_ids'], dtype=torch.long))
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]

# Define custom collate function
def collate_fn(batch):
    # pad_id = 1
    # left padding
    batch_flipped = [item.flip(0) for item in batch]
    batch_inputs_padded = pad_sequence(batch_flipped, batch_first=True, padding_value=PAD_TOKEN_ID).flip(1)
    
    # Create attention mask
    attention_mask = (batch_inputs_padded != PAD_TOKEN_ID).long()

    return {'input_ids': batch_inputs_padded, 'attention_mask': attention_mask}
    
def evaluate(model_path: str, max_length: int, pattern_names: List[str], device='cuda'):

    # Load the model
    model = ChameleonForCausalLM.from_pretrained(
        ANOLE_PATH_HF_TRAINED / model_path,
        torch_dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()

    processor = ChameleonProcessor.from_pretrained(ANOLE_PATH_HF)

    # outputs_per_pattern = {}
    for pattern_name in pattern_names:
        os.makedirs(RESULTS_DIR / f"{model_path}", exist_ok=True)
        output_path = RESULTS_DIR / f"{model_path}/{pattern_name}.jsonl"

        if os.path.exists(output_path):
            print(f"File '{output_path}' already exists.")
            continue
        else:
            print(f"Evaluating '{pattern_name}'.")

        # Prepare the Dataset
        eval_dataset = TokenizedDataset(
            data_dir=os.path.join(DATASET_TOKENIZED_DIR, "eval"),
            pattern_names=[pattern_name]
        )

        eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=8,
            collate_fn=collate_fn,
            shuffle=False
        )

        # Run inference
        outputs = []
        for entry in tqdm(eval_dataloader):
            with torch.no_grad():
                generate_ids = model.generate(
                    input_ids = entry['input_ids'].to(device),
                    attention_mask = entry['attention_mask'].to(device),
                    max_length = max_length,
                    do_sample=False,
                    pad_token_id=PAD_TOKEN_ID
                )

            batch_decoded_outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for batch_decoded_output in batch_decoded_outputs:
                outputs.append({'output_text': batch_decoded_output, 'answer_predicted_letter': extract_final_answer(batch_decoded_output)})

        # Concat and Save
        output_df = pd.DataFrame(outputs)
        data_raw = pd.read_json(DATASET_RAW_DIR / f'eval/{pattern_name}.json', lines=True)
        output_full = pd.concat([data_raw, output_df], axis=1)

        output_full['answer_predicted'] = output_full.apply(lambda row: letter_to_answer(row), axis=1)

        output_full['answer'] = output_full['answer'].astype(str)
        output_full['answer_predicted'] = output_full['answer_predicted'].astype(str)

        with open(output_path, 'w') as output_file:
            for id, r in output_full.iterrows():
                output_file.write(json.dumps(r.to_dict()) + '\n')

    #     outputs_per_pattern[pattern_name] = deepcopy(outputs)
    
    # return outputs_per_pattern

def letter_to_answer(row):
    map_letter_to_option = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    answer = map_letter_to_option.get(row['answer_predicted_letter'], -1)

    if answer == -1:
        answer_predicted = ''
    else:
        try:
            answer_predicted = row['options'][answer]
        except:
            answer_predicted = ''
    
    return answer_predicted


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--max_length', type=int, default=2048)
    args = parser.parse_args()

    pattern_names = ["circle_size_number", "color_number_hexagon", "grid_number", "polygon_sides_number", "shape_morph", "shape_size_hexagon", "triangle", 'color_grid', 'color_overlap_squares', 'grid_number_color', 'rectangle_height_color', 'shape_reflect', "size_cycle", 'venn', 'color_hexagon', 'color_size_circle', 'polygon_sides_color', 'rectangle_height_number', 'shape_size_grid', "size_grid"]
    # pattern_names = {
    #     'in_domain': ['color_number_hexagon', 'grid_number_color', 'rectangle_height_color', 'polygon_sides_number', 'triangle'],
    #     'out_of_domain': ["circle_size_number", "grid_number", "shape_morph", "shape_size_hexagon", 'color_grid', 'color_overlap_squares', 'shape_reflect', "size_cycle", 'venn', 'color_hexagon', 'color_size_circle', 'polygon_sides_color', 'rectangle_height_number', 'shape_size_grid', "size_grid"]
    # }

    # outputs_per_pattern = 
    evaluate(
        model_path=args.model_path,
        max_length=args.max_length,
        pattern_names=pattern_names
    )

    # # Concatenate with the original dataset -> Compute accuracy
    # for pattern_name in pattern_names:
    #     output_df = pd.DataFrame(outputs_per_pattern[pattern_name])
    #     data_raw = pd.read_json(DATASET_RAW_DIR / f'eval/{pattern_name}.json', lines=True)
    #     output_full = pd.concat([data_raw, output_df], axis=1)

    #     output_full['answer_predicted'] = output_full.apply(lambda row: letter_to_answer(row), axis=1)

    #     output_full['answer'] = output_full['answer'].astype(str)
    #     output_full['answer_predicted'] = output_full['answer_predicted'].astype(str)

    #     # Save
    #     output_path = RESULTS_DIR / f"{args.model_path}/{pattern_name}.jsonl"
    #     with open(output_path, 'w') as output_file:
    #         for id, r in output_df.iterrows():
    #             output_file.write(json.dumps(r.to_dict()) + '\n')
