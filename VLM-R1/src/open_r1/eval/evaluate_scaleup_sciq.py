import re
import os
import json
import argparse

from typing import List, Dict

from pathlib import Path
from tqdm import tqdm

import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from open_r1.eval.format_prompt_eval import format_prompt_sqa
import random
random.seed(42)

def load_model_and_processor(model_path: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # default processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=56*56,
        max_pixels=1920*1920
    )

    return model, processor

def make_conversation(sample, answers, image_folder):
    # https://github.com/QwenLM/Qwen2.5-VL/blob/fe0d43a3b74d70b40d28062c8b44d05978a0ed98/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L112C1-L113C1

    sample_formatted = format_prompt_sqa(sample, answers)
    
    if 'images' in sample:
        image_path = os.path.join(image_folder, sample['image'])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": sample_formatted['prompt']},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample_formatted['prompt']},
                ],
            }
        ]

    return messages, sample_formatted['answer']

def generate_responses(data, model, processor):
    responses = []

    for messages, gnd_response in tqdm(data):
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Inference: Greedy Decoding
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            responses.append({"ground_truth_response": gnd_response, "generated_response": output_text[0]})
        except Exception as e:
            print(e)
            print("messages")
            print(messages)
            print("gnd_response")
            print(gnd_response)
            raise Exception

    return responses

def extract_answer(text: str) -> List[str]:
    # Extract the final answer within <answer> </answer> tags
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        # lowercase and strip the answer
        return [match.group(1).lower().strip()]
    else:
        return []

def compute_precision_recall_f1(preds: Dict[str, List[str]], true_answers: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F1-score for multi-label classification.
    """
    true_positives, false_positives, false_negatives = 0, 0, 0
    exact_match_count = 0
    total = len(true_answers)

    for qid, pred in preds.items():
        true_set = set(true_answers[qid])
        pred_set = set(pred)

        true_positives += len(true_set & pred_set)       # Correctly predicted answers
        false_positives += len(pred_set - true_set)      # Incorrectly predicted answers
        false_negatives += len(true_set - pred_set)      # Missed correct answers

        if true_set == pred_set:
            exact_match_count += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = exact_match_count / total if total > 0 else 0

    return {"accuracy": accuracy * 100, "precision": precision * 100, "recall": recall * 100, "f1_score": f1_score * 100}

def compute_scores(generated_responses):
    all_ground_truth = {}
    all_predictions = {}
    
    # Extract answer from each question-answer pair
    for rid, data in enumerate(generated_responses):
        ground_truth = extract_answer(f"<answer>{data['ground_truth_response']}</answer>")
        prediction = extract_answer(data['generated_response'])

        all_ground_truth[rid] = ground_truth
        all_predictions[rid] = prediction
    
    # Compute scores
    precision_recall_f1 = compute_precision_recall_f1(all_predictions, all_ground_truth)

    return {**precision_recall_f1}

def main(args):
    # Load dataset
    with open(args.questions_file, "r") as f:
        data = list(json.load(f))
    
    # only select a subset with images
    data = [
        {
            'question_id': d['id'],
            'image': d['image'],
            'text': d['conversations'][0]['value'],
        } for d in data if 'image' in d
    ]

    data = random.sample(data, min(2000, len(data))) # select a subset of evaluation dataset

    with open(args.answers_file, "r") as f:
        answers = [json.loads(line) for line in f if line.strip()]
        answers = {ans['question_id']: ans['text'] for ans in answers}

    # Map the conversations
    data = [make_conversation(sample=sample, answers=answers, image_folder=args.image_folder) for sample in data]
    print(data[0])

    # Generate and evaluate
    output_path = Path(args.output_data_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if (output_path / 'generated_responses.json').exists():
        with open(output_path / 'generated_responses.json', 'r') as f:
            generated_responses = json.load(f)
    else:
        # Load the model
        model, processor = load_model_and_processor(args.model_path)
        generated_responses = generate_responses(data, model, processor)
    
        with open(output_path / 'generated_responses.json', 'w') as f:
            json.dump(generated_responses, f, indent=4) 

    scores = compute_scores(generated_responses)
    
    with open(output_path / 'scores.json', 'w') as f:
        json.dump(scores, f, indent=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--questions_file", type=str, required=True)
    parser.add_argument("--answers_file", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_data_dir", type=str)
    args = parser.parse_args()

    main(args)