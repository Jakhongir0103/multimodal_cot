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

from open_r1.utils.format_prompt import format_prompt, format_okvqa_prompt_val

def load_model_and_processor(model_path: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # default processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=4*4,
        max_pixels=1920*1920
    )

    return model, processor

def make_conversation(sample, data_dir, explanation_type, dataset, system_prompt=False):
    # https://github.com/QwenLM/Qwen2.5-VL/blob/fe0d43a3b74d70b40d28062c8b44d05978a0ed98/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L112C1-L113C1

    if dataset == 'drivingvqa':
        sample_formatted = format_prompt(sample, explanation_type=explanation_type)
        image_path = os.path.join(data_dir, sample['img_filename'])
    elif dataset == 'aokvqa':
        sample_formatted = format_okvqa_prompt_val(sample, explanation_type=explanation_type)
        image_path = os.path.join(data_dir, f"aokvqa/images/val2017/{sample['image_id']:012}.jpg")

    if system_prompt:
        messages = [
            {
                "role": "system",
                "content": "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."
            }
        ]
    else:
        messages = []

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": sample_formatted['prompt']},
            ],
        }
    )

    return messages, sample_formatted['response']

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

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=512)
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
        # Extract capital letters A-D or numbers in parentheses, remove duplicates, and sort
        results = sorted(set(re.findall(r"[A-D]|\(\d+\)", match.group(1))))
        return results
    else:
        return []

def compute_subset_accuracy(preds: Dict[str, List[str]], true_answers: Dict[str, List[str]]) -> float:
    """
    Computes the Exam Score (Subset Accuracy): the proportion of questions
    where the model predicted all correct answers and no incorrect answers.
    """
    exact_matches = sum(set(pred) == set(true_answers[qid]) for qid, pred in preds.items())
    return 100 * exact_matches / len(preds) if preds else 0

def compute_precision_recall_f1(preds: Dict[str, List[str]], true_answers: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F1-score for multi-label classification.
    """
    true_positives, false_positives, false_negatives = 0, 0, 0

    for qid, pred in preds.items():
        true_set = set(true_answers[qid])
        pred_set = set(pred)

        true_positives += len(true_set & pred_set)       # Correctly predicted answers
        false_positives += len(pred_set - true_set)      # Incorrectly predicted answers
        false_negatives += len(true_set - pred_set)      # Missed correct answers

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision * 100, "recall": recall * 100, "f1_score": f1_score * 100}

def compute_scores(generated_responses):
    all_ground_truth = {}
    all_predictions = {}
    
    # Extract answer from each question-answer pair
    for rid, data in enumerate(generated_responses):
        ground_truth = extract_answer(data['ground_truth_response'])
        prediction = extract_answer(data['generated_response'])

        all_ground_truth[rid] = ground_truth
        all_predictions[rid] = prediction
    
    # Compute scores
    subset_accuracy = compute_subset_accuracy(all_predictions, all_ground_truth)
    precision_recall_f1 = compute_precision_recall_f1(all_predictions, all_ground_truth)

    return {"exam_score": subset_accuracy, **precision_recall_f1}

def main(args):
    # Load dataset
    if args.dataset == 'drivingvqa':
        with open(args.input_data_dir + '/DrivingVQA/test.json', "r") as f:
            data = list(json.load(f).values())
        # Remove double questions: 789 -> 576
        data = [d for d in data if not d['has_multiple_questions']]
        # Filter out large images: 576 -> 474
        data = [d for d in data if d['img_size'][0] * d['img_size'][1] <= 3686400]

    elif args.dataset == 'aokvqa':
        with open(args.input_data_dir + '/aokvqa/val.json', "r") as f:
            data = list(json.load(f))

    # Map the conversations
    data = [make_conversation(sample, args.input_data_dir, args.explanation_type, args.dataset, args.system_prompt) for sample in data]
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
    parser.add_argument("--input_data_dir", type=str, required=True)
    parser.add_argument("--explanation_type", type=str, required=True, help="bbox | original")
    parser.add_argument("--output_data_dir", type=str)
    parser.add_argument("--system_prompt", action="store_true")
    parser.add_argument("--dataset", type=str, required=True, help="drivingvqa | aokvqa")
    args = parser.parse_args()

    main(args)