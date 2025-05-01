import re
import os
import json
import argparse

from pathlib import Path
from tqdm import tqdm

from sklearn.metrics import f1_score

import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from open_r1.utils.format_prompt import format_prompt

def load_model_and_processor(model_path: str="/lid/home/saydalie/multimodal_cot/LLaMA-Factory/output/qwen2_5_vl-7b/sft/bbox"):
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

def make_conversation(sample, data_dir, explanation_type, system_prompt=False):
    # https://github.com/QwenLM/Qwen2.5-VL/blob/fe0d43a3b74d70b40d28062c8b44d05978a0ed98/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L112C1-L113C1

    sample_formatted = format_prompt(sample, explanation_type=explanation_type)
    image_path = os.path.join(data_dir, sample['img_filename'])

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

    return responses

def extract_answer(text: str):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def normalize_answer(answer):
    """Normalizes an answer by removing punctuation, whitespace, and converting to lowercase."""
    return re.sub(r'[^\w\s]', '', answer).lower().strip()

def compute_metrics(generated_responses):
    n_no_format = 0
    n_correct = 0
    all_ground_truth = []
    all_predictions = []
    
    # Process each question-answer pair
    for data in generated_responses:
        ground_truth = extract_answer(data['ground_truth_response'])
        prediction = extract_answer(data['generated_response'])
        
        if prediction is None:
            n_no_format += 1
            prediction = "no_answer_provided"  # Handle cases where no answer was extracted
        
        # Normalize answers
        normalized_ground_truth = normalize_answer(ground_truth) if ground_truth else "no_answer_available"
        normalized_prediction = normalize_answer(prediction)
        
        # Add to collections for metrics calculation
        all_ground_truth.append(normalized_ground_truth)
        all_predictions.append(normalized_prediction)
        
        # Calculate accuracy
        if normalized_prediction == normalized_ground_truth:
            n_correct += 1
    
    # Calculate metrics
    no_format = n_no_format / len(generated_responses) if generated_responses else 0
    accuracy = n_correct / len(generated_responses) if generated_responses else 0

    # Calculate f1 score with different averaging methods
    f1_micro = f1_score(all_ground_truth, all_predictions, average='micro')
    f1_macro = f1_score(all_ground_truth, all_predictions, average='macro')
    f1_weighted = f1_score(all_ground_truth, all_predictions, average='weighted')
    
    return {
        'no_format': no_format,
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def main(args):
    # Load dataset
    with open(args.input_data_dir + '/DrivingVQA/test.json', "r") as f:
        data = list(json.load(f).values())

    # Remove double questions: 789 -> 576
    data = [d for d in data if not d['has_multiple_questions']]

    # Filter out large images: 576 -> 474
    data = [d for d in data if d['img_size'][0] * d['img_size'][1] <= 3686400]

    # Map the conversations
    data = [make_conversation(sample, args.input_data_dir, args.explanation_type, args.system_prompt) for sample in data]
    print(data[0])

    # Load the model
    model, processor = load_model_and_processor(args.model_path)

    # Generate and evaluate
    output_path = Path(args.output_data_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if (output_path / 'generated_responses.json').exists():
        with open(output_path / 'generated_responses.json', 'r') as f:
            generated_responses = json.load(f)
    else:
        generated_responses = generate_responses(data, model, processor)
    
        with open(output_path / 'generated_responses.json', 'w') as f:
            json.dump(generated_responses, f, indent=4) 

    metrics = compute_metrics(generated_responses)
    
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_data_dir", type=str, required=True)
    parser.add_argument("--explanation_type", type=str, required=True, help="bbox | original")
    parser.add_argument("--output_data_dir", type=str, default="/lid/home/saydalie/multimodal_cot/results/qwen2_5_vl")
    parser.add_argument("--system_prompt", action="store_true")
    args = parser.parse_args()

    main(args)

