import re
import json
import argparse

from pathlib import Path
from tqdm import tqdm

import torch
from datasets import Dataset
from transformers import set_seed

set_seed(42)

import sys
sys.path.append("/lid/home/saydalie/multimodal_cot/SEED/")

from models.seed_llama_tokenizer import SeedLlamaTokenizer
from models.model_tools import get_pretrained_llama_causal_model

user_token = "USER"
assistant_token = "ASSISTANT"

DATA_PATH = "/lid/home/saydalie/multimodal_cot/SEED/data/ReSQ/test_resq.json"
MODEL_NAME_OR_PATH = "/lid/home/saydalie/multimodal_cot/SEED/checkpoints/seed-llama-8b-sft-comm"
MODEL_DIR = "/lid/home/saydalie/multimodal_cot/models/SEED_trained/"
RESULTS_DIR = "/lid/home/saydalie/multimodal_cot/results/SEED_resq/"

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

IMG_FLAG = '<image>'
NUM_IMG_TOKNES = 32
NUM_IMG_CODES = 8192
image_id_shift = 32000

IMAGE_TEXT_PROMPT = """{bos}{user_token}: I now describe a scene and ask a question about it. First, generate an image that helps visualize the scene described. Then, think about the reasoning process in the mind and then provide with the final answer. The final answer is either one of {candidate_answers}. The image and reasoning process are enclosed within <think> </think> tags, while the final answer is enclosed within <answer> </answer> tags, i.e., <think> image and the textual reasoning process here </think> <answer> the final answer here </answer>.
{question}
{assistant_token}:"""

TEXT_PROMPT = """{bos}{user_token}: I now describe a scene and ask a question about it. First, think about the reasoning process in the mind and then provide with the final answer. The final answer is either one of {candidate_answers}. The reasoning process and the final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> the final answer here </answer>.
{question}
{assistant_token}:"""

def load_tokenizer():
    tokenizer = SeedLlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME_OR_PATH,
        fp16=True,
        padding_side='left',
        load_diffusion=False
    )

    return tokenizer

def load_dataset(bos_token, prompt_type):
    assert prompt_type in ['text', 'image']

    prompt = TEXT_PROMPT if prompt_type == 'text' else IMAGE_TEXT_PROMPT

    with open(DATA_PATH, "r") as file:
        data = json.load(file)['data']
        
    test_dataset = []

    for d in data:
        story = d['story']
        questions = d['questions']
        
        for q in questions:
            question = q['question']
            answer = q['answer'][0]
            candidate_answers = q['candidate_answers']
            num_1st_context_sentences = q['num_1st_context_sentences']

            scene = ' '.join(story[:num_1st_context_sentences])
            test_dataset.append({
                'prompt': prompt.format(
                    bos=bos_token,
                    user_token=user_token,
                    assistant_token=assistant_token,
                    candidate_answers=candidate_answers,
                    question=f"{scene} {question}"),
                'scene': scene,
                'answer': answer
            })

    test_dataset = Dataset.from_list(test_dataset)
    return test_dataset

def decode_image_text(generate_ids, tokenizer):
    "Return extracted text from output."

    boi_list = torch.where(generate_ids == tokenizer(BOI_TOKEN, add_special_tokens=False).input_ids[0])[0]
    eoi_list = torch.where(generate_ids == tokenizer(EOI_TOKEN, add_special_tokens=False).input_ids[0])[0]

    if len(boi_list) == 0 and len(eoi_list) == 0:
        text_ids = generate_ids
        texts = tokenizer.decode(text_ids, skip_special_tokens=True)
        return texts

    else:
        boi_index = boi_list[0]

        text_ids = generate_ids[:boi_index]
        if len(text_ids) != 0:
            texts = tokenizer.decode(text_ids, skip_special_tokens=True)
            return texts
        else:
            return ""

def extract_answer(text: str):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def normalize_answer(answer):
    """Normalizes an answer by stripping whitespace, converting to lowercase, and removing punctuation."""
    return re.sub(r'[^a-zA-Z0-9]', '', answer).lower()

def evaluate(model_name, prompt_type, device):
    tokenizer = load_tokenizer()
    test_dataset = load_dataset(tokenizer.bos_token, prompt_type=prompt_type)

    if model_name == 'seed-llama-8b-sft-comm':
        model_name_or_path = MODEL_NAME_OR_PATH
    else:
        model_name_or_path = MODEL_DIR + model_name + "/final_merged"
    
    model = get_pretrained_llama_causal_model(
        pretrained_model_name_or_path = model_name_or_path,
        torch_dtype="bf16",
        low_cpu_mem_usage=True
    )

    # match the special tokens
    model.config.eos_token_id = model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = model.generation_config.pad_token_id = tokenizer.pad_token_id

    model = model.eval().to(device)

    generation_config = {
            'temperature': 1.0,
            'num_beams': 1,
            'max_new_tokens': 512,
            'top_p': 0.9,
            'do_sample': True
        }
    
    output_path = Path(RESULTS_DIR) / f"{model_name}/resq_test_output.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results if file exists
    if output_path.exists():
        with open(output_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    for q_idx, sample in enumerate(tqdm(test_dataset)):
        if str(q_idx) in results:
            print("Skipping", q_idx)
            continue
        
        # inference
        input_ids = tokenizer(sample['prompt'], add_special_tokens=False, return_tensors='pt').input_ids
        input_ids = input_ids.to(model.device)

        result_individual = []
        for _ in range(5):
            with torch.no_grad():
                generate_ids = model.generate(
                    input_ids=input_ids,
                    **generation_config
                )
                generate_ids = generate_ids[0][input_ids.shape[1]:]

            output_text = decode_image_text(generate_ids, tokenizer)
            predicted_answer = extract_answer(output_text)

            if predicted_answer == None:
                result_individual.append(None)
            else:
                match = normalize_answer(predicted_answer) == normalize_answer(sample['answer'])
                result_individual.append(match)
        
        # Save after each question
        results[str(q_idx)] = result_individual
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--prompt_type', type=str)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate(args.model_name, args.prompt_type, device)
