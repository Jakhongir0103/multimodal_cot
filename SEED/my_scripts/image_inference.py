import hydra
import re
import pyrootutils
import os
from tqdm import tqdm
import torch

from omegaconf import OmegaConf
import json
from typing import Optional
import transformers
from PIL import Image
import pandas as pd

from torchvision.transforms.functional import InterpolationMode
from transformers.generation.configuration_utils import GenerationConfig

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

user_token = "USER"
assistant_token = "ASSISTANT"

DATA_DIR_PATH = "/lid/home/saydalie/multimodal_cot/SEED/data/ReSQ/"
MODEL_NAME_OR_PATH = "/lid/home/saydalie/multimodal_cot/SEED/checkpoints/seed-llama-8b-sft-comm/"

import sys
sys.path.append("/lid/home/saydalie/multimodal_cot/SEED/")

from models.seed_llama_tokenizer import SeedLlamaTokenizer
from models.transforms import get_transform
from models.model_tools import get_pretrained_llama_causal_model

def generate(tokenizer, input_tokens, generation_config, model):
    """Only for batch_size=1"""

    input_ids = tokenizer(input_tokens, add_special_tokens=False, return_tensors='pt').input_ids
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        generate_ids = model.generate(
            input_ids=input_ids,
            generation_config=generation_config
        )
    generate_ids = generate_ids[0][input_ids.shape[1]:]
    
    return generate_ids

def replace_img_tags(input_text):
    img_pattern = re.compile(r'<img>(.*?)</img>', re.IGNORECASE)
    img_matches = img_pattern.findall(input_text)
    
    for i, match in enumerate(img_matches):
        replacement = f'<IMAGE>'
        input_text = input_text.replace(f'<img>{match}</img>', replacement)
    
    return input_text

def load_tokenizer():
    tokenizer = SeedLlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME_OR_PATH,
        fp16=True,
        padding_side='left',
        load_diffusion=False
    )

    return tokenizer

transform = get_transform(type='clip', keep_ratio=True, image_size=224)

PROMPT="""{bos}{user_token}: {image_tokens} {question} The answer is either one of {candidate_answers}.
{assistant_token}:"""

def load_dataset(tokenizer):
    data_resq_clean = pd.read_csv(DATA_DIR_PATH+"train_resq_clean.csv")
    data_resq_images = pd.read_csv(DATA_DIR_PATH+"train_resq_images_paths.csv")

    img_tokens_all = {0: [], 1: [], 2: []}
    for _, row in tqdm(data_resq_images.iterrows(), desc='Encoding images', total=data_resq_images.shape[0]):
        for image_idx in range(3):
            image_path = row[f'image_{image_idx}']
            if not image_path.startswith("/lid/home/saydalie/multimodal_cot/SEED/data/ReSQ/images/"):
                img_tokens_all[image_idx].append(None)
                continue

            # encode image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).to(tokenizer.device)
            img_ids = tokenizer.encode_image(image_torch=image_tensor)
            img_ids = img_ids.view(-1).cpu().numpy()
            img_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(item) for item in img_ids]) + EOI_TOKEN            

            img_tokens_all[image_idx].append(img_tokens)

    data_resq_images['img_tokens_0'] = img_tokens_all[0]
    data_resq_images['img_tokens_1'] = img_tokens_all[1]
    data_resq_clean['img_tokens_2'] = img_tokens_all[2]

    data = pd.merge(data_resq_clean, data_resq_images, on='index', how='left')
    return data

generation_config = GenerationConfig(
    temperature=0.9,
    num_beams=1,
    max_new_tokens=128,
    top_p=1.0,
    top_k=50,
    do_sample=True
)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = load_tokenizer()
    data = load_dataset(tokenizer=tokenizer)

    model = get_pretrained_llama_causal_model(
        pretrained_model_name_or_path=MODEL_NAME_OR_PATH,
        torch_dtype="bf16",
        low_cpu_mem_usage=True
    ).to(device)

    # match the special tokens
    model.config.eos_token_id = model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = model.generation_config.pad_token_id = tokenizer.pad_token_id

    output_texts = {0: [], 1: [], 2: []}
    for _, row in tqdm(data.iterrows(), desc='Generating', total=data.shape[0]):
        candidate_answers = row['candidate_answers']
        question = row['question']

        for image_idx in range(3):
            if row[f'img_tokens_{image_idx}'] is None:
                output_texts[image_idx].append(None)
                continue

            input_tokens = PROMPT.format(
                bos=tokenizer.bos_token,
                user_token=user_token,
                assistant_token=assistant_token,
                candidate_answers=candidate_answers,
                question=question,
                image_tokens=row[f'img_tokens_{image_idx}']
            )
            generate_ids = generate(tokenizer, input_tokens, generation_config, model)
            output_text = tokenizer.batch_decode(generate_ids.unsqueeze(0), skip_special_tokens=False)[0]
            output_texts[image_idx].append(output_text)

    data['output_text_0'] = output_texts[0]
    data['output_text_1'] = output_texts[1]
    data['output_text_2'] = output_texts[2]

    data.to_csv(DATA_DIR_PATH + 'train_resq_images_model_output.csv', index=False)

