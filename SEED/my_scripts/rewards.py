import re
import math

from sentence_transformers import SentenceTransformer, util
from models.seed_llama_tokenizer import SeedLlamaTokenizer

import sys
sys.path.append("/lid/home/saydalie/multimodal_cot/SEED/")

MODEL_NAME_OR_PATH = "/lid/home/saydalie/multimodal_cot/SEED/checkpoints/seed-llama-8b-sft-comm/"
IMAGE_ID_SHIFT = 32000

# Load the CLIP model
model = SentenceTransformer('clip-ViT-L-14')

# Load the tokenizer
tokenizer = SeedLlamaTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME_OR_PATH,
    fp16=True,
    padding_side='left',
    load_diffusion=True,
    diffusion_path="stabilityai/stable-diffusion-2-1-unclip",
    device=model.device
)

def _rescale(num):
    if num <= 0.2:
        return 0.0
    elif num >= 0.4:
        return 1.0
    else:
        # linear scaling between 0.2 and 0.4
        return (num - 0.2) / (0.4 - 0.2)

def image_clip(completions, **kwargs):
    """Reward function for text and first image alignment using cosine similarity."""
    scene_batch = kwargs["scene"]

    image_pattern = re.compile(r'<img>(.*?)</img>', re.IGNORECASE)
    rewards = []
    for completion, text in zip(completions, scene_batch):
        image_match = image_pattern.search(completion)

        # keep only the first image
        image_tokens = image_match.group(1).strip().split() if image_match else []

        # image should be of 32 tokens with each token in the format of `<img_{:05d}>`
        if len(image_tokens) == 32 and all([re.match(r'^<img_\d{5}>$', token) for token in image_tokens]):
            image_ids = (tokenizer(''.join(image_tokens), return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)['input_ids'] - IMAGE_ID_SHIFT).reshape(1,-1)
            image = tokenizer.decode_image(image_ids)[0]

            img_emb = model.encode(image)
            text_emb = model.encode(text)

            # compute cosine similarity and rescale
            cos_scores = util.cos_sim(img_emb, text_emb)
            rewards.append(_rescale(cos_scores.item()))
        else:
            rewards.append(0.0)

    return rewards

def image_count_reward(completions, **kwargs):
    """Reward function that checks the number of images."""
    image_pattern = re.compile(r'<img>(.*?)</img>', re.IGNORECASE)
    rewards = []
    for completion in completions:
        image_matches = image_pattern.findall(completion)
        image_nums = len(image_matches)

        if image_nums==0 or completion.count("<img>") != completion.count("</img>"):
            # no image or invalid image tokens
            rewards.append(0.0)
        elif image_nums==1:
            rewards.append(1.0)
        else:
            # exponential reward decay as number of images increases
            rewards.append(math.exp(-(image_nums - 1)))
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern_strict = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    matches_strict = [re.match(pattern_strict, completion.strip()) for completion in completions]

    pattern_loose = r"^[\s.,;:!?\"'()\-]*<think>.*?</think>\s*<answer>.*?</answer>[\s.,;:!?\"'()\-]*$"
    matches_loose = [re.match(pattern_loose, completion.strip()) for completion in completions]

    rewards = []
    for ms, ml in zip(matches_strict, matches_loose):
        if ms:
            rewards.append(1.0)
        elif ml:
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards

def _normalize_answer(answer):
    """Normalizes an answer by stripping whitespace, converting to lowercase, and removing punctuation."""
    return re.sub(r'[^a-zA-Z0-9]', '', answer).lower()
    
def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    print("--"*20)
    answers = kwargs["answer"]
    rewards = []
    for completion, correct_answer in zip(completions, answers):
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)

        if match and _normalize_answer(match.group(1)) == _normalize_answer(correct_answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    print(f"{rewards[-1]}: {completions[-1]}")
    print()
    return rewards