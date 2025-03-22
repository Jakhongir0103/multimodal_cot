import re
import os
import json
import pathlib
from types import MethodType

import torch

from datasets import Dataset
from peft import LoraConfig, get_peft_model

from trl import GRPOConfig
from trl import GRPOTrainer

import sys
sys.path.append("/lid/home/saydalie/multimodal_cot/SEED/")

from models.seed_llama_tokenizer import SeedLlamaTokenizer
from models.model_tools import get_pretrained_llama_causal_model

# BOI_TOKEN = '<img>'
# EOI_TOKEN = '</img>'
# IMG_TOKEN = '<img_{:05d}>'

user_token = "USER"
assistant_token = "ASSISTANT"

DATA_PATH = "/lid/home/saydalie/multimodal_cot/SEED/data/ReSQ/train_resq.json"
ENCODER_PATH = "/lid/home/saydalie/multimodal_cot/SEED/checkpoints/seed-tokenizer-2/seed_quantizer.pt"
MODEL_NAME_OR_PATH = "/lid/home/saydalie/multimodal_cot/SEED/checkpoints/seed-llama-8b-sft-comm"
DEEPSPEED_CONFIG = "/lid/home/saydalie/multimodal_cot/SEED/MultiModalLLM/configs/deepspeed/stage2_bf16.json"
OUTPUT_DIR = "/lid/home/saydalie/multimodal_cot/models/SEED_trained"

def _image_reward_one(completion, **kwargs):
    imgage_pattern = re.compile(r'<img>(.*?)</img>', re.IGNORECASE)
    imgage_matches = imgage_pattern.findall(completion)
    
    if not imgage_matches:
        # no valid <img>...</img> pairs
        return 0.0

    if completion.count("<img>") != completion.count("</img>"):
        # some <img>...</img> pairs are valid, but not all
        return 0.5

    num_invalid_images = 0
    for match in imgage_matches:
        tokens = match.strip().split()
        if len(tokens) != 32:
            num_invalid_images += 1
    
    if num_invalid_images == 0:
        # all are valid images
        return 1.0
        
    if len(imgage_matches) > num_invalid_images:
        # at least one invalid image
        return 0.5

    # all are invalid images
    return 0.0

def _image_reward_one_naive(completion, **kwargs):
    imgage_pattern = re.compile(r'<img>(.*?)</img>', re.IGNORECASE)
    imgage_matches = imgage_pattern.findall(completion)
    
    if not imgage_matches:
        # no valid <img>...</img> pairs
        return 0.0

    return 1.0

def image_reward(completions, **kwargs):
    """Reward function that checks if the completion has a valid images."""
    return [_image_reward_one_naive(completion) for completion in completions]

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r".*?<think>.*?</think>\s*<answer>.*?</answer>.*?"
    matches = [re.match(pattern, completion) for completion in completions]
    return [1.0 if match else 0.0 for match in matches]

def _normalize_answer(answer):
    """Normalizes an answer by stripping whitespace, converting to lowercase, and removing punctuation."""
    return re.sub(r'[^a-zA-Z0-9]', '', answer).lower()
    
def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    answers = kwargs["answer"]
    rewards = []
    for completion, correct_answer in zip(completions, answers):
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if match and _normalize_answer(match.group(1)) == _normalize_answer(correct_answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def load_tokenizer():
    tokenizer = SeedLlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path="AILab-CVC/seed-tokenizer-2",
        fp16=True,
        padding_side='left',
        load_diffusion=True,
        encoder_url=ENCODER_PATH,
        diffusion_path="stabilityai/stable-diffusion-2-1-unclip"
    )

    # Define a wrapper function that forces `skip_special_tokens=False`
    batch_decode = tokenizer.batch_decode
    def wrapped_batch_decode(self, *args, **kwargs):
        kwargs["skip_special_tokens"] = False
        return batch_decode(*args, **kwargs)
    tokenizer.batch_decode = MethodType(wrapped_batch_decode, tokenizer)

    return tokenizer

def load_dataset(bos_token):
    prompt="""{bos}{user_token}: I now describe a scene and ask a question about it. First, think about the reasoning process using an interleaved combination of images and text. You should generate an image when necessary to support reasoning, then describe insights from the image. Finally, provide with the answer. The final answer is either one of {candidate_answers}. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process with images and text here </think> <answer> the final answer here </answer>.
    {question}
    {assistant_token}:"""

    with open(DATA_PATH, "r") as file:
        data = json.load(file)['data']
        
    train_dataset = []

    for d in data:
        story = d['story']
        questions = d['questions']
        
        for q in questions:
            question = q['question']
            answer = q['answer'][0]
            candidate_answers = q['candidate_answers']
            num_1st_context_sentences = q['num_1st_context_sentences']

            scene = ' '.join(story[:num_1st_context_sentences])
            train_dataset.append({
                'prompt': prompt.format(
                    bos=bos_token,
                    user_token=user_token,
                    assistant_token=assistant_token,
                    candidate_answers=candidate_answers,
                    question=f"{scene} {question}"),
                'answer': answer
            })


    train_dataset = Dataset.from_list(train_dataset)
    return train_dataset

def train():
    tokenizer = load_tokenizer()
    train_dataset = load_dataset(tokenizer.bos_token)

    model = get_pretrained_llama_causal_model(
        pretrained_model_name_or_path=MODEL_NAME_OR_PATH,
        torch_dtype="bf16",
        low_cpu_mem_usage=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        task_type='CAUSAL_LM',
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
        modules_to_save=['embed_tokens', 'lm_head', 'input_layernorm', 'post_attention_layernorm', 'norm']
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Configure training arguments using GRPOConfig
    # asser (num_gpus * batch_size) % num_generations == 0
    run_name = "seed-llama-8b-GRPO-resq"
    output_dir = os.path.join(OUTPUT_DIR, run_name)

    os.environ["WANDB_RUN_ID"] = run_name
    os.environ["WANDB_RESUME"] = "allow"

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=10,
        learning_rate=5e-7,
        remove_unused_columns=False,  # to access the `answer` column in accuracy_reward
        eval_strategy="no",
        weight_decay=5e-2,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        torch_empty_cache_steps=1,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-5,
        deepspeed=DEEPSPEED_CONFIG,
        # # Inference Optimization
        # Parameters that control the data preprocessing
        temperature=0.9,
        max_completion_length=1024,
        num_generations=2,
        max_prompt_length=512,
        # Parameters related to reporting and saving
        save_strategy="steps",
        save_steps=0.05,
        logging_strategy='steps',
        logging_steps=1,
        log_level='warning',
        logging_nan_inf_filter="no",
        push_to_hub=False,
        report_to="wandb",
        run_name=run_name
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, accuracy_reward, image_reward],
        args=training_args,
        train_dataset=train_dataset
    )

    if list(pathlib.Path(output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    torch.cuda.synchronize()
    trainer.save_model(os.path.join(output_dir, 'final'))

if __name__ == "__main__":
    train()