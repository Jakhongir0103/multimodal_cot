"""
TODO:
1. completion_mask not working -- generation_config
2. completion texts, if they are meaningful
3. check if image_text_alignment function is working
"""

import os
import json
import pathlib
import argparse

import torch

from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from trl import GRPOConfig, GRPOTrainer
from trainer.grpo_trainer import GRPOTrainerCustom

import sys
sys.path.append("/lid/home/saydalie/multimodal_cot/SEED/")

from models.seed_llama_tokenizer import SeedLlamaTokenizer
from models.model_tools import get_pretrained_llama_causal_model

from rewards import image_text_alignment_reward, image_count_reward, format_reward, accuracy_reward

user_token = "USER"
assistant_token = "ASSISTANT"

MODEL_NAME_OR_PATH = "/lid/home/saydalie/multimodal_cot/SEED/checkpoints/seed-llama-8b-sft-comm/"
DEEPSPEED_CONFIG = "/lid/home/saydalie/multimodal_cot/SEED/MultiModalLLM/configs/deepspeed/stage2_bf16.json"
OUTPUT_DIR = "/lid/home/saydalie/multimodal_cot/models/SEED_trained"

def load_tokenizer():
    tokenizer = SeedLlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME_OR_PATH,
        fp16=True,
        padding_side='left',
        load_diffusion=False
    )

    # removes <img> and </img>, as we need them in reward calculation
    tokenizer.additional_special_tokens = []

    return tokenizer

PROMPT_IMG = """{bos}{user_token}: I describe you my goal, and give you 2 solutions to achieve it. Generate images to visualize the 2 described solutions, and reason over them with text to figure out which solution achieves the goal. Then, provide with the final answer as 0 or 1, referring to solution 1 or solution 2 respectively. So, the final answer is either "0" or "1". The image and reasoning process are enclosed within <think> </think> tags, while the final answer is enclosed within <answer> </answer> tags, i.e., <think> image and the textual reasoning process here </think> <answer> the final answer here </answer>.
Goal: {goal}
Solution 1: {sol1}
Solution 2: {sol2}
{assistant_token}:"""

PROMPT_TXT = """{bos}{user_token}: I describe you my goal, and give you 2 solutions to achieve it. Reason over the 2 offered solutions to figure out which one achieves the goal. Then, provide with the final answer as 0 or 1, referring to solution 1 or solution 2 respectively. So, the final answer is either "0" or "1". The reasoning process and the final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> the final answer here </answer>.
Goal: {goal}
Solution 1: {sol1}
Solution 2: {sol2}
{assistant_token}:"""

def load_train_dataset(bos_token, text_only_reasoning=False):
    data = load_dataset('ybisk/piqa', split='train', trust_remote_code=True)    # {"goal": goal, "sol1": sol1, "sol2": sol2, "label": lab}
    data = data.select(range(1000))

    print(data)
    print(data[0])

    # columns after map: ["prompt", "answer"]
    if text_only_reasoning:
        data = data.map(lambda sample: {'prompt': PROMPT_TXT.format(bos=bos_token, user_token=user_token, assistant_token=assistant_token, goal=sample['goal'], sol1=sample['sol1'], sol2=sample['sol2']), 'answer': str(sample['label'])}, remove_columns=["goal", "sol1", "sol2", "label"])
    else:
        data = data.map(lambda sample: {'prompt': PROMPT_IMG.format(bos=bos_token, user_token=user_token, assistant_token=assistant_token, goal=sample['goal'], sol1=sample['sol1'], sol2=sample['sol2']), 'answer': str(sample['label'])}, remove_columns=["goal", "sol1", "sol2", "label"])

    print(data)
    print(data[0])

    return data

def train(args):
    tokenizer = load_tokenizer()
    train_dataset = load_train_dataset(tokenizer.bos_token, args.text_only_reasoning)

    model = get_pretrained_llama_causal_model(
        pretrained_model_name_or_path=MODEL_NAME_OR_PATH,
        torch_dtype="bf16",
        low_cpu_mem_usage=True
    )
    model.apply_loss_on_text_only = args.apply_loss_on_text_only

    # match the special tokens
    model.config.eos_token_id = model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = model.generation_config.pad_token_id = tokenizer.pad_token_id

    # modules_to_save=['embed_tokens', 'lm_head', 'input_layernorm', 'post_attention_layernorm', 'norm']
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        task_type='CAUSAL_LM',
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Configure training arguments using GRPOConfig
    # asser (num_gpus * batch_size) % num_generations == 0
    output_dir = os.path.join(OUTPUT_DIR, args.run_name)

    os.environ["WANDB_RUN_ID"] = args.run_name
    os.environ["WANDB_RESUME"] = "allow"

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=10,
        learning_rate=5e-7,
        remove_unused_columns=False,
        eval_strategy="no",
        weight_decay=5e-2,
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        torch_empty_cache_steps=1,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-5,
        deepspeed=DEEPSPEED_CONFIG,
        # # Inference Optimization - TODO: fix the vllm issues
        # use_vllm=True,
        # vllm_max_model_len=512,
        # vllm_device='auto',
        # vllm_dtype='auto',
        # Parameters that control the data preprocessing
        temperature=1.0,
        max_completion_length=512,
        num_generations=8,
        max_prompt_length=256,
        beta=0.04,
        # Grpo specific
        # reward_weights=[1.0, 1.0, 2.0, 2.0],
        # Parameters related to reporting and saving
        save_strategy="steps",
        save_steps=0.05,
        save_total_limit=2,
        logging_strategy='steps',
        logging_steps=1,
        log_level='warning',
        logging_nan_inf_filter="no",
        push_to_hub=False,
        report_to="wandb",
        run_name=args.run_name
    )

    trainer = GRPOTrainerCustom(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards
    )

    if list(pathlib.Path(output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    torch.cuda.synchronize()
    trainer.save_model(os.path.join(output_dir, 'final'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="seed-llama-8b-GRPO")
    parser.add_argument('--loss_type', type=str, default="grpo")
    parser.add_argument('--scale_rewards', action="store_true")
    parser.add_argument('--apply_loss_on_text_only', action="store_true")
    parser.add_argument('--text_only_reasoning', action="store_true")
    args = parser.parse_args()

    train(args)