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

from datasets import Dataset
from peft import LoraConfig, get_peft_model

from trl import GRPOConfig, GRPOTrainer
from trainer.grpo_trainer import GRPOTrainerCustom

import sys
sys.path.append("/lid/home/saydalie/multimodal_cot/SEED/")

from models.seed_llama_tokenizer import SeedLlamaTokenizer
from models.model_tools import get_pretrained_llama_causal_model

from rewards import image_clip, image_count_reward, format_reward, accuracy_reward

user_token = "USER"
assistant_token = "ASSISTANT"

DATA_PATH = "/lid/home/saydalie/multimodal_cot/SEED/data/ReSQ/train_resq.json"
MODEL_NAME_OR_PATH = "/lid/home/saydalie/multimodal_cot/SEED/checkpoints/seed-llama-8b-sft-comm/"
DEEPSPEED_CONFIG = "/lid/home/saydalie/multimodal_cot/SEED/MultiModalLLM/configs/deepspeed/stage2_bf16.json"
OUTPUT_DIR = "/lid/home/saydalie/multimodal_cot/models/SEED_trained"

def load_tokenizer(device):
    tokenizer = SeedLlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME_OR_PATH,
        fp16=True,
        padding_side='left',
        device=device,
        load_diffusion=False
    )

    # removes <img> and </img>, as we need them in reward calculation
    tokenizer.additional_special_tokens = []

    return tokenizer

PROMPT = """{bos}{user_token}: I now describe a scene and ask a question about it. First, generate an image that helps visualize the scene described. Then, think about the reasoning process in the mind and then provide with the final answer. The final answer is either one of {candidate_answers}. The image and reasoning process are enclosed within <think> </think> tags, while the final answer is enclosed within <answer> </answer> tags, i.e., <think> image and the textual reasoning process here </think> <answer> the final answer here </answer>.
{question}
{assistant_token}:"""

# PROMPT = """{bos}{user_token}: I now describe a scene and ask a question about it. First, think about the reasoning process in the mind and then provide with the final answer. The final answer is either one of {candidate_answers}. The reasoning process and the final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> the final answer here </answer>.
# {question}
# {assistant_token}:"""

def load_dataset(bos_token):
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
                'prompt': PROMPT.format(
                    bos=bos_token,
                    user_token=user_token,
                    assistant_token=assistant_token,
                    candidate_answers=candidate_answers,
                    question=f"{scene} {question}"),
                'scene': scene,
                'answer': answer
            })

    train_dataset = Dataset.from_list(train_dataset)
    return train_dataset

def get_device_name() -> str:
    """
    Returns the name of the device where this module is running on.
    """
    if torch.cuda.is_available():
        if torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return f"cuda:{local_rank}"
    return "cpu"
    
def train(args):
    device = get_device_name()
    tokenizer = load_tokenizer(device)
    train_dataset = load_dataset(tokenizer.bos_token)
    # train_dataset = train_dataset.map(lambda sample: {'scene_ids': tokenizer.encode(sample['scene'])})

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
        reward_weights=[1.0, 1.0, 6.0, 6.0],
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
        reward_funcs=[image_clip, image_count_reward, format_reward, accuracy_reward],
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
    args = parser.parse_args()

    train(args)