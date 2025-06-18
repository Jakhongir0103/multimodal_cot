#!/bin/bash

base_model="/lid/home/saydalie/multimodal_cot/SEED/checkpoints/seed-llama-8b-sft-comm"

lora_paths=("seed-llama-8b-GRPO-resq-dapo-txt" "seed-llama-8b-GRPO-resq-dapo-no_image_reward-text_loss_only" "seed-llama-8b-GRPO-resq-dapo-no_image_reward" "seed-llama-8b-GRPO-resq-dapo-image_count_reward" "seed-llama-8b-GRPO-resq-dapo-clip" "seed-llama-8b-GRPO-resq-grpo" "seed-llama-8b-GRPO-resq-drgrpo" "seed-llama-8b-GRPO-resq-dapo")

for lora_path in "${lora_paths[@]}"; do
  echo "Merging ${lora_path}"

  python src/tools/merge_lora_weights.py \
    --model_cfg configs/model/seed_8b_lora_sfted.yaml \
    --tokenizer_cfg configs/tokenizer/seed_llama_tokenizer.yaml \
    --base_model ${base_model} \
    --lora_model /lid/home/saydalie/multimodal_cot/models/SEED_trained/${lora_path}/final \
    --save_path /lid/home/saydalie/multimodal_cot/models/SEED_trained/${lora_path}/final_merged
  
done