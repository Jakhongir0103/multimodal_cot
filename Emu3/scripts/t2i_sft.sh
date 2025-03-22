#!/bin/bash

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")

export PYTHONPATH=$(pwd)

DATA_PATH="/lid/home/saydalie/multimodal_cot/Emu3/data/train/list/train.json"
OUTPUT_PATH="/lid/home/saydalie/multimodal_cot/models/Emu3_trained"
MODEL_PATH="/lid/home/saydalie/multimodal_cot/Emu3-models/Emu3-Stage1/snapshots/083e245ab5a4a3de4992d6da04238c96958ec141/"
EXP_NAME="Emu3-PuzzleVQA"

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    emu3/train/train.py \
    --model_name_or_path ${MODEL_PATH} \
    --deepspeed scripts/zero3.json \
    --data_path ${DATA_PATH} \
    --null_prompt_prob 0.05 \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_text False \
    --image_area 262144 \
    --max_position_embeddings 25000 \
    --output_dir ${OUTPUT_PATH}"/"${EXP_NAME} \
    --bf16 True \
    --tf32 False \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 0.05 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --min_learning_rate 1e-6 \
    --weight_decay 0.1 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --warmup_steps 30 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --logging_strategy steps \
    --logging_steps 0.01 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name ${EXP_NAME}