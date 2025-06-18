#!/bin/bash
export WANDB_API_KEY="625a24b8d51739a2c2ed657050c26b7c14b5fd9a"
export WANDB_ENTITY="jakhongir-saydaliev-epfl"
export WANDB_PROJECT="multimodal_cot"

export HF_TOKEN="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"

export BASE_PATH="/lid/home/saydalie"

cd $BASE_PATH/multimodal_cot/anole/training

torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=4 \
    train_txt.py \
    > "$BASE_PATH/_runai_out/train_txt.txt" 2>&1