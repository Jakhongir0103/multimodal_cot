export DATA_PATH="/lid/home/saydalie/multimodal_cot/VLM-R1/data/"
export OUT_PATH="/lid/home/saydalie/multimodal_cot/VLM-R1/output/"
export REPO="/lid/home/saydalie/multimodal_cot/VLM-R1/"
export BASE_LOG_PATH="$REPO/logs"
export PYTHONPATH="$PYTHONPATH:$REPO/src/"

export WANDB_API_KEY="625a24b8d51739a2c2ed657050c26b7c14b5fd9a"
export WANDB_ENTITY="jakhongir-saydaliev-epfl"
export WANDB_PROJECT="multimodal_cot"
export HF_TOKEN="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"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# train
WORLD_SIZE=${WORLD_SIZE:-1}
WORLD_SIZE=1
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=$(python -c "import torch; print(torch.cuda.device_count())") # keep 1 gpu for vllm

RUN_NAME="Qwen2.5-VL-7B-SFT-bbox"

# torchrun \
#     --nproc_per_node=${NGPUS} \
#     --nnodes=${WORLD_SIZE} \
#     --node_rank=${RANK} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \

python -u \
    $REPO/src/open_r1/sft.py \
    --output_dir $OUT_PATH/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name $DATA_PATH \
    --explanation_type bbox \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --seed 42 \
    --report_to none \
    --gradient_checkpointing true \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_strategy steps \
    --save_steps 0.05 \
    --save_total_limit 2 \
    --save_only_model false \
    --push_to_hub=false \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM

    # --attn_implementation flash_attention_2 \