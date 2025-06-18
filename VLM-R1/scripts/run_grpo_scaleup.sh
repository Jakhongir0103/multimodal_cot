cd $REPO/src/open_r1/

RUN_NAME="Qwen2.5-VL-3B-GRPO-scaleup-acc_bbox_format"
MODEL_PATH="/lid/home/saydalie/multimodal_cot/LLaMA-Factory/output/qwen2_5_vl-3b/sft/aokvqa-bbox-full-open-ended/epoch_1"

export DEBUG_MODE="true"
export LOG_PATH="$BASE_LOG_PATH/debug_log_$RUN_NAME.txt"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# train
WORLD_SIZE=${WORLD_SIZE:-1}
WORLD_SIZE=1
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")

torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=${WORLD_SIZE} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    $REPO/src/open_r1/grpo_scaleup.py \
    --output_dir $OUT_PATH/$RUN_NAME \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_PATH \
    --num_generations 4 \
    --explanation_type bbox \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 0.05 \
    --save_total_limit 3 \
    --save_only_model false \
    --push_to_hub=false \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    --min_pixels 3136 \
    --max_pixels 3686400 \
    --reward_funcs accuracy bbox format \
    --deepspeed ${REPO}/src/open_r1/config/zero2.json \