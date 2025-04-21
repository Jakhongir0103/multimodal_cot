# replace with your WandB details

export DATA_PATH="/lid/home/saydalie/multimodal_cot/VLM-R1/data/"
export OUT_PATH="/lid/home/saydalie/multimodal_cot/VLM-R1/output/"
export REPO="/lid/home/saydalie/multimodal_cot/VLM-R1/"
export BASE_LOG_PATH="$REPO/logs"

export WANDB_API_KEY="625a24b8d51739a2c2ed657050c26b7c14b5fd9a"
export WANDB_ENTITY="jakhongir-saydaliev-epfl"
export WANDB_PROJECT="multimodal_cot"

export HF_TOKEN="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"

export PYTHONPATH="$PYTHONPATH:$REPO/src/"

cd $REPO/src/open_r1/

RUN_NAME="Qwen2.5-VL-7B-SFT-lora"

python -u sft.py \
    --output_dir $OUT_PATH/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name $DATA_PATH \
    --explanation_type bbox \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --seed 42 \
    --report_to none \
    --gradient_checkpointing true \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 0.05 \
    --save_total_limit 3 \
    --save_only_model false \
    --push_to_hub=false \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \