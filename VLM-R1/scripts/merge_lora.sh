export REPO="/lid/home/saydalie/multimodal_cot/VLM-R1/"
export OUT_PATH="${REPO}/output"

cd $REPO/src/open_r1/

BASE_MODEL_PATH="/lid/home/saydalie/multimodal_cot/LLaMA-Factory/output/qwen2_5_vl-3b/sft/aokvqa-bbox-full-open-ended/epoch_1"
RUN_NAME="Qwen2.5-VL-3B-GRPO-scaleup-acc_bbox_format"

python -u merge_lora.py \
    --baseline_model_path $BASE_MODEL_PATH \
    --adapter_model_path $OUT_PATH/$RUN_NAME/checkpoint-79768 \
    --output_path $OUT_PATH/merged/$RUN_NAME/checkpoint-79768