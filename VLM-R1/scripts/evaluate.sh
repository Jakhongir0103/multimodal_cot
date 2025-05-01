export DATA_PATH="/lid/home/saydalie/multimodal_cot/VLM-R1/data/"
export REPO="/lid/home/saydalie/multimodal_cot/VLM-R1/"
export PYTHONPATH="$PYTHONPATH:$REPO/src/"

python -u $REPO/src/open_r1/evaluate.py \
    --model_path /lid/home/saydalie/multimodal_cot/LLaMA-Factory/output/qwen2_5_vl-7b/sft_incorrect/bbox \
    --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen2_5_vl/sft_incorrect/bbox \
    --input_data_dir $DATA_PATH \
    --explanation_type bbox