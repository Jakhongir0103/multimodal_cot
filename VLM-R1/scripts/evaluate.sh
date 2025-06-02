export REPO="/lid/home/saydalie/multimodal_cot/VLM-R1/"
export DATA_PATH="$REPO/data/"
export PYTHONPATH="$PYTHONPATH:$REPO/src/"

python -u $REPO/src/open_r1/evaluate.py \
    --model_path $REPO/output/Qwen2.5-VL-7B-GRPO-aokvqa-1ep-bbox_rew/merged \
    --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen2_5_vl/grpo_aokvqa_1ep/acc_bbox/aokvqa/no_system \
    --input_data_dir $DATA_PATH \
    --dataset aokvqa \
    --explanation_type bbox

python -u $REPO/src/open_r1/evaluate.py \
    --model_path $REPO/output/Qwen2.5-VL-7B-GRPO-aokvqa-1ep-bbox_rew/merged \
    --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen2_5_vl/grpo_aokvqa_1ep/acc_bbox/drivingvqa/no_system \
    --input_data_dir $DATA_PATH \
    --dataset drivingvqa \
    --explanation_type bbox