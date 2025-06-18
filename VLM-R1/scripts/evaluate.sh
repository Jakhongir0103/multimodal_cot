# SFT
for seed in 42 32 22; do
    python -u $REPO/src/open_r1/eval/evaluate.py \
        --model_path /lid/home/saydalie/multimodal_cot/LLaMA-Factory/output/qwen2_5_vl-7b/sft/aokvqa-bbox-full-seed_$seed \
        --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen2_5_vl/sft_aokvqa/$seed/aokvqa/no_system \
        --input_data_dir $DATA_PATH \
        --dataset aokvqa \
        --explanation_type bbox

    python -u $REPO/src/open_r1/eval/evaluate.py \
        --model_path /lid/home/saydalie/multimodal_cot/LLaMA-Factory/output/qwen2_5_vl-7b/sft/aokvqa-bbox-full-seed_$seed \
        --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen2_5_vl/sft_aokvqa/$seed/drivingvqa/no_system \
        --input_data_dir $DATA_PATH \
        --dataset drivingvqa \
        --explanation_type bbox
done

# GRPO
for seed in 42 32 22; do
    python -u $REPO/src/open_r1/eval/evaluate.py \
        --model_path $REPO/output/Qwen2.5-VL-7B-GRPO-aokvqa-acc_bbox_format-seed_$seed/merged \
        --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen2_5_vl/grpo_aokvqa/$seed/acc_bbox/aokvqa/no_system \
        --input_data_dir $DATA_PATH \
        --dataset aokvqa \
        --explanation_type bbox

    python -u $REPO/src/open_r1/eval/evaluate.py \
        --model_path $REPO/output/Qwen2.5-VL-7B-GRPO-aokvqa-acc_bbox_format-seed_$seed/merged \
        --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen2_5_vl/grpo_aokvqa/$seed/acc_bbox/drivingvqa/no_system \
        --input_data_dir $DATA_PATH \
        --dataset drivingvqa \
        --explanation_type bbox
done