export REPO="/lid/home/saydalie/multimodal_cot/VLM-R1/"
export DATA_PATH="$REPO/data/"
export PYTHONPATH="$PYTHONPATH:$REPO/src/"

# vqav2
python -u $REPO/src/open_r1/eval/evaluate_bench.py \
    --model_path /lid/home/saydalie/multimodal_cot/VLM-R1/output/merged/Qwen2.5-VL-3B-GRPO-scaleup-acc_bbox_format/checkpoint-79768 \
    --questions_file $DATA_PATH/scaleup_eval/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl \
    --answers_file $DATA_PATH/scaleup_eval/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/llava-v1.5-13b/merge.jsonl \
    --image_folder $DATA_PATH/scaleup_eval/eval/vqav2/test2015 \
    --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen_3b_2_5_vl/grpo_scaleup/79768/vqav2

# vizwiz
python -u $REPO/src/open_r1/eval/evaluate_bench.py \
    --model_path /lid/home/saydalie/multimodal_cot/VLM-R1/output/merged/Qwen2.5-VL-3B-GRPO-scaleup-acc_bbox_format/checkpoint-79768 \
    --questions_file $DATA_PATH/scaleup_eval/eval/vizwiz/llava_test.jsonl \
    --answers_file $DATA_PATH/scaleup_eval/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --image_folder $DATA_PATH/scaleup_eval/eval/vizwiz/test \
    --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen_3b_2_5_vl/grpo_scaleup/79768/vizwiz

# textvqa
python -u $REPO/src/open_r1/eval/evaluate_bench.py \
    --model_path /lid/home/saydalie/multimodal_cot/VLM-R1/output/merged/Qwen2.5-VL-3B-GRPO-scaleup-acc_bbox_format/checkpoint-79768 \
    --questions_file $DATA_PATH/scaleup_eval/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --answers_file $DATA_PATH/scaleup_eval/eval/textvqa/answers/llava-v1.5-13b.jsonl \
    --image_folder $DATA_PATH/scaleup_eval/eval/textvqa/train_images \
    --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen_3b_2_5_vl/grpo_scaleup/79768/textvqa

# pope
python -u $REPO/src/open_r1/eval/evaluate_bench.py \
    --model_path /lid/home/saydalie/multimodal_cot/VLM-R1/output/merged/Qwen2.5-VL-3B-GRPO-scaleup-acc_bbox_format/checkpoint-79768 \
    --questions_file $DATA_PATH/scaleup_eval/eval/pope/llava_pope_test.jsonl \
    --answers_file $DATA_PATH/scaleup_eval/eval/pope/answers/llava-v1.5-13b.jsonl \
    --image_folder $DATA_PATH/scaleup_eval/eval/pope/val2014 \
    --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen_3b_2_5_vl/grpo_scaleup/79768/pope

# gqa
python -u $REPO/src/open_r1/eval/evaluate_bench.py \
    --model_path /lid/home/saydalie/multimodal_cot/VLM-R1/output/merged/Qwen2.5-VL-3B-GRPO-scaleup-acc_bbox_format/checkpoint-79768 \
    --questions_file $DATA_PATH/scaleup_eval/eval/gqa/llava_gqa_testdev_balanced.jsonl \
    --answers_file $DATA_PATH/scaleup_eval/eval/gqa/answers/llava_gqa_testdev_balanced/llava-v1.5-13b.jsonl \
    --image_folder $DATA_PATH/scaleup_eval/eval/gqa/images \
    --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen_3b_2_5_vl/grpo_scaleup/79768/gqa

# scienceqa
python -u $REPO/src/open_r1/eval/evaluate_sqa.py \
    --model_path /lid/home/saydalie/multimodal_cot/VLM-R1/output/merged/Qwen2.5-VL-3B-GRPO-scaleup-acc_bbox_format/checkpoint-79768 \
    --questions_file $DATA_PATH/scaleup_eval/eval/scienceqa/llava_test_CQM-A.json \
    --answers_file $DATA_PATH/scaleup_eval/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --image_folder $DATA_PATH/scaleup_eval/eval/scienceqa/test \
    --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen_3b_2_5_vl/grpo_scaleup/79768/scienceqa