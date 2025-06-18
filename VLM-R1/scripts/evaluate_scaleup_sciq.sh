python -u $REPO/src/open_r1/eval/evaluate_scaleup_sciq.py \
    --model_path /lid/home/saydalie/multimodal_cot/VLM-R1/output/merged/Qwen2.5-VL-3B-GRPO-scaleup-acc_bbox_format/checkpoint-116584 \
    --questions_file $DATA_PATH/scaleup_eval/eval/scienceqa/llava_test_CQM-A.json \
    --answers_file $DATA_PATH/scaleup_eval/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --image_folder $DATA_PATH/scaleup_eval/eval/scienceqa/test \
    --output_data_dir /lid/home/saydalie/multimodal_cot/results/qwen_3b_2_5_vl/grpo_scaleup/116584/scienceqa