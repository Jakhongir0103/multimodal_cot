# Data formatting for SFT-warmup model
python -u $REPO/src/open_r1/prepare_sft_data.py \
    --data_dir $DATA_PATH \
    --explanation_type bbox \
    --output_name aokvqa_bbox_full_open_ended \
    --dataset aokvqa \
    --seed 42

# Data formatting for SFT-no-reasoning model on VisCOT dataset
python -u $REPO/src/open_r1/prepare_sft_data_scaleup.py \
    --data_dir $DATA_PATH/scaleup \
    --output_name scaleup \
    --seed 42