[x] Filter out huge dataset from driving vqa train set
[x] Redo the data preparation of filtered train dataset for LLamaFactory
[x] Train SFT models on full dataset with/without bboxes
[x] Do inference with SFT models: are they generating bounding boxes next to an element?
[x] Evaluation code on Driving VQA test set.

[x] Check bounding boxes format
[x] Train SFT on full aokvqa checkpointing after each epoch

[x] Check scale up data for coco + opentext, ...
[x] Change bbox of data using smart_resize: `grpo_scaleup.py`/ `format_prompt_scaleup.py`
[x] Grpo Train: base_model=SFT_open_ended (1 epoch)

[x] Train SFT on A-OKVQA seed = {42, 32, 22}
[x] Evaluate on DrivingVQA and A-OKVQA

[x] Train GRPO on A-OKVQA seed = {42, 32, 22}
    - RUN_NAME, MODEL_PATH, --seed
[x] Evaluate on DrivingVQA and A-OKVQA

[x] Train SFT on A-OKVQA + VisCOT
[x] Evaluate on VisCOT

[x] LLM as a judge evaluation {DrivingVQA, A-OKVQA}

# Presentation
ResQ experiments:
- setup:
    - GRPO
    - dataset
    - model
- experiments
    - all ablations: reward; loss function; imag-text alignment variants

    - GRPO Variants: DPO, GRPO, Dr-GRPO
        seed-llama-8b-GRPO-resq-grpo
        seed-llama-8b-GRPO-resq-drgrpo
        seed-llama-8b-GRPO-resq-dapo
    - Rewards combinations:
        seed-llama-8b-GRPO-resq-dapo-txt (accuracy, format; textual baseline)
        seed-llama-8b-GRPO-resq-dapo (accuracy, format, image_count, image_alignment)
        seed-llama-8b-GRPO-resq-dapo-image_count_reward (accuracy, format, image_count)
        seed-llama-8b-GRPO-resq-dapo-no_image_reward (accuracy, format)
    - Image+text loss vs text-only loss
        seed-llama-8b-GRPO-resq-dapo-no_image_reward-text_loss_only
        seed-llama-8b-GRPO-resq-dapo-no_image_reward
    - Image alignment method
        seed-llama-8b-GRPO-resq-dapo
        seed-llama-8b-GRPO-resq-dapo-clip

- results:
    - loss plots
    - accuracies

Final conclusion: GPT-4 experiment and the final say

Next steps:
- Image understanding

## Appendix
PuzzleVQA experiment:
- setup:
    - dataset
    - model
- results