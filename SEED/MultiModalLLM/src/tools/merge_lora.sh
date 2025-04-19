lora_path="/lid/home/saydalie/multimodal_cot/models/SEED_trained/seed-llama-8b-GRPO-resq"
base_model="/lid/home/saydalie/multimodal_cot/SEED/checkpoints/seed-llama-8b-sft-comm"

python src/tools/merge_lora_weights.py \
  --model_cfg configs/model/seed_8b_lora_sfted.yaml \
  --tokenizer_cfg configs/tokenizer/seed_llama_tokenizer.yaml \
  --base_model ${base_model} \
  --lora_model ${lora_path}/final \
  --save_path ${lora_path}/final_merged 