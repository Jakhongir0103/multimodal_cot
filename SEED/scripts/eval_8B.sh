type="SEED_Llama_8B"
save_path="./outputs/${type}"


GPU_NUM=8
for i in $(seq 0 $((GPU_NUM-1))); do
    python -u scripts/eval_CoMM.py \
    --cur_gpu_id ${i} \
    --save_path ${save_path} \
    --config configs/llm/seed_llama8b_CoMM.yaml
done