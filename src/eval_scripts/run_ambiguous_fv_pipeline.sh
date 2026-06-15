#!/bin/bash
# FV + steering pipeline for the 4 ambiguous tasks (magnitude, identity, count_vowels,
# count_consonants). Single 24GB GPU fits ONE GPT-J fp16 instance, so GPU stages serialize;
# the 3 top-N FV builds run on CPU in parallel and overlap the GPU steering.
cd /workspace/function_vectors
export HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1
TASKS="magnitude identity count_vowels count_consonants"
mkdir -p results/_ambiguous_logs

echo "===== STAGE 1: task-specific FVs + mean activations + CIE (GPU) ====="
python src/compute_function_vectors.py --dataset_names $TASKS \
  --batch_size 16 --overwrite --continue_on_error \
  > results/_ambiguous_logs/stage1_taskspecific_fv.log 2>&1
echo "stage 1 exit=$?"

echo "===== STAGE 2 (CPU, parallel): build train-pooled-head FVs top-10/20/40 ====="
for N in 10 20 40; do
  python src/eval_scripts/compute_all_task_fvs_from_multitask_heads.py \
    --task_manifest task_splits/ambiguous_4.json --tasks $TASKS \
    --n_top_heads $N --device cpu \
    --output_root results/gptj_fv_multitask_top${N}_ambiguous \
    --manifest_name fv_manifest_ambiguous.json --overwrite \
    > results/_ambiguous_logs/stage2_build_top${N}.log 2>&1 &
done
BUILD_PIDS=$!

echo "===== STAGE 3 (GPU, serial): steering eval at n=10/20/40 ====="
for N in 10 20 40; do
  echo "--- steering n_top_heads=$N ---"
  python src/eval_scripts/evaluate_heldout_multitask_head_fvs.py \
    --tasks $TASKS --n_top_heads $N \
    --output_root results/heldout_ambiguous_eval_top${N} \
    > results/_ambiguous_logs/stage3_steer_top${N}.log 2>&1
  echo "steer n=$N exit=$?"
done

wait
echo "===== PIPELINE DONE ====="