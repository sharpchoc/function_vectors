#!/bin/bash
# Post constrained-extraction: build train-pooled FVs from the constrained mean activations,
# then steering eval restricted to DIFFERENTIATOR test queries. GPU-serial (24GB = 1 GPT-J).
ARTIFACTS="${FV_ARTIFACTS_ROOT:-artifacts}"; RESULTS="${FV_RESULTS_ROOT:-results}"; LOGS="${FV_LOGS_ROOT:-logs}"
cd /workspace/function_vectors
export HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1
TASKS="magnitude identity count_vowels count_consonants"
FVROOT="$ARTIFACTS/gptj_fv_ambiguous_constrained"
PARTNERS='{"magnitude":"identity","identity":"magnitude","count_vowels":"count_consonants","count_consonants":"count_vowels"}'

echo "===== STAGE A: train-pooled FVs from constrained mean activations (GPU serial) ====="
for N in 10 20 40; do
  python src/eval_scripts/compute_all_task_fvs_from_multitask_heads.py \
    --task_manifest task_splits/ambiguous_4.json --tasks $TASKS --n_top_heads $N --device cuda \
    --fv_root $FVROOT --output_root "$ARTIFACTS/gptj_fv_ambiguous_constrained_top${N}" \
    --manifest_name fv_manifest.json --overwrite \
    > "$LOGS/_ambiguous_logs/cstr_build_top${N}.log" 2>&1 ; echo "build $N exit=$?"
done

echo "===== STAGE B: differentiator-restricted steering eval (GPU serial) ====="
for N in 10 20 40; do
  python src/eval_scripts/evaluate_heldout_multitask_head_fvs.py \
    --tasks $TASKS --n_top_heads $N --fv_root $FVROOT \
    --restrict_differentiator --partners "$PARTNERS" \
    --output_root "$RESULTS/direction1_ambiguous/heldout_constrained_differ_top${N}" \
    > "$LOGS/_ambiguous_logs/cstr_steer_top${N}.log" 2>&1 ; echo "steer $N exit=$?"
done
echo "===== CONSTRAINED STEERING DONE ====="