#!/bin/bash
# GPU chain for the mean-residual task-unique object (USER PROPOSAL 2026-09-02):
# ablation (6-shot + 1-shot, 4 conditions) with the unit direction u_hat, then 6-shot dummy
# steering with c + u_A at L0 and L5. Sequential on one pod.
set -u
REPO=/workspace/function_vectors
cd "$REPO" || exit 1
export FV_ARTIFACTS_ROOT=$REPO/artifacts
export PATH=/workspace/micromamba/envs/fv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$REPO/logs/meanresid
mkdir -p "$LOG"
rm -f "$LOG/chain.done" "$LOG/chain.failed"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
MD=/workspace/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots/47e169305d2e8376be1d31e765533382721b2cc1
BA=$FV_ARTIFACTS_ROOT/69_task_run/bottom_up_ablation/bankA
date -u
rc=0
for ns in 6 1; do
  python src/sandbox/ext_steerability/ablate_readdir_pc5.py --n_shots $ns \
    --bases_path "$BA/meanresid_top1_bases.pt" --out_sub bankA_meanresid_top1 --model_dir "$MD" || rc=1
done
for L in 0 5; do
  python src/sandbox/ext_steerability/sixshot_l67top1_steer.py --layer $L --model_dir "$MD" \
    --vectors_path "$BA/carrier_plus_meanresid_vectors.pt" \
    --out_root "$FV_ARTIFACTS_ROOT/69_task_run/meanresid_steering/sixshot_L$L" \
    --token_budget 11000 --batch_cap 16 || rc=1
done
date -u
if [ $rc -eq 0 ]; then touch "$LOG/chain.done"; else touch "$LOG/chain.failed"; fi
echo "exit $rc"
