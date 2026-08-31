#!/bin/bash
# GPU-pod launcher: 6-shot dummy steering with the fixed L6/7+top-1 vector at L1
# (peak of the 1-shot sweep, sweep_layer_summary.csv: L1 best=0.2259, L3 0.2235).
set -u
WT=/workspace/function_vectors/.claude/worktrees/fv-l5to7-top1
cd "$WT" || exit 1
export FV_ARTIFACTS_ROOT=/workspace/function_vectors/artifacts
export PATH=/workspace/micromamba/envs/fv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$WT/logs/l67top1_sweep
mkdir -p "$LOG"
rm -f "$LOG/sixshot.done" "$LOG/sixshot.failed"
MD=/workspace/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots/47e169305d2e8376be1d31e765533382721b2cc1
date -u
# token_budget/batch_cap reduced from the sixshot_dummy defaults (24000/48): OOM at the
# fp32 lm_head logits on the 32GB RTX PRO 4500 (first attempt 2026-08-31 19:47 UTC)
python src/sandbox/ext_steerability/sixshot_l67top1_steer.py --layer 1 --model_dir "$MD" \
  --token_budget 11000 --batch_cap 16
rc=$?
date -u
if [ $rc -eq 0 ]; then touch "$LOG/sixshot.done"; else touch "$LOG/sixshot.failed"; fi
echo "exit $rc"
