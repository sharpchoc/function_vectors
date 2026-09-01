#!/bin/bash
# Bank-A 6-shot dummy steering with w_A at L1 (peak of the bankA 1-shot sweep, 0.2197).
set -u
WT=/workspace/function_vectors/.claude/worktrees/fv-l5to7-top1
cd "$WT" || exit 1
export FV_ARTIFACTS_ROOT=/workspace/function_vectors/artifacts
export PATH=/workspace/micromamba/envs/fv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$WT/logs/bankA
mkdir -p "$LOG"
rm -f "$LOG/sixshot.done" "$LOG/sixshot.failed"
MD=/workspace/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots/47e169305d2e8376be1d31e765533382721b2cc1
BA=$FV_ARTIFACTS_ROOT/69_task_run/bottom_up_ablation/bankA
date -u
python src/sandbox/ext_steerability/sixshot_l67top1_steer.py --layer 1 --model_dir "$MD" \
  --vectors_path "$BA/l67_plus_top1_vectors.pt" \
  --out_root "$FV_ARTIFACTS_ROOT/69_task_run/l67top1_steering_bankA/sixshot" \
  --token_budget 11000 --batch_cap 16
rc=$?
date -u
if [ $rc -eq 0 ]; then touch "$LOG/sixshot.done"; else touch "$LOG/sixshot.failed"; fi
echo "exit $rc"
