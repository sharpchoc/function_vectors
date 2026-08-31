#!/bin/bash
# GPU-pod launcher: fixed-vector (L6/7 mean + top-1 dir) layer sweep, 1-shot dummy.
# Usage: run_l67top1_sweep.sh <shard_idx> <shard_n>
set -u
SHARD_IDX=${1:?shard_idx}
SHARD_N=${2:?shard_n}
WT=/workspace/function_vectors/.claude/worktrees/fv-l5to7-top1
cd "$WT" || exit 1
export FV_ARTIFACTS_ROOT=/workspace/function_vectors/artifacts
export PATH=/workspace/micromamba/envs/fv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$WT/logs/l67top1_sweep
mkdir -p "$LOG"
rm -f "$LOG/shard$SHARD_IDX.done" "$LOG/shard$SHARD_IDX.failed"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
MD=/workspace/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots/47e169305d2e8376be1d31e765533382721b2cc1
date -u
python src/sandbox/ext_steerability/sweep_l67top1_layers.py \
  --shard_idx "$SHARD_IDX" --shard_n "$SHARD_N" --model_dir "$MD"
rc=$?
date -u
if [ $rc -eq 0 ]; then touch "$LOG/shard$SHARD_IDX.done"; else touch "$LOG/shard$SHARD_IDX.failed"; fi
echo "exit $rc"
