#!/usr/bin/env bash
# Runner for steer_randlabel_swap.py (mean-removed 11-dir bases variant) on a GPU pod (fv env from the shared volume).
# usage: pod_run.sh <tag> [script args...]     e.g.  pod_run.sh pc5_n1 --n_shots 1
set -uo pipefail
REPO=/workspace/function_vectors/.claude/worktrees/lowdim-steering
LOGDIR=$REPO/logs/lowdim_steering
M=models--EleutherAI--gpt-j-6b
SNAP=47e169305d2e8376be1d31e765533382721b2cc1

# bare runpod/pytorch image has no rsync: copy the small metadata dirs, symlink blobs
mkdir -p /root/hf/hub/$M
cp -a /workspace/.cache/huggingface/hub/$M/snapshots /root/hf/hub/$M/ 2>/dev/null || true
cp -a /workspace/.cache/huggingface/hub/$M/refs /root/hf/hub/$M/ 2>/dev/null || true
ln -sfn /workspace/.cache/huggingface/hub/$M/blobs /root/hf/hub/$M/blobs
export HF_HOME=/root/hf
export HF_HUB_OFFLINE=1
export PATH=/workspace/micromamba/envs/fv/bin:$PATH
export FV_ARTIFACTS_ROOT=/workspace/function_vectors/artifacts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $REPO
TAG="$1"; shift
python src/sandbox/ext_steerability/steer_randlabel_swap.py \
  --model_dir /root/hf/hub/$M/snapshots/$SNAP "$@" > "$LOGDIR/$TAG.log" 2>&1
rc=$?
[ $rc -eq 0 ] && touch "$LOGDIR/$TAG.done" || touch "$LOGDIR/$TAG.failed"
exit $rc
