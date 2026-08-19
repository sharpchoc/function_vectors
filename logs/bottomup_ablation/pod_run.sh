#!/usr/bin/env bash
# Runner for ablate_readdir_labeltokens.py on a GPU pod (fv env from the shared volume).
# usage: pod_run.sh <tag> [script args...]     e.g.  pod_run.sh n1 --n_shots 1
set -uo pipefail
REPO=/workspace/function_vectors/.claude/worktrees/bottomup-ablation
LOGDIR=$REPO/logs/bottomup_ablation
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

# canonical copy of the CF pairing next to the run outputs in the shared artifacts tree
mkdir -p /workspace/function_vectors/artifacts/69_task_run/bottom_up_ablation
cp -n $REPO/artifacts/69_task_run/bottom_up_ablation/cf_task_pairs.json \
      /workspace/function_vectors/artifacts/69_task_run/bottom_up_ablation/ 2>/dev/null || true

cd $REPO
TAG="$1"; shift
python src/sandbox/ext_steerability/ablate_readdir_labeltokens.py \
  --model_dir /root/hf/hub/$M/snapshots/$SNAP "$@" > "$LOGDIR/$TAG.log" 2>&1
rc=$?
[ $rc -eq 0 ] && touch "$LOGDIR/$TAG.done" || touch "$LOGDIR/$TAG.failed"
exit $rc
