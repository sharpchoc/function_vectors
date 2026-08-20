#!/bin/bash
# FV presence above the generic-FV baseline: GPU-pod launcher for capture_69_presence_gm.py
# (all 69 tasks, single forward-only process). Run from a pod that mounts the shared
# /workspace. Writes artifacts/69_task_run/presence_vs_acc_gm/<task>.npz in the MAIN
# checkout's artifacts dir (FV_ARTIFACTS_ROOT) and logs/presence_gm/run.{done,failed}
# in the worktree.
set -u
WT=/workspace/function_vectors/.claude/worktrees/agent-a873fed4d84e79f82
cd "$WT" || exit 1
export FV_ARTIFACTS_ROOT=/workspace/function_vectors/artifacts
export PATH=/workspace/micromamba/envs/fv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$WT/logs/presence_gm
mkdir -p "$LOG"
rm -f "$LOG/run.done" "$LOG/run.failed"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
TASKS=$(python -c "import json;d=json.load(open('task_splits/extended_steerable_69_prunedfail.json'));print(' '.join(sorted(d['train_tasks']+d['heldout_tasks'])))")
date -u
python src/eval_scripts/capture_69_presence_gm.py --tasks $TASKS
rc=$?
date -u
if [ $rc -eq 0 ]; then touch "$LOG/run.done"; else touch "$LOG/run.failed"; fi
echo "exit $rc"
