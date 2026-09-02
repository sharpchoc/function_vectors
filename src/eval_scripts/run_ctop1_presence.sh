#!/bin/bash
# GPU-pod launcher: carrier + v1 presence capture over all 69 tasks (capture_69_ctop1_presence.py).
set -u
REPO=/workspace/function_vectors
cd "$REPO" || exit 1
export FV_ARTIFACTS_ROOT=$REPO/artifacts
export PATH=/workspace/micromamba/envs/fv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$REPO/logs/ctop1
mkdir -p "$LOG"
rm -f "$LOG/presence.done" "$LOG/presence.failed"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
TASKS=$(python -c "import json;d=json.load(open('task_splits/extended_steerable_69_prunedfail.json'));print(' '.join(sorted(d['train_tasks']+d['heldout_tasks'])))")
date -u
python src/eval_scripts/capture_69_ctop1_presence.py --tasks $TASKS
rc=$?
date -u
if [ $rc -eq 0 ]; then touch "$LOG/presence.done"; else touch "$LOG/presence.failed"; fi
echo "exit $rc"
