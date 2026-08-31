#!/bin/bash
# GPU-pod launcher: L5-7 top-1 task-unique ablation (USER REQUEST 2026-08-31).
# Runs the standard 4-condition ablation (ablate_readdir_pc5.py) with the
# meanremoved_L5to7_top1 bases, 6-shot then 1-shot. Run from a pod mounting /workspace.
# Outputs -> MAIN checkout artifacts (FV_ARTIFACTS_ROOT):
#   artifacts/69_task_run/bottom_up_ablation/meanremoved_L5to7_top1/n{6,1}shot/<task>.json
set -u
WT=/workspace/function_vectors/.claude/worktrees/fv-l5to7-top1
cd "$WT" || exit 1
export FV_ARTIFACTS_ROOT=/workspace/function_vectors/artifacts
export PATH=/workspace/micromamba/envs/fv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$WT/logs/l5to7_top1
mkdir -p "$LOG"
rm -f "$LOG/run.done" "$LOG/run.failed"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
BASES=$FV_ARTIFACTS_ROOT/69_task_run/bottom_up_ablation/meanremoved_L5to7_top1_bases.pt
# explicit COMPLETE snapshot: load_model_eager's sorted(glob)[-1] default picks the
# weights-only f3f42882 snapshot (no config.json) and crashes
MD=/workspace/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots/47e169305d2e8376be1d31e765533382721b2cc1
date -u
python src/sandbox/ext_steerability/ablate_readdir_pc5.py --n_shots 6 \
  --bases_path "$BASES" --out_sub meanremoved_L5to7_top1 --model_dir "$MD" && \
python src/sandbox/ext_steerability/ablate_readdir_pc5.py --n_shots 1 \
  --bases_path "$BASES" --out_sub meanremoved_L5to7_top1 --model_dir "$MD"
rc=$?
date -u
if [ $rc -eq 0 ]; then touch "$LOG/run.done"; else touch "$LOG/run.failed"; fi
echo "exit $rc"
