#!/bin/bash
# 1-shot FV-direction cue-token ablation (69 tasks, L9-27 clamp only), GPU-pod launcher.
# Mirrors the 2026-08-19 6-shot study (ablate_fv_cue6.py) with --n_shots 1:
#   means (1-shot cue-token grand mean) -> combine -> eval (+ in-run seed-matched
#   real1_baseline). Artifacts: artifacts/69_task_run/FV_ablation/{cue_means_1shot,
#   grand_mean_cue1.pt, eval_1shot}. Run from a pod that mounts the shared /workspace.
set -u
WT=/workspace/function_vectors/.claude/worktrees/fv-ablation-1shot
cd "$WT" || exit 1
export FV_ARTIFACTS_ROOT=/workspace/function_vectors/artifacts
export PATH=/workspace/micromamba/envs/fv/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MD=/workspace/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots/47e169305d2e8376be1d31e765533382721b2cc1
S=src/sandbox/ext_steerability/ablate_fv_cue6.py
LOG=$WT/logs/fv_ablation_1shot
mkdir -p "$LOG"
rm -f "$LOG/run.done" "$LOG/run.failed"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python "$S" --stage means --n_shots 1 --model_dir "$MD" \
  && python "$S" --stage combine --n_shots 1 \
  && python "$S" --stage eval --n_shots 1 --layer_cfgs L9to27 --with_baseline --model_dir "$MD"
rc=$?
if [ $rc -eq 0 ]; then touch "$LOG/run.done"; else touch "$LOG/run.failed"; fi
echo "exit $rc"
