#!/usr/bin/env bash
# Parallel CIE / top-AIE-head selection under the VARIABLE-ICL regime over the TRAIN
# abstractive tasks (train_tasks from task_splits/abstractive_train_test_tasks_29.json),
# stored in results/multitask_aie_heads_varicl.
#
# Each prompt draws a random 1-10 ICL demonstration count, keeps only prompts the model
# answers correctly (capped at 170 per task), and reads both the mean head activations and
# the CIE intervention/measurement at the query predictive token (last non-pad token, T=-1).
# CIE is pooled over the train tasks to pick the shared top-head set.
#
# GPT-J loads in fp16 (~12 GB), so several instances fit on one 80 GB A100. Each worker process
# owns a disjoint shard of tasks (tasks[shard::NUM_SHARDS]) and writes per-task
# <task>/<task>_cie_result.pt files. A final --reduce step aggregates them into the global
# top-head artifact. This script does NOT build function vectors (that is a separate, non-sharded
# step: src/eval_scripts/compute_all_task_fvs_varicl.py).
#
# Usage:
#   bash src/eval_scripts/run_multitask_varicl_all_tasks.sh [NUM_SHARDS] [SAVE_PATH_ROOT]
#
# Env overrides: CUDA_VISIBLE_DEVICES (default 0), NUM_SHARDS, SAVE_PATH_ROOT.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

NUM_SHARDS="${1:-${NUM_SHARDS:-3}}"
SAVE_PATH_ROOT="${2:-${SAVE_PATH_ROOT:-results/multitask_aie_heads_varicl}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT="src/eval_scripts/compute_multitask_varicl_heads.py"
LOG_DIR="${SAVE_PATH_ROOT}/_logs"
mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --task_split_key train_tasks
  --abstractive_only
  --query_split valid
  --demo_split train
  --n_top_heads 40
  --batch_size 8
  --min_shots 1
  --max_shots 10
  --max_successful_prompts 170
  --filter_to_correct_icl
  --save_per_prompt_effects
  --save_path_root "${SAVE_PATH_ROOT}"
  --num_shards "${NUM_SHARDS}"
)

echo "Launching ${NUM_SHARDS} worker shard(s) on GPU(s) ${CUDA_VISIBLE_DEVICES} -> ${SAVE_PATH_ROOT}"
pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  log="${LOG_DIR}/shard_${shard}.log"
  echo "  shard ${shard} -> ${log}"
  python "${SCRIPT}" "${COMMON_ARGS[@]}" --shard_index "${shard}" >"${log}" 2>&1 &
  pids+=($!)
done

# Wait for all shards; abort with a clear message if any fails.
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "ERROR: shard ${i} failed. See ${LOG_DIR}/shard_${i}.log" >&2
    fail=1
  fi
done
if [[ "${fail}" -ne 0 ]]; then
  echo "One or more shards failed; skipping reduce." >&2
  exit 1
fi

echo "All shards done. Reducing per-task results into the combined artifact..."
python "${SCRIPT}" "${COMMON_ARGS[@]}" --reduce --overwrite

echo "Done. Artifact + metadata under ${SAVE_PATH_ROOT}"
