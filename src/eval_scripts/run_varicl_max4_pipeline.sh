#!/usr/bin/env bash
# Build the max_shots=4 variable-ICL FV variant (top-40 heads), parallel to the existing
# train_varicl (max_shots=10) pipeline. Single behavioral change vs run_multitask_varicl_all_tasks.sh:
# --max_shots 4. Encapsulates the documented stage-2 gotcha (test-task activations must be computed
# separately and copied into the main root before building FVs for all 29 tasks).
#
# Usage: bash src/eval_scripts/run_varicl_max4_pipeline.sh
set -euo pipefail
ARTIFACTS="${FV_ARTIFACTS_ROOT:-artifacts}"; RESULTS="${FV_RESULTS_ROOT:-results}"; LOGS="${FV_LOGS_ROOT:-logs}"

cd "$(dirname "$0")/../.."   # repo root
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MAIN_ROOT="$ARTIFACTS/multitask_aie_heads_varicl_max4"
TEST_ROOT="$RESULTS/general/_varicl_testtasks_max4"
FV_ROOT="$ARTIFACTS/function_vectors/gpt-j/train_varicl_max4_top40"
LOG_DIR="${MAIN_ROOT}/_logs"
mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --abstractive_only
  --query_split valid
  --demo_split train
  --n_top_heads 40
  --batch_size 8
  --min_shots 1
  --max_shots 4
  --max_successful_prompts 170
  --filter_to_correct_icl
  --save_per_prompt_effects
)

echo "=== [1a] train-task head selection (max4, top40) -> ${MAIN_ROOT} ==="
python src/eval_scripts/compute_multitask_varicl_heads.py \
  --task_split_key train_tasks "${COMMON_ARGS[@]}" \
  --save_path_root "${MAIN_ROOT}" --num_shards 1 2>&1 | tee "${LOG_DIR}/train_compute.log"

echo "=== [1a-reduce] pool train-task CIE -> multitask_top_aie_heads.pt ==="
python src/eval_scripts/compute_multitask_varicl_heads.py \
  --task_split_key train_tasks "${COMMON_ARGS[@]}" \
  --save_path_root "${MAIN_ROOT}" --num_shards 1 --reduce --overwrite 2>&1 | tee "${LOG_DIR}/train_reduce.log"

echo "=== [1b] test-task activations (gotcha) -> ${TEST_ROOT} ==="
python src/eval_scripts/compute_multitask_varicl_heads.py \
  --task_split_key test_tasks "${COMMON_ARGS[@]}" \
  --save_path_root "${TEST_ROOT}" --num_shards 1 2>&1 | tee "${LOG_DIR}/test_compute.log"

echo "=== [1b-copy] copy 9 test-task mean activations into ${MAIN_ROOT} ==="
TEST_TASKS=$(python -c "import json; print(' '.join(json.load(open('task_splits/abstractive_train_test_tasks_29.json'))['test_tasks']))")
for t in ${TEST_TASKS}; do
  src="${TEST_ROOT}/${t}/${t}_mean_head_activations_varicl.pt"
  dst_dir="${MAIN_ROOT}/${t}"
  mkdir -p "${dst_dir}"
  cp -v "${src}" "${dst_dir}/"
done

echo "=== [1c] build per-task FVs at top-40 -> ${FV_ROOT} ==="
python src/eval_scripts/compute_all_task_fvs_varicl.py \
  --fv_root "${MAIN_ROOT}" \
  --heads_path "${MAIN_ROOT}/multitask_top_aie_heads.pt" \
  --n_top_heads 40 \
  --output_root "${FV_ROOT}" --overwrite 2>&1 | tee "${LOG_DIR}/fvbuild.log"

echo "=== DONE: FVs in ${FV_ROOT} ==="
