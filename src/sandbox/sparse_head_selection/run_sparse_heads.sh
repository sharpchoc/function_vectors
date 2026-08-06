#!/usr/bin/env bash
# SANDBOX sparse-optimization head selection - thin runner.
# Usage:
#   ./run_sparse_heads.sh check            # FV-construction consistency check only
#   ./run_sparse_heads.sh smoke            # tiny end-to-end run
#   ./run_sparse_heads.sh cv [lambdas...]  # LOTO folds (optionally a lambda shard, e.g. "cv 0.01 0.02")
#   ./run_sparse_heads.sh reduce           # aggregate + final retrain + outputs
#   ./run_sparse_heads.sh all
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root

MODE="${1:-all}"
shift || true
EXTRA=()
if [ "$MODE" = "cv" ] && [ "$#" -gt 0 ]; then
    EXTRA=(--lambdas "$@")
fi

PYTHON="${PYTHON:-python}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p logs/sandbox_sparse_heads
LOG="logs/sandbox_sparse_heads/${MODE}_$(date +%Y%m%d_%H%M%S).log"
echo "logging to $LOG"
"$PYTHON" src/sandbox/sparse_head_selection/train_sparse_heads.py --mode "$MODE" "${EXTRA[@]}" 2>&1 | tee "$LOG"
