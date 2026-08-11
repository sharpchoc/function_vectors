#!/bin/bash
# SANDBOX sparse-PC CV shard runner. Usage: run_shard.sh <shard_tag> <lambda> [<lambda> ...]
set -u
TAG=$1; shift
cd /workspace/function_vectors || exit 1
LOG=logs/sandbox_pc/cv_${TAG}.log
mkdir -p logs/sandbox_pc
echo "=== shard ${TAG} lambdas: $* on $(hostname) $(date -u) ===" >> "$LOG"
/workspace/micromamba/envs/fv/bin/python src/sandbox/sparse_head_selection/train_sparse_pcs.py \
    --mode cv --lambdas "$@" >> "$LOG" 2>&1
rc=$?
if [ $rc -eq 0 ]; then touch logs/sandbox_pc/cv_${TAG}.done; else touch logs/sandbox_pc/cv_${TAG}.failed; fi
echo "=== shard ${TAG} exit ${rc} $(date -u) ===" >> "$LOG"
