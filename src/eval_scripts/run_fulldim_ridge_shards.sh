#!/usr/bin/env bash
# Launch the direct full-dim (4096->4096) activation->FV ridge regression as sharded tmux
# windows, one shard per ICL example index (1..10). Each shard loads its activation directory
# once and processes its token roles x 29 layers.
#
# Single A100 80GB: true GPU compute is serialized across processes, so the default caps at 3
# concurrent shards (enough to overlap one shard's disk loading with another's compute; more
# just contends for SMs). Shards are assigned to workers round-robin; each tmux window runs its
# assigned shards sequentially. Fully restartable: re-running skips finished shards unless
# OVERWRITE=1.
#
# Usage:
#   bash src/eval_scripts/run_fulldim_ridge_shards.sh                 # all 10 shards, 3 windows
#   CONCURRENCY=5 bash src/eval_scripts/run_fulldim_ridge_shards.sh   # 5 windows
#   ICL_INDICES="1 2 3" bash src/eval_scripts/run_fulldim_ridge_shards.sh
#   OVERWRITE=1 bash src/eval_scripts/run_fulldim_ridge_shards.sh
#
# Monitor:  tmux attach -t "$SESSION"      (Ctrl-b n / Ctrl-b w to switch windows)
# Then merge:  python src/eval_scripts/merge_fulldim_ridge_results.py --input_dir "$OUTPUT_DIR"
set -euo pipefail

SESSION="${SESSION:-fvridge}"
CONCURRENCY="${CONCURRENCY:-3}"
ICL_INDICES="${ICL_INDICES:-1 2 3 4 5 6 7 8 9 10}"
OUTPUT_DIR="${OUTPUT_DIR:-results/fulldim_ridge_activation_to_fv}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
[ "${OVERWRITE:-0}" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --overwrite"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Worker is overridable so the same launcher drives the full-dim and PCA-space sweeps, e.g.:
#   WORKER=src/eval_scripts/regress_activation_to_fv_pca_ridge.py \
#   OUTPUT_DIR=results/pca_ridge_activation_to_fv SESSION=pcaridge bash <this script>
WORKER="${WORKER:-src/eval_scripts/regress_activation_to_fv_fulldim_ridge.py}"

read -r -a ALL <<< "$ICL_INDICES"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Kill it first: tmux kill-session -t $SESSION" >&2
  exit 1
fi

echo "Launching ${#ALL[@]} shards across $CONCURRENCY tmux windows (session '$SESSION', GPU $GPU)."
echo "Output dir: $OUTPUT_DIR"

for ((w=0; w<CONCURRENCY; w++)); do
  # Round-robin: worker w handles indices w, w+CONCURRENCY, w+2*CONCURRENCY, ...
  assigned=()
  for ((i=w; i<${#ALL[@]}; i+=CONCURRENCY)); do
    assigned+=("${ALL[$i]}")
  done
  [ ${#assigned[@]} -eq 0 ] && continue

  # Build the sequential command for this window.
  cmd="cd '$REPO_DIR'; export CUDA_VISIBLE_DEVICES=$GPU;"
  for n in "${assigned[@]}"; do
    cmd="$cmd echo '=== shard icl$n ==='; python $WORKER --icl_index $n --output_dir '$OUTPUT_DIR' $EXTRA_ARGS;"
  done
  cmd="$cmd echo 'WORKER $w DONE'; exec bash"

  win_name="w${w}_icl$(IFS=,; echo "${assigned[*]}")"
  if [ "$w" -eq 0 ]; then
    tmux new-session -d -s "$SESSION" -n "$win_name" "$cmd"
  else
    tmux new-window -t "$SESSION" -n "$win_name" "$cmd"
  fi
  echo "  window $w -> shards: ${assigned[*]}"
done

echo
echo "Attach:  tmux attach -t $SESSION"
echo "Merge when done:  python src/eval_scripts/merge_fulldim_ridge_results.py --input_dir '$OUTPUT_DIR'"
