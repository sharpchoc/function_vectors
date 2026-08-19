#!/usr/bin/env bash
# Monitor feed for the low-dim steering run: done/failed marker + failure tail.
LOGDIR=/workspace/function_vectors/.claude/worktrees/lowdim-steering/logs/lowdim_steering
while true; do
  for st in done failed; do
    f=$LOGDIR/swap1.$st
    if [ -f "$f" ]; then
      echo "swap1 -> $st"
      if [ "$st" = failed ]; then tail -5 "$LOGDIR/swap1.log"; fi
      echo "run terminal"
      exit 0
    fi
  done
  sleep 120
done
