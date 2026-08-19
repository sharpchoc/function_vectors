#!/usr/bin/env bash
# Monitor feed for the pc5 ablation run: per-shot done/failed markers + failure tails.
LOGDIR=/workspace/function_vectors/.claude/worktrees/multidir-ablation/logs/multidir_ablation
seen=""
while true; do
  for tag in pc5_n1 pc5_n6; do
    for st in done failed; do
      f=$LOGDIR/$tag.$st
      if [ -f "$f" ] && ! echo "$seen" | grep -q "$tag.$st"; then
        echo "$tag -> $st"
        seen="$seen $tag.$st"
        if [ "$st" = failed ]; then tail -5 "$LOGDIR/$tag.log"; fi
      fi
    done
  done
  if [ -f $LOGDIR/pc5_n6.done ] || [ -f $LOGDIR/pc5_n6.failed ] || [ -f $LOGDIR/pc5_n1.failed ]; then
    echo "run terminal"
    break
  fi
  sleep 120
done
