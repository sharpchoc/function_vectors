#!/usr/bin/env bash
# Monitor feed for the centered-pc5 ablation run: per-shot done/failed markers + failure tails.
LOGDIR=/workspace/function_vectors/.claude/worktrees/multidir-ablation/logs/centered_ablation
seen=""
while true; do
  for tag in pc5c_n1 pc5c_n6; do
    for st in done failed; do
      f=$LOGDIR/$tag.$st
      if [ -f "$f" ] && ! echo "$seen" | grep -q "$tag.$st"; then
        echo "$tag -> $st"
        seen="$seen $tag.$st"
        if [ "$st" = failed ]; then tail -5 "$LOGDIR/$tag.log"; fi
      fi
    done
  done
  if [ -f $LOGDIR/pc5c_n6.done ] || [ -f $LOGDIR/pc5c_n6.failed ] || [ -f $LOGDIR/pc5c_n1.failed ]; then
    echo "run terminal"
    break
  fi
  sleep 120
done
