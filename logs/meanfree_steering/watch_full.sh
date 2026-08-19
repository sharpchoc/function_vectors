#!/usr/bin/env bash
# Monitor feed for the mean-free steering run: per-shot done/failed markers + failure tails.
LOGDIR=/workspace/function_vectors/.claude/worktrees/multidir-ablation/logs/meanfree_steering
seen=""
while true; do
  for tag in meanfree; do
    for st in done failed; do
      f=$LOGDIR/$tag.$st
      if [ -f "$f" ] && ! echo "$seen" | grep -q "$tag.$st"; then
        echo "$tag -> $st"
        seen="$seen $tag.$st"
        if [ "$st" = failed ]; then tail -5 "$LOGDIR/$tag.log"; fi
      fi
    done
  done
  if [ -f $LOGDIR/meanfree.done ] || [ -f $LOGDIR/meanfree.failed ] || [ -f $LOGDIR/meanfree.failed ]; then
    echo "run terminal"
    break
  fi
  sleep 120
done
