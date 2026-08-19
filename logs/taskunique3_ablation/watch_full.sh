#!/usr/bin/env bash
# Monitor feed for the mean-removed-11dir ablation run: per-shot done/failed markers + failure tails.
LOGDIR=/workspace/function_vectors/.claude/worktrees/bottomup-ablation/logs/taskunique3_ablation
seen=""
while true; do
  for tag in mr3_n1 mr3_n6; do
    for st in done failed; do
      f=$LOGDIR/$tag.$st
      if [ -f "$f" ] && ! echo "$seen" | grep -q "$tag.$st"; then
        echo "$tag -> $st"
        seen="$seen $tag.$st"
        if [ "$st" = failed ]; then tail -5 "$LOGDIR/$tag.log"; fi
      fi
    done
  done
  if [ -f $LOGDIR/mr3_n6.done ] || [ -f $LOGDIR/mr3_n6.failed ] || [ -f $LOGDIR/mr3_n1.failed ]; then
    echo "run terminal"
    break
  fi
  sleep 120
done
