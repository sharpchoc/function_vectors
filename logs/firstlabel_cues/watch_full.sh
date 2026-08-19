#!/usr/bin/env bash
# Monitor feed for the first-label effect-on-cues run: per-shot done/failed markers + failure tails.
LOGDIR=/workspace/function_vectors/.claude/worktrees/multidir-ablation/logs/firstlabel_cues
seen=""
while true; do
  for tag in flc; do
    for st in done failed; do
      f=$LOGDIR/$tag.$st
      if [ -f "$f" ] && ! echo "$seen" | grep -q "$tag.$st"; then
        echo "$tag -> $st"
        seen="$seen $tag.$st"
        if [ "$st" = failed ]; then tail -5 "$LOGDIR/$tag.log"; fi
      fi
    done
  done
  if [ -f $LOGDIR/flc.done ] || [ -f $LOGDIR/flc.failed ] || [ -f $LOGDIR/flc.failed ]; then
    echo "run terminal"
    break
  fi
  sleep 120
done
