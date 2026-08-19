#!/usr/bin/env bash
# Monitor feed for the 1-shot effect-on-cue run: per-shot done/failed markers + failure tails.
LOGDIR=/workspace/function_vectors/.claude/worktrees/multidir-ablation/logs/effect1shot
seen=""
while true; do
  for tag in eff1; do
    for st in done failed; do
      f=$LOGDIR/$tag.$st
      if [ -f "$f" ] && ! echo "$seen" | grep -q "$tag.$st"; then
        echo "$tag -> $st"
        seen="$seen $tag.$st"
        if [ "$st" = failed ]; then tail -5 "$LOGDIR/$tag.log"; fi
      fi
    done
  done
  if [ -f $LOGDIR/eff1.done ] || [ -f $LOGDIR/eff1.failed ] || [ -f $LOGDIR/eff1.failed ]; then
    echo "run terminal"
    break
  fi
  sleep 120
done
