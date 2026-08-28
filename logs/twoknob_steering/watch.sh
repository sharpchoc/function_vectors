#!/usr/bin/env bash
# Monitor feed: reports each arm's done/failed marker; exits when both arms are terminal
# or any arm fails.
LOGDIR=/workspace/function_vectors/.claude/worktrees/twoknob-steering/logs/twoknob_steering
while true; do
  for tag in twoknob meanfree_rl; do
    if [ -f "$LOGDIR/$tag.failed" ]; then
      echo "$tag -> failed"
      tail -8 "$LOGDIR/$tag.log"
      echo "run terminal"
      exit 0
    fi
  done
  if [ -f "$LOGDIR/twoknob.done" ] && [ -f "$LOGDIR/meanfree_rl.done" ]; then
    echo "both arms done"
    echo "run terminal"
    exit 0
  fi
  sleep 120
done
