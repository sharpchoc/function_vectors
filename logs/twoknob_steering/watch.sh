#!/usr/bin/env bash
# Monitor feed: reports failures immediately; exits when both shards' both arms are done
# or any arm fails.
LOGDIR=/workspace/function_vectors/.claude/worktrees/twoknob-steering/logs/twoknob_steering
while true; do
  for tag in twoknob_s0 twoknob_s1 meanfree_rl_s0 meanfree_rl_s1; do
    if [ -f "$LOGDIR/$tag.failed" ]; then
      echo "$tag -> failed"
      tail -8 "$LOGDIR/$tag.log"
      echo "run terminal"
      exit 0
    fi
  done
  n=0
  for tag in twoknob_s0 twoknob_s1 meanfree_rl_s0 meanfree_rl_s1; do
    [ -f "$LOGDIR/$tag.done" ] && n=$((n+1))
  done
  if [ "$n" -eq 4 ]; then
    echo "all four arms done"
    echo "run terminal"
    exit 0
  fi
  sleep 120
done
