#!/usr/bin/env bash
# Sequential driver: two-knob grid first (the headline), then the mean-free randlabel C-arm.
LOGDIR=/workspace/function_vectors/.claude/worktrees/twoknob-steering/logs/twoknob_steering
bash "$LOGDIR/pod_run.sh" twoknob steer_twoknob_dummy
bash "$LOGDIR/pod_run.sh" meanfree_rl steer_meanfree_randlabel
