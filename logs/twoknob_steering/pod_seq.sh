#!/usr/bin/env bash
# Sequential driver: two-knob grid first (the headline), then the mean-free randlabel C-arm.
# usage: pod_seq.sh <shard_idx> <shard_n>
LOGDIR=/workspace/function_vectors/.claude/worktrees/twoknob-steering/logs/twoknob_steering
SI="${1:-0}"; SN="${2:-1}"
bash "$LOGDIR/pod_run.sh" twoknob_s$SI steer_twoknob_dummy --shard_idx "$SI" --shard_n "$SN"
bash "$LOGDIR/pod_run.sh" meanfree_rl_s$SI steer_meanfree_randlabel --shard_idx "$SI" --shard_n "$SN"
