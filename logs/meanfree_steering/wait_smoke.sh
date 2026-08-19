#!/usr/bin/env bash
# Wait for the smoke run's terminal marker, then dump the log tail.
L=/workspace/function_vectors/.claude/worktrees/multidir-ablation/logs/meanfree_steering
until [ -f "$L/smoke_mf.done" ] || [ -f "$L/smoke_mf.failed" ]; do sleep 30; done
ls "$L"/smoke_mf.done "$L"/smoke_mf.failed 2>/dev/null
tail -30 "$L/smoke_mf.log"
