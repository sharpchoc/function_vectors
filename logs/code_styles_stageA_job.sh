#!/bin/bash
# stage A of the refresh (no OpenRouter needed): rollouts k=0..4 + log-prob margins + per-prompt L8/L24 activations of ALL k=3/4 prompts
# (judge flags patched later by fix_pair_flags.py). <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"; PY=/workspace/micromamba/envs/fv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( $PY src/sandbox/style_translation/rollout.py --model qwen25_code --families $FAMS > logs/regen_rollout_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/logprob_margin.py --model qwen25_code --families $FAMS > logs/regen_logprob_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/capture_prompt_pairs_code.py --model qwen25_code --families $FAMS --ks 3 4 --read_layer 8 --write_layer 24 --select all > logs/regen_pairs_$T.log 2>&1 ) \
  && echo "JOB DONE $T" >> logs/stageA_$T.log || echo "JOB FAILED $T" >> logs/stageA_$T.log
