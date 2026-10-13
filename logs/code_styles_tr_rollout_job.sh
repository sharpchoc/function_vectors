#!/bin/bash
# step-3 rollouts on the code-translation prompts (model key qwen25_code_tr): <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"; PY=/workspace/micromamba/envs/fv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY src/sandbox/style_translation/rollout.py --model qwen25_code_tr --families $FAMS --token_budget 12000 --batch_cap 16 --append > logs/tr_rollout_$T.log 2>&1 \
  && echo "JOB DONE $T" >> logs/tr_rollout_$T.log || echo "JOB FAILED $T" >> logs/tr_rollout_$T.log
