#!/bin/bash
# step 3 (k = 0..4 rollouts) + step 3b (first-token log-prob margin) on the clean code corpus, model key qwen25_code; pass shard tag then families
cd /workspace/function_vectors || exit 1
T="$1"; shift
rm -f /root/cs3_$T.done /root/cs3_$T.failed
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( /workspace/micromamba/envs/fv/bin/python src/sandbox/style_translation/rollout.py --model qwen25_code --families "$@" > logs/code_styles_rollout_$T.log 2>&1 \
  && /workspace/micromamba/envs/fv/bin/python src/sandbox/style_translation/logprob_margin.py --model qwen25_code --families "$@" > logs/code_styles_logprob_$T.log 2>&1 ) \
  && touch /root/cs3_$T.done || touch /root/cs3_$T.failed
