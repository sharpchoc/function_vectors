#!/bin/bash
# read->write causal test: <tag> <families...>   (read_causal.py, read layer 8, alphas .5 1 2 4 8, 40 held-out docs)
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"; PY=/workspace/micromamba/envs/fv/bin/python; A=artifacts/style_translation/qwen25_code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY src/sandbox/style_translation/read_causal.py --model qwen25_code --families $FAMS --read_layer 8 --alphas 0.5 1 2 4 8 --limit 40 --from_layer 20 --out_dir $A/read_causal > logs/read_causal_$T.log 2>&1 \
  && echo "JOB DONE $T" >> logs/read_causal_$T.log || echo "JOB FAILED $T" >> logs/read_causal_$T.log
