#!/bin/bash
# per-prompt features for the prompts the correctness filter excluded: <tag> <families...>  -> prompt_pairs_rest/<family>.npz
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/workspace/micromamba/envs/fv/bin/python src/sandbox/style_translation/capture_prompt_pairs_code.py --model qwen25_code --families $FAMS --ks 3 4 --read_layer 8 --write_layer 24 --select incorrect --out_name prompt_pairs_rest > logs/pairs_rest_$T.log 2>&1 \
  && echo "JOB DONE $T" >> logs/pairs_rest_$T.log || echo "JOB FAILED $T" >> logs/pairs_rest_$T.log
