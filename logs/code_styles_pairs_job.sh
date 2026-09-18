#!/bin/bash
# per-prompt read (L8, evidence mean) / write (L24, cue) features: <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/workspace/micromamba/envs/fv/bin/python src/sandbox/style_translation/capture_prompt_pairs_code.py --model qwen25_code --families $FAMS --ks 3 4 --read_layer 8 --write_layer 24 > logs/pairs_capture_$T.log 2>&1 \
  && echo "JOB DONE $T" >> logs/pairs_capture_$T.log || echo "JOB FAILED $T" >> logs/pairs_capture_$T.log
