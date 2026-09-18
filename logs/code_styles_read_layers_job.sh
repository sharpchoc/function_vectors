#!/bin/bash
# per-prompt read features at L6..L18 (every other layer), companion to prompt_pairs: <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/workspace/micromamba/envs/fv/bin/python src/sandbox/style_translation/capture_prompt_read_layers.py --model qwen25_code --families $FAMS --layers 6 8 10 12 14 16 18 > logs/read_layers_$T.log 2>&1 \
  && echo "JOB DONE $T" >> logs/read_layers_$T.log || echo "JOB FAILED $T" >> logs/read_layers_$T.log
