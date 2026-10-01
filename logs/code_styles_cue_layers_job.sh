#!/bin/bash
# per-prompt cue-token activations of paired k = 4 prompts, layers 20..28: <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"; PY=/workspace/micromamba/envs/fv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY src/sandbox/style_translation/capture_cue_layers.py --model qwen25_code --families $FAMS --k 4 --layers $(seq 20 28) > logs/cue_layers_$T.log 2>&1 \
  && echo "JOB DONE $T" >> logs/cue_layers_$T.log || echo "JOB FAILED $T" >> logs/cue_layers_$T.log
