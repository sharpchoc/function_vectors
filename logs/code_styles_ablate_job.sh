#!/bin/bash
# write-feature ablation at the cue token at EVERY layer 1..27 (k = 4, 40 held-out docs per pole): <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"; PY=/workspace/micromamba/envs/fv/bin/python; A=artifacts/style_translation/qwen25_code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY src/sandbox/style_translation/write_ablate.py --model qwen25_code --families $FAMS --k 4 --layers $(seq 1 27) --limit 40 --out_dir $A/ablation/full_k4/$T > logs/ablate_$T.log 2>&1 \
  && echo "JOB DONE $T" >> logs/ablate_$T.log || echo "JOB FAILED $T" >> logs/ablate_$T.log
