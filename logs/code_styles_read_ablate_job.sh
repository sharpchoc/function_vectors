#!/bin/bash
# read-feature ablation at every evidence token, layers 0..27 (k = 4, alt context, 40 held-out docs): <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"; PY=/workspace/micromamba/envs/fv/bin/python; A=artifacts/style_translation/qwen25_code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY src/sandbox/style_translation/read_ablate.py --model qwen25_code --families $FAMS --k 4 --layers $(seq 0 27) --limit 40 --contexts alt --out_dir $A/ablation/read_full_k4/$T > logs/rablate_$T.log 2>&1 \
  && echo "JOB DONE $T" >> logs/rablate_$T.log || echo "JOB FAILED $T" >> logs/rablate_$T.log
