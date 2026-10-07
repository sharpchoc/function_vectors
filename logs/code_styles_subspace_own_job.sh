#!/bin/bash
# own-counterpart subspace patch towards the alternative convention, layers 20..28 (needs cue_pairs_layers captured): <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"; PY=/workspace/micromamba/envs/fv/bin/python; A=artifacts/style_translation/qwen25_code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY src/sandbox/style_translation/write_subspace_patch_own.py --model qwen25_code --families $FAMS --layers $(seq 20 28) --limit 40 --out_dir $A/steering/subspace_patch_own_k4/$T > logs/subspace_own_$T.log 2>&1 \
  && echo "JOB DONE $T" >> logs/subspace_own_$T.log || echo "JOB FAILED $T" >> logs/subspace_own_$T.log
