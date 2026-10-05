#!/bin/bash
# all-family subspace patch towards the ALTERNATIVE convention: capture cue pairs (layers 20..28, k = 4) then patch2/patch1/add2 per layer: <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"; PY=/workspace/micromamba/envs/fv/bin/python; A=artifacts/style_translation/qwen25_code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( $PY src/sandbox/style_translation/capture_cue_layers.py --model qwen25_code --families $FAMS --k 4 --layers $(seq 20 28) > logs/cue_layers_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/write_subspace_patch.py --model qwen25_code --families $FAMS --layers $(seq 20 28) --limit 40 --targets alt --out_dir $A/steering/subspace_patch_k4/$T > logs/subspace_patch_$T.log 2>&1 ) \
  && echo "JOB DONE $T" >> logs/subspace_patch_$T.log || echo "JOB FAILED $T" >> logs/subspace_patch_$T.log
