#!/bin/bash
# twin patch at the k = 4 cue (natural-context prompts patched with the alternative twin's coordinates), layers 20..28: <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"; PY=/workspace/micromamba/envs/fv/bin/python; A=artifacts/style_translation/qwen25_code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY src/sandbox/style_translation/write_subspace_patch_k4.py --model qwen25_code --families $FAMS --layers $(seq 20 28) --limit 40 --out_dir $A/steering/subspace_patch_k4twin/$T > logs/subspace_k4_$T.log 2>&1 \
  && echo "JOB DONE $T" >> logs/subspace_k4_$T.log || echo "JOB FAILED $T" >> logs/subspace_k4_$T.log
