#!/bin/bash
# read-feature steering, full pool at the selected setting: <tag> <families...>  -> read_steer/full_k3/<tag>/<family>.json
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"
PY=/workspace/micromamba/envs/fv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY src/sandbox/style_translation/read_sweep.py --model qwen25_code --families $FAMS --layers 8 --alphas 4 --with_base --limit 40 \
   --out_dir artifacts/style_translation/qwen25_code/read_steer/full_k3/$T > logs/read_full_$T.log 2>&1 \
  && echo "JOB DONE $T" >> logs/read_full_$T.log || echo "JOB FAILED $T" >> logs/read_full_$T.log
