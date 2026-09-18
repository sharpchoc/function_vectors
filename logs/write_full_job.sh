#!/bin/bash
# write-feature steering, full pool: <tag> <families...>  — capture this shard's vectors (train docs, k∈{3,4}), then steer
# the first 40 held-out 0-shot prompts at L24/L26 × α 2/4 (+ unsteered baseline). Sweep families are skipped by capture (files exist).
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"
PY=/workspace/micromamba/envs/fv/bin/python
rm -f /root/wf_$T.done /root/wf_$T.failed
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( $PY src/sandbox/style_translation/capture_cues.py --model qwen25_code --families $FAMS --ks 3 4 --unpaired --split train --out_tag k3_train > logs/write_full_capture_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/write_sweep.py --model qwen25_code --families $FAMS --layers 24 26 --alphas 2 4 --with_base --limit 40 \
       --out_dir artifacts/style_translation/qwen25_code/steering/full_k3/$T > logs/write_full_$T.log 2>&1 ) \
  && touch /root/wf_$T.done || touch /root/wf_$T.failed
