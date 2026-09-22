#!/bin/bash
# stage B of the refresh (after the judge): write vectors + write steering, read vectors + read steering, per-prompt activations. <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"; PY=/workspace/micromamba/envs/fv/bin/python; A=artifacts/style_translation/qwen25_code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( for f in $FAMS; do until [ -f $A/rollouts/$f.judged ]; do sleep 20; done; done \
  && $PY src/sandbox/style_translation/capture_cues.py --model qwen25_code --families $FAMS --ks 3 4 --unpaired --split train --out_tag k3_train > logs/regen_capture_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/write_sweep.py --model qwen25_code --families $FAMS --layers 24 26 --alphas 2 4 --with_base --limit 40 --out_dir $A/steering/full_k3/$T > logs/regen_wsweep_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/capture_read.py --model qwen25_code --families $FAMS --ks 3 4 --split train --out_tag k3_train > logs/regen_readcap_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/read_sweep.py --model qwen25_code --families $FAMS --layers 8 --alphas 4 --with_base --limit 40 --out_dir $A/read_steer/full_k3/$T > logs/regen_rsweep_$T.log 2>&1 ) \
  && echo "JOB DONE $T" >> logs/stageB_$T.log || echo "JOB FAILED $T" >> logs/stageB_$T.log
