#!/bin/bash
# write-feature pipeline on the translation prompts (qwen25_code_tr): cue vectors (k 3,4 train, unpaired) then the layer x strength sweep on held-out 0-shot prompts: <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"; PY=/workspace/micromamba/envs/fv/bin/python; A=artifacts/style_translation/qwen25_code_tr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( $PY src/sandbox/style_translation/capture_cues.py --model qwen25_code_tr --families $FAMS --ks 3 4 --unpaired --split train --out_tag k3_train --token_budget 12000 > logs/tr_capture_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/write_sweep.py --model qwen25_code_tr --families $FAMS --layers 4 8 12 16 20 22 24 26 27 --alphas 1 2 4 --with_base --limit 40 --batch 20 --out_dir $A/steering/sweep_k3/$T > logs/tr_wsweep_$T.log 2>&1 ) \
  && echo "JOB DONE $T" >> logs/tr_wsweep_$T.log || echo "JOB FAILED $T" >> logs/tr_wsweep_$T.log
