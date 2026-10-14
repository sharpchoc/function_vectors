#!/bin/bash
# read-feature pipeline on the translation prompts (qwen25_code_tr): evidence-token read vectors (k 3,4 train) then the layer x strength sweep on held-out k = 3 prompts: <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"; PY=/workspace/micromamba/envs/fv/bin/python; A=artifacts/style_translation/qwen25_code_tr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( $PY src/sandbox/style_translation/capture_read.py --model qwen25_code_tr --families $FAMS --ks 3 4 --split train --out_tag k3_train --token_budget 12000 > logs/tr_readcap_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/read_sweep.py --model qwen25_code_tr --families $FAMS --layers 2 4 6 8 10 12 14 16 20 24 --alphas 1 2 4 8 --with_base --limit 40 --batch 12 --token_budget 12000 --out_dir $A/read_steer/sweep_k3/$T > logs/tr_rsweep_$T.log 2>&1 ) \
  && echo "JOB DONE $T" >> logs/tr_rsweep_$T.log || echo "JOB FAILED $T" >> logs/tr_rsweep_$T.log
