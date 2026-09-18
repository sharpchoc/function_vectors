#!/bin/bash
# read features: <tag> "<sweep layers>" <base 0|1> [capture families...]
# 1) capture_read.py for the given families (if any); 2) wait until the 10 sweep families' read vectors exist; 3) read_sweep.py shard.
cd /workspace/function_vectors || exit 1
T="$1"; LAYERS="$2"; BASE="$3"; shift 3; FAMS="$@"
PY=/workspace/micromamba/envs/fv/bin/python; V=artifacts/style_translation/qwen25_code/read_features/vectors_k3_train
SWEEP="py_is_none py_fstring js_hungarian py_comprehension js_strict_eq py_quotes py_self_name hex_constants docstring_quotes r_assignment"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( if [ -n "$FAMS" ]; then $PY src/sandbox/style_translation/capture_read.py --model qwen25_code --families $FAMS --ks 3 4 --split train --out_tag k3_train > logs/read_capture_$T.log 2>&1 || exit 1; fi
  for i in $(seq 1 360); do ok=1; for f in $SWEEP; do [ -f $V/$f.npz ] || ok=0; done; [ "$ok" = "1" ] && break; sleep 10; done
  $PY src/sandbox/style_translation/read_sweep.py --model qwen25_code --families $SWEEP --layers $LAYERS $( [ "$BASE" = "1" ] && echo --with_base ) \
     --out_dir artifacts/style_translation/qwen25_code/read_steer/sweep_k3/$T > logs/read_sweep_$T.log 2>&1 ) \
  && echo "JOB DONE $T" >> logs/read_sweep_$T.log || echo "JOB FAILED $T" >> logs/read_sweep_$T.log
