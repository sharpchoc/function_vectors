#!/bin/bash
# write-feature steering sweep shard: <tag> <capture:0|1> <base:0|1> <layers...>   (10 sweep families; vectors from the training documents)
cd /workspace/function_vectors || exit 1
T="$1"; CAP="$2"; BASE="$3"; shift 3
FAMS="py_is_none py_fstring js_hungarian py_comprehension js_strict_eq py_quotes py_self_name hex_constants docstring_quotes r_assignment"
PY=/workspace/micromamba/envs/fv/bin/python; V=artifacts/style_translation/qwen25_code/steering/vectors_k3_train
rm -f /root/ws_$T.done /root/ws_$T.failed
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( if [ "$CAP" = "1" ]; then $PY src/sandbox/style_translation/capture_cues.py --model qwen25_code --families $FAMS --ks 3 4 --unpaired --split train --out_tag k3_train > logs/write_capture_$T.log 2>&1 && touch $V/DONE; fi
  until [ -f $V/DONE ]; do sleep 10; done
  $PY src/sandbox/style_translation/write_sweep.py --model qwen25_code --families $FAMS --layers "$@" $( [ "$BASE" = "1" ] && echo --with_base ) --out_dir artifacts/style_translation/qwen25_code/steering/sweep_k3/$T > logs/write_sweep_$T.log 2>&1 ) \
  && touch /root/ws_$T.done || touch /root/ws_$T.failed
