#!/bin/bash
# bug 14/15 re-sample for the families whose counted lists changed: step 3 (k = 0..4 rollouts + first-token log-prob margin),
# then write steering (train-split cue vectors k in {3,4} + L24/L26 x alpha 2/4 on the first 40 held-out docs, with baseline).
# usage: <tag> <families...>   (stale rollout / logprob / vector / full_k3 files of these families must be deleted BEFORE launch:
# rollout.py and capture_cues.py skip families whose output exists)
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"
PY=/workspace/micromamba/envs/fv/bin/python
rm -f /root/rf_$T.done /root/rf_$T.failed
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( $PY src/sandbox/style_translation/rollout.py --model qwen25_code --families $FAMS > logs/refix_rollout_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/logprob_margin.py --model qwen25_code --families $FAMS > logs/refix_logprob_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/capture_cues.py --model qwen25_code --families $FAMS --ks 3 4 --unpaired --split train --out_tag k3_train > logs/refix_capture_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/write_sweep.py --model qwen25_code --families $FAMS --layers 24 26 --alphas 2 4 --with_base --limit 40 \
       --out_dir artifacts/style_translation/qwen25_code/steering/full_k3/$T > logs/refix_sweep_$T.log 2>&1 ) \
  && touch /root/rf_$T.done || touch /root/rf_$T.failed
