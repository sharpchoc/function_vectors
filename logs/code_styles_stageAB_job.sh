#!/bin/bash
# sequential refresh: stage A (rollouts k=0..4 if missing, log-probs, per-prompt activations of ALL k=3/4 prompts) then stage B
# (wait for the CPU judge markers -> write vectors + write steering L24/L26 x a2/a4 -> read vectors + read steering L8 a4). <tag> <families...>
cd /workspace/function_vectors || exit 1
T="$1"; shift; FAMS="$@"; PY=/workspace/micromamba/envs/fv/bin/python; A=artifacts/style_translation/qwen25_code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( $PY src/sandbox/style_translation/rollout.py --model qwen25_code --families $FAMS > logs/regen_rollout_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/logprob_margin.py --model qwen25_code --families $FAMS > logs/regen_logprob_$T.log 2>&1 \
  && for f in $FAMS; do until [ -f $A/read_features/evidence/$f.json ]; do sleep 20; done; done \
  && $PY src/sandbox/style_translation/capture_prompt_pairs_code.py --model qwen25_code --families $FAMS --ks 3 4 --read_layer 8 --write_layer 24 --select all > logs/regen_pairs_$T.log 2>&1 \
  && echo "STAGE A DONE $T" >> logs/stageAB_$T.log \
  && for f in $FAMS; do until [ -f $A/rollouts/$f.judged ]; do sleep 20; done; done \
  && $PY src/sandbox/style_translation/capture_cues.py --model qwen25_code --families $FAMS --ks 3 4 --unpaired --split train --out_tag k3_train > logs/regen_capture_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/write_sweep.py --model qwen25_code --families $FAMS --layers 24 26 --alphas 2 4 --with_base --limit 40 --out_dir $A/steering/full_k3/$T > logs/regen_wsweep_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/capture_read.py --model qwen25_code --families $FAMS --ks 3 4 --split train --out_tag k3_train > logs/regen_readcap_$T.log 2>&1 \
  && $PY src/sandbox/style_translation/read_sweep.py --model qwen25_code --families $FAMS --layers 8 --alphas 4 --with_base --limit 40 --out_dir $A/read_steer/full_k3/$T > logs/regen_rsweep_$T.log 2>&1 ) \
  && echo "JOB DONE $T" >> logs/stageAB_$T.log || echo "JOB FAILED $T" >> logs/stageAB_$T.log
