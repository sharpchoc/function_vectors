#!/bin/bash
# GPU-pod chains re-running every SVD-based (ctop1 / v1) result with the mean-residual objects
# (USER DECISION 2026-09-03: u_A = mean carrier-removed L5-7 residual; s_A = c + u_A).
# Usage: run_meanresid_all.sh sA0|sA1|sA2   s_A 1-shot 28-layer sweep shard, then s_A 6-shot @ L1|L6|L7
#        run_meanresid_all.sh wA0|wA1|wA2   w_A' (= own L6/7 mean + u_A) sweep shard, then
#                                            wA0: presence capture (carrier + u_hat_A)
#                                            wA1: Claim-4 cue effect of s_A @L0 (6-shot + 1-shot)
#                                            wA2: projection swap alpha*u_A @L6 (6-shot + 1-shot dummy)
#        run_meanresid_all.sh wsix <LAYER>   w_A' 6-shot at the sweep's peak layer
set -u
CHAIN=${1:?chain}
REPO=/workspace/function_vectors
cd "$REPO" || exit 1
export FV_ARTIFACTS_ROOT=$REPO/artifacts
export PATH=/workspace/micromamba/envs/fv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$REPO/logs/meanresid_all
mkdir -p "$LOG"
MARK=$CHAIN${2:+_L$2}; rm -f "$LOG/$MARK.done" "$LOG/$MARK.failed"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
MD=/workspace/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots/47e169305d2e8376be1d31e765533382721b2cc1
BA=$FV_ARTIFACTS_ROOT/69_task_run/bottom_up_ablation/bankA
SVEC=$BA/carrier_plus_meanresid_vectors.pt
WVEC=$BA/l67_plus_meanresid_vectors.pt
SOUT=$FV_ARTIFACTS_ROOT/69_task_run/meanresid_steering
WOUT=$FV_ARTIFACTS_ROOT/69_task_run/l67meanresid_steering
SIX="--token_budget 11000 --batch_cap 16"
date -u
rc=0
case "$CHAIN" in
  sA0|sA1|sA2)
    S=${CHAIN#sA}; L=$([ $S = 0 ] && echo 1 || ([ $S = 1 ] && echo 6 || echo 7))
    python src/sandbox/ext_steerability/sweep_l67top1_layers.py --shard_idx "$S" --shard_n 3 \
      --model_dir "$MD" --vectors_path "$SVEC" --out_root "$SOUT" || rc=1
    python src/sandbox/ext_steerability/sixshot_l67top1_steer.py --layer "$L" --model_dir "$MD" \
      --vectors_path "$SVEC" --out_root "$SOUT/sixshot_L$L" $SIX || rc=1
    ;;
  wA0|wA1|wA2)
    S=${CHAIN#wA}
    python src/sandbox/ext_steerability/sweep_l67top1_layers.py --shard_idx "$S" --shard_n 3 \
      --model_dir "$MD" --vectors_path "$WVEC" --out_root "$WOUT" || rc=1
    case "$S" in
      0)
        TASKS=$(python -c "import json;d=json.load(open('task_splits/extended_steerable_69_prunedfail.json'));print(' '.join(sorted(d['train_tasks']+d['heldout_tasks'])))")
        python src/eval_scripts/capture_69_ctop1_presence.py --tasks $TASKS \
          --vectors_path "$SVEC" --bases_path "$BA/meanresid_top1_bases.pt" \
          --out_root "$FV_ARTIFACTS_ROOT/69_task_run/meanresid_presence" || rc=1
        ;;
      1)
        for NS in 6 1; do
          SUF=$([ $NS = 1 ] && echo _1shot || echo "")
          python src/sandbox/ext_steerability/steer_effect_on_cue.py --n_shots $NS --model_dir "$MD" \
            --vectors_path "$SVEC" --inject_layer 0 \
            --out_root "$FV_ARTIFACTS_ROOT/69_task_run/meanresid_effect_on_write$SUF" || rc=1
        done
        ;;
      2)
        python src/sandbox/ext_steerability/steer_taskunique_svd.py --n_dirs 1 \
          --bases_path "$BA/meanresid_swap_bases.pt" --acts_root "$BA/actsfmt" \
          --alphas 0 0.5 1 2 4 8 --model_dir "$MD" \
          --out_root "$FV_ARTIFACTS_ROOT/69_task_run/raw_mean_steering/bankA_meanresid_swap_dummy" || rc=1
        ;;
    esac
    ;;
  wsix)
    L=${2:?layer}
    python src/sandbox/ext_steerability/sixshot_l67top1_steer.py --layer "$L" --model_dir "$MD" \
      --vectors_path "$WVEC" --out_root "$WOUT/sixshot_L$L" $SIX || rc=1
    ;;
  *) echo "unknown chain $CHAIN"; rc=1 ;;
esac
date -u
if [ $rc -eq 0 ]; then touch "$LOG/$MARK.done"; else touch "$LOG/$MARK.failed"; fi
echo "exit $rc"
