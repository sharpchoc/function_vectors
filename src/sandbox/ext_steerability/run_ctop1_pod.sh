#!/bin/bash
# GPU-pod chains for the carrier + n_A*v1 ("ctop1") steering vector (USER-ADJUDICATED 2026-09-02).
# Usage: run_ctop1_pod.sh sweep0|sweep1|sweep2        (1-shot layer sweep, 3 task shards)
#        run_ctop1_pod.sh sixshot <LAYER>             (6-shot dummy at the sweep's peak layer)
#        run_ctop1_pod.sh cue                         (u_A@L0 effect on the cue token, 6+1 shot)
set -u
CHAIN=${1:?chain}
REPO=/workspace/function_vectors
cd "$REPO" || exit 1
export FV_ARTIFACTS_ROOT=$REPO/artifacts
export PATH=/workspace/micromamba/envs/fv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$REPO/logs/ctop1
mkdir -p "$LOG"
MARK=$CHAIN${2:+_L$2}; rm -f "$LOG/$MARK.done" "$LOG/$MARK.failed"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# explicit COMPLETE snapshot (default glob picks the weights-only f3f42882 snapshot)
MD=/workspace/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots/47e169305d2e8376be1d31e765533382721b2cc1
VEC=$FV_ARTIFACTS_ROOT/69_task_run/bottom_up_ablation/bankA/carrier_plus_top1_vectors.pt
OUT=$FV_ARTIFACTS_ROOT/69_task_run/ctop1_steering
date -u
rc=0
case "$CHAIN" in
  sweep0|sweep1|sweep2)
    S=${CHAIN#sweep}
    python src/sandbox/ext_steerability/sweep_l67top1_layers.py \
      --shard_idx "$S" --shard_n 3 --model_dir "$MD" \
      --vectors_path "$VEC" --out_root "$OUT" || rc=1
    ;;
  sixshot)
    LAYER=${2:?layer}
    python src/sandbox/ext_steerability/sixshot_l67top1_steer.py --layer "$LAYER" \
      --model_dir "$MD" --vectors_path "$VEC" --out_root "$OUT/sixshot_L$LAYER" \
      --token_budget 11000 --batch_cap 16 || rc=1
    ;;
  cue)
    # effect of u_A injection (L0, all six slots) on the final cue token, 6-shot + 1-shot
    python src/sandbox/ext_steerability/steer_effect_on_cue.py --n_shots 6 --model_dir "$MD" \
      --vectors_path "$VEC" --inject_layer 0 \
      --out_root "$FV_ARTIFACTS_ROOT/69_task_run/ctop1_effect_on_write" || rc=1
    python src/sandbox/ext_steerability/steer_effect_on_cue.py --n_shots 1 --model_dir "$MD" \
      --vectors_path "$VEC" --inject_layer 0 \
      --out_root "$FV_ARTIFACTS_ROOT/69_task_run/ctop1_effect_on_write_1shot" || rc=1
    ;;
  *) echo "unknown chain $CHAIN"; rc=1 ;;
esac
date -u
if [ $rc -eq 0 ]; then touch "$LOG/$MARK.done"; else touch "$LOG/$MARK.failed"; fi
echo "exit $rc"
