#!/bin/bash
# Bank-(a) migration GPU chains (USER DECISION 2026-09-01). Usage: run_bankA_pod.sh <chain>
#   chain=abl    : five ablation-ladder configs, n6+n1, bankA bases
#   chain=steer  : per-prompt capture + swap + two-knob with bankA bases/adapter
#   chain=sweep0|sweep1|sweep2 : w_A layer sweep shard (bankA vectors)
set -u
CHAIN=${1:?chain}
WT=/workspace/function_vectors/.claude/worktrees/fv-l5to7-top1
cd "$WT" || exit 1
export FV_ARTIFACTS_ROOT=/workspace/function_vectors/artifacts
export PATH=/workspace/micromamba/envs/fv/bin:$PATH
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$WT/logs/bankA
mkdir -p "$LOG"
rm -f "$LOG/$CHAIN.done" "$LOG/$CHAIN.failed"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
MD=/workspace/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots/47e169305d2e8376be1d31e765533382721b2cc1
BA=$FV_ARTIFACTS_ROOT/69_task_run/bottom_up_ablation/bankA
date -u
rc=0
case "$CHAIN" in
  abl)
    for cfg in mr11 top3 top1 L6to9_top3 L5to7_top1; do
      for ns in 6 1; do
        python src/sandbox/ext_steerability/ablate_readdir_pc5.py --n_shots $ns \
          --bases_path "$BA/${cfg}_bases.pt" --out_sub "bankA_${cfg}" --model_dir "$MD" \
          || rc=1
      done
    done
    ;;
  steer)
    python src/sandbox/ext_steerability/capture_label_resid_perprompt.py || rc=1
    python src/sandbox/ext_steerability/steer_taskunique_svd.py --n_dirs 1 \
      --bases_path "$BA/top3_bases.pt" --acts_root "$BA/actsfmt" \
      --out_root "$FV_ARTIFACTS_ROOT/69_task_run/raw_mean_steering/bankA_taskunique_svd_dummy" \
      --model_dir "$MD" || rc=1
    python src/sandbox/ext_steerability/steer_twoknob_dummy.py \
      --bases_path "$BA/top3_bases.pt" --acts_root "$BA/actsfmt" \
      --out_root "$FV_ARTIFACTS_ROOT/69_task_run/raw_mean_steering/bankA_twoknob_dummy" \
      --model_dir "$MD" || rc=1
    ;;
  sweep0|sweep1|sweep2)
    S=${CHAIN#sweep}
    python src/sandbox/ext_steerability/sweep_l67top1_layers.py \
      --shard_idx "$S" --shard_n 3 --model_dir "$MD" \
      --vectors_path "$BA/l67_plus_top1_vectors.pt" \
      --out_root "$FV_ARTIFACTS_ROOT/69_task_run/l67top1_steering_bankA" || rc=1
    ;;
  *) echo "unknown chain $CHAIN"; rc=1 ;;
esac
date -u
if [ $rc -eq 0 ]; then touch "$LOG/$CHAIN.done"; else touch "$LOG/$CHAIN.failed"; fi
echo "exit $rc"
