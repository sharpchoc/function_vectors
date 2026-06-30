#!/usr/bin/env python3
"""varicl-only Qwen3-8B run: per-task varicl CIE+mean (dynamic, 4 GPUs) -> reduce (train pool)
-> build varicl FVs. Excludes the long-context tasks (ag_news, commonsense_qa) from the pool.
Reuses the tested dynamic scheduler from run_qwen3_fv_fast. Resumable (skips done outputs)."""
import sys
sys.path.insert(0, "/workspace/function_vectors")
from pathlib import Path
from src.eval_scripts.run_qwen3_fv_fast import (
    schedule, py, MODEL, METRIC, VC_ROOT, FV_VARICL, SPLIT, TRAIN, ABS29, LONG_CTX,
)

POOL = [t for t in ABS29 if t not in LONG_CTX]            # 27 abstractive (excl 2 long)
TRAIN_POOL = [t for t in TRAIN if t not in LONG_CTX]      # 18 train (excl 2 long)

# Phase 1: per-task varicl CIE+mean (dynamic across GPUs)
jobs = []
for t in POOL:
    jobs.append(dict(label=f"vc_{t}", done=VC_ROOT / t / f"{t}_mean_head_activations_varicl.pt",
        cmd=py("src/eval_scripts/compute_multitask_varicl_heads.py", "--tasks", t,
               "--abstractive_only", "--model_name", MODEL, "--query_split", "valid",
               "--demo_split", "train", "--n_top_heads", 40, "--batch_size", 8,
               "--batch_size_filter_eval", 8, "--min_shots", 1, "--max_shots", 10,
               "--max_successful_prompts", 170, "--filter_to_correct_icl", "--generate_str",
               "--metric", METRIC, "--save_per_prompt_effects", "--save_path_root", VC_ROOT,
               "--num_shards", 2, "--shard_index", 0)))
print("===== varicl PHASE 1: per-task CIE/mean =====")
schedule(jobs, 4, False)

# Phase 2: reduce over the 18-task train pool -> varicl head set
print("===== varicl PHASE 2: reduce (train pool) =====")
schedule([dict(label="vc_reduce", done=VC_ROOT / "multitask_top_aie_heads.pt",
    cmd=py("src/eval_scripts/compute_multitask_varicl_heads.py", "--tasks", *TRAIN_POOL,
           "--reduce", "--model_name", MODEL, "--n_top_heads", 40,
           "--save_path_root", VC_ROOT, "--overwrite"))], 1, False)

# Phase 3: build varicl FVs over the 27 pool tasks
print("===== varicl PHASE 3: build FVs =====")
schedule([dict(label="vc_build", done=FV_VARICL / "antonym" / "antonym_function_vector.pt",
    cmd=py("src/eval_scripts/compute_all_task_fvs_varicl.py", "--task_manifest",
           "task_splits/abstractive_train_test_tasks_29.json", "--tasks", *POOL,
           "--model_name", MODEL, "--n_top_heads", 40, "--fv_root", VC_ROOT,
           "--heads_path", VC_ROOT / "multitask_top_aie_heads.pt",
           "--output_root", FV_VARICL, "--overwrite"))], 1, False)
print("VARICL_ALL_DONE")
