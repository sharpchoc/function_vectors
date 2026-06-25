#!/usr/bin/env python3
"""Orchestrate the full Qwen3-8B function-vector rerun across N GPUs on one pod.

Reproduces the GPT-J FV methods for Qwen/Qwen3-8B (raw ICL, full-answer `--generate_str`
correctness filtering) over all non-extractive tasks, sharded across GPUs for wall-clock.

Waves (barriers between them):
  A   task_specific  : per-task CIE + mean-acts + FV for all STD tasks (fills the shared mean-act cache)
  B   multitask CIE  : all-29 abstractive CIE (sharded) -> reduce -> train_test + train head sets
  C   train FV builds: train_selected top-10/20/40 + train_test top-10 (reuse task_specific mean-acts)
  D   varicl         : varicl + varicl_max4 train CIE (sharded) -> reduce; test/paired mean-acts; Stage-2 builds
  E   ambiguous      : constrained-FV pipeline over the ambiguous family

Run a single wave with --only A|B|C|D|E ; print commands without running via --dry_run.
GPU sharding: per-task waves split the task list across CUDA_VISIBLE_DEVICES 0..NGPU-1;
sharded scripts use their built-in --num_shards/--shard_index.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/workspace/function_vectors")
MODEL = "Qwen/Qwen3-8B"
ART = Path(os.environ.get("FV_ARTIFACTS_ROOT", str(REPO / "artifacts/qwen3-8b")))

# tunables
BATCH = 32          # CIE / mean-activation batch (n_trials=25 fits a single CIE batch)
FBATCH = 16         # rank-filter eval batch (genstr filter is unbatched internally)
METRIC = "f1_score" # score==1 under f1 is effectively exact for single-word answers
N_MEAN = 100
N_IE = 25
MAX_SHOTS_VARICL = 10
MAX_SHOTS_MAX4 = 4
MAX_OK_PROMPTS = 170

# output roots (all under ART -> artifacts/qwen3-8b, gitignored except head metadata)
TS_ROOT      = ART / "function_vectors/task_specific"
MTALL_ROOT   = ART / "multitask_aie_heads_all_tasks_qwen3"   # train+test pool (29)
MT_ROOT      = ART / "multitask_aie_heads_qwen3"             # train pool (20), reduced from same CIE
VC_ROOT      = ART / "multitask_aie_heads_varicl_qwen3"
VC4_ROOT     = ART / "multitask_aie_heads_varicl_max4_qwen3"
VC_TEST_ROOT = ART / "_varicl_testtasks_qwen3"              # gitignored scratch (artifacts/_*)
VC4_TEST_ROOT= ART / "_varicl_max4_testtasks_qwen3"
FV_TRAIN_SEL = ART / "function_vectors/train_selected"
FV_TRAIN_TEST= ART / "function_vectors/train_test_selected"
FV_VARICL    = ART / "function_vectors/train_varicl_top40"
FV_VARICL4   = ART / "function_vectors/train_varicl_max4_top40"
FV_AMBIG     = ART / "function_vectors/ambiguous_constrained"
LOGDIR       = ART / "_run_logs"

SPLIT = json.loads((REPO / "task_splits/abstractive_train_test_tasks_29.json").read_text())
ABS29 = SPLIT["train_tasks"] + SPLIT["test_tasks"]
PAIRED_ONLY = ["east_neighbor", "west_neighbor", "next_in_period", "next_in_group",
               "next_number", "prev_number", "rhyme"]
DIGITS = ["next_number_digits", "prev_number_digits"]
STD_TASKS = ABS29 + PAIRED_ONLY + DIGITS                 # 38 tasks (task_specific universe)
# ambiguous-family disambiguation tasks (overlap/differentiator structured) for the constrained FV
AMBIG_TASKS = ["magnitude", "identity", "past_tense", "past_participle",
               "first_letter", "last_letter", "capital_city", "largest_city",
               "count_vowels", "count_consonants"]


def env_for(gpu):
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    e["HF_HOME"] = "/workspace/.cache/huggingface"
    e["HF_HUB_OFFLINE"] = "1"
    e["FV_ARTIFACTS_ROOT"] = str(ART)
    e["PYTHONUNBUFFERED"] = "1"
    return e


def run_pool(jobs, dry):
    """jobs: list of (label, gpu, cmd[list]). Run concurrently (already GPU-assigned), barrier."""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for label, gpu, cmd in jobs:
        log = LOGDIR / f"{label}.log"
        print(f"[gpu{gpu}] {label}: {' '.join(cmd)}")
        if dry:
            continue
        f = open(log, "w")
        p = subprocess.Popen(cmd, cwd=str(REPO), env=env_for(gpu), stdout=f, stderr=subprocess.STDOUT)
        procs.append((label, p, f))
    fails = []
    for label, p, f in procs:
        rc = p.wait(); f.close()
        print(f"   -> {label} exit {rc}")
        if rc != 0:
            fails.append(label)
    if fails:
        raise SystemExit(f"FAILED jobs: {fails} (see {LOGDIR})")


def chunks(lst, n):
    return [lst[i::n] for i in range(n)]


def py(script, *flags):
    return [sys.executable, script, *map(str, flags)]


# ---------- Wave A: task_specific ----------
def wave_A(ngpu, dry):
    groups = chunks(STD_TASKS, ngpu)
    jobs = []
    for g, tasks in enumerate(groups):
        if not tasks:
            continue
        cmd = py("src/compute_function_vectors.py", "--dataset_names", *tasks,
                 "--model_name", MODEL, "--n_top_heads", 10, "--generate_str", "--metric", METRIC,
                 "--batch_size", BATCH, "--n_mean_activations_trials", N_MEAN,
                 "--n_indirect_effect_trials", N_IE, "--save_path_root", TS_ROOT,
                 "--filter_to_correct_icl", "--continue_on_error", "--overwrite")
        jobs.append((f"A_taskspec_g{g}", g, cmd))
    run_pool(jobs, dry)


# ---------- Wave B: multitask CIE (all 29) sharded + reduce -> train_test & train head sets ----------
def wave_B(ngpu, dry):
    jobs = []
    for s in range(ngpu):
        cmd = py("src/eval_scripts/compute_multitask_top_aie_heads.py", "--all_split_tasks",
                 "--abstractive_only", "--model_name", MODEL, "--query_split", "valid",
                 "--demo_split", "train", "--n_top_heads", 40, "--batch_size", BATCH,
                 "--batch_size_filter_eval", FBATCH, "--filter_to_correct_icl", "--generate_str",
                 "--metric", METRIC, "--mean_activations_root", TS_ROOT,
                 "--save_path_root", MTALL_ROOT, "--save_per_prompt_effects",
                 "--num_shards", ngpu, "--shard_index", s)
        jobs.append((f"B_mtcie_s{s}", s, cmd))
    run_pool(jobs, dry)
    # reduce -> all-tasks (train_test) head set
    run_pool([("B_reduce_alltasks", 0,
               py("src/eval_scripts/compute_multitask_top_aie_heads.py", "--all_split_tasks",
                  "--model_name", MODEL, "--n_top_heads", 40, "--save_path_root", MTALL_ROOT,
                  "--reduce", "--overwrite"))], dry)
    # train pool: reuse the same per-task CIE files -> copy into MT_ROOT, reduce over train_tasks
    if not dry:
        MT_ROOT.mkdir(parents=True, exist_ok=True)
        for t in SPLIT["train_tasks"]:
            src, dst = MTALL_ROOT / t, MT_ROOT / t
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
    run_pool([("B_reduce_train", 0,
               py("src/eval_scripts/compute_multitask_top_aie_heads.py", "--task_split_key",
                  "train_tasks", "--model_name", MODEL, "--n_top_heads", 40,
                  "--save_path_root", MT_ROOT, "--reduce", "--overwrite"))], dry)


# ---------- Wave C: train_selected (10/20/40) + train_test (10) FV builds ----------
def wave_C(ngpu, dry):
    builds = []
    for n in (10, 20, 40):
        out = FV_TRAIN_SEL if n == 10 else Path(f"{FV_TRAIN_SEL}_top{n}")
        builds.append((MT_ROOT / "multitask_top_aie_heads.pt", n, out, "train_selected"))
    builds.append((MTALL_ROOT / "multitask_top_aie_heads.pt", 10, FV_TRAIN_TEST, "train_test"))
    # build per (heads, n, manifest); shard the 38 tasks across GPUs within each build
    manifests = [("task_splits/abstractive_train_test_tasks_29.json", ABS29),
                 ("task_splits/paired_tasks_7.json", PAIRED_ONLY + DIGITS)]
    for heads_path, n, out_root, tag in builds:
        for man, mtasks in manifests:
            if not mtasks:
                continue
            groups = chunks(mtasks, ngpu)
            jobs = []
            for g, tasks in enumerate(groups):
                if not tasks:
                    continue
                cmd = py("src/eval_scripts/compute_all_task_fvs_from_multitask_heads.py",
                         "--task_manifest", man, "--model_name", MODEL, "--n_top_heads", n,
                         "--fv_root", TS_ROOT, "--heads_path", heads_path,
                         "--output_root", out_root, "--tasks", *tasks,
                         "--manifest_name", f"fv_manifest.part{g}.json", "--overwrite")
                jobs.append((f"C_{tag}_n{n}_{Path(man).stem}_g{g}", g, cmd))
            run_pool(jobs, dry)


# ---------- Wave D: varicl + varicl_max4 ----------
def _varicl_flow(ngpu, dry, max_shots, main_root, test_root, fv_out, tag):
    # Stage 1: train-pool CIE (sharded) + reduce
    jobs = []
    for s in range(ngpu):
        cmd = py("src/eval_scripts/compute_multitask_varicl_heads.py", "--task_split_key",
                 "train_tasks", "--abstractive_only", "--model_name", MODEL, "--query_split",
                 "valid", "--demo_split", "train", "--n_top_heads", 40, "--batch_size", BATCH,
                 "--batch_size_filter_eval", FBATCH, "--min_shots", 1, "--max_shots", max_shots,
                 "--max_successful_prompts", MAX_OK_PROMPTS, "--filter_to_correct_icl",
                 "--generate_str", "--metric", METRIC, "--save_per_prompt_effects",
                 "--save_path_root", main_root, "--num_shards", ngpu, "--shard_index", s)
        jobs.append((f"D_{tag}_traincie_s{s}", s, cmd))
    run_pool(jobs, dry)
    run_pool([(f"D_{tag}_reduce", 0,
               py("src/eval_scripts/compute_multitask_varicl_heads.py", "--task_split_key",
                  "train_tasks", "--model_name", MODEL, "--n_top_heads", 40,
                  "--save_path_root", main_root, "--reduce", "--overwrite"))], dry)
    # Stage 1b (gotcha): varicl mean-acts for the non-train tasks the Stage-2 manifest needs
    nontrain = SPLIT["test_tasks"]
    jobs = []
    for s in range(ngpu):
        cmd = py("src/eval_scripts/compute_multitask_varicl_heads.py", "--tasks", *nontrain,
                 "--abstractive_only", "--model_name", MODEL, "--query_split", "valid",
                 "--demo_split", "train", "--n_top_heads", 40, "--batch_size", BATCH,
                 "--batch_size_filter_eval", FBATCH, "--min_shots", 1, "--max_shots", max_shots,
                 "--max_successful_prompts", MAX_OK_PROMPTS, "--filter_to_correct_icl",
                 "--generate_str", "--metric", METRIC, "--save_per_prompt_effects",
                 "--save_path_root", test_root, "--num_shards", ngpu, "--shard_index", s)
        jobs.append((f"D_{tag}_testact_s{s}", s, cmd))
    run_pool(jobs, dry)
    if not dry:
        for t in nontrain:
            src, dst = test_root / t, main_root / t
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
    # Stage 2: build varicl FVs (single process; fast)
    run_pool([(f"D_{tag}_build", 0,
               py("src/eval_scripts/compute_all_task_fvs_varicl.py", "--task_manifest",
                  "task_splits/abstractive_train_test_tasks_29.json", "--model_name", MODEL,
                  "--n_top_heads", 40, "--fv_root", main_root,
                  "--heads_path", main_root / "multitask_top_aie_heads.pt",
                  "--output_root", fv_out, "--overwrite"))], dry)


def wave_D(ngpu, dry):
    _varicl_flow(ngpu, dry, MAX_SHOTS_VARICL, VC_ROOT, VC_TEST_ROOT, FV_VARICL, "varicl")
    _varicl_flow(ngpu, dry, MAX_SHOTS_MAX4, VC4_ROOT, VC4_TEST_ROOT, FV_VARICL4, "varicl4")


# ---------- Wave E: ambiguous constrained ----------
def wave_E(ngpu, dry):
    groups = chunks(AMBIG_TASKS, ngpu)
    jobs = []
    for g, tasks in enumerate(groups):
        if not tasks:
            continue
        cmd = py("src/eval_scripts/compute_ambiguous_constrained_fv.py", "--tasks", *tasks,
                 "--model_name", MODEL, "--n_top_heads", 10, "--n_trials", N_MEAN,
                 "--n_ie_trials", N_IE, "--batch_size", 8, "--output_root", FV_AMBIG)
        jobs.append((f"E_ambig_g{g}", g, cmd))
    run_pool(jobs, dry)


WAVES = {"A": wave_A, "B": wave_B, "C": wave_C, "D": wave_D, "E": wave_E}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngpu", type=int, default=4)
    ap.add_argument("--only", choices=list(WAVES), default=None, help="Run a single wave.")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    print(f"NGPU={a.ngpu} ART={ART}")
    print(f"STD_TASKS={len(STD_TASKS)} AMBIG={len(AMBIG_TASKS)}")
    order = [a.only] if a.only else ["A", "B", "C", "D", "E"]
    for w in order:
        print(f"\n========== WAVE {w} ==========")
        WAVES[w](a.ngpu, a.dry_run)
    print("\nDONE")


if __name__ == "__main__":
    main()
