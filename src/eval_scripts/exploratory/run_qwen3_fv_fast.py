#!/usr/bin/env python3
"""Fast, resumable, dynamically-scheduled Qwen3-8B function-vector run.

Fixes the inefficiencies of run_qwen3_fv_all.py:
  * DYNAMIC GPU work-queue (single-task jobs; a free GPU immediately grabs the next job) -> no
    stragglers, no idle GPUs (the old static round-robin let one long task idle 3 GPUs for hours).
  * RESUMABLE: every job is skipped if its output already exists (reuses banked artifacts).
  * FILTER REUSE: symlinks task_specific's ICL-correct filter into the name the multitask builder
    expects, so the multitask CIE never recomputes the (slow, long-context) filter.
  * ADAPTIVE BATCH: per-task batch sized from the prompt token length (big for short tasks, small
    for long-context tasks like ag_news) -> no OOM, no global slowdown.
  * Combined with the now-BATCHED generate_str filter (eval_utils / varicl_utils).

Phases: 1) all per-task CIE/mean jobs (dynamic) -> 2) reduces -> 3) FV builds (dynamic).
Usage: python src/eval_scripts/run_qwen3_fv_fast.py --ngpu 4 [--dry_run] [--only 1|2|3]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/workspace/function_vectors")
MODEL = "Qwen/Qwen3-8B"
ART = Path(os.environ.get("FV_ARTIFACTS_ROOT", str(REPO / "artifacts/qwen3-8b")))
METRIC = "f1_score"
N_MEAN, N_IE = 100, 25
LOGDIR = ART / "_run_logs_fast"

TS_ROOT    = ART / "function_vectors/task_specific"
MTALL_ROOT = ART / "multitask_aie_heads_all_tasks_qwen3"
MT_ROOT    = ART / "multitask_aie_heads_qwen3"
VC_ROOT    = ART / "multitask_aie_heads_varicl_qwen3"
VC4_ROOT   = ART / "multitask_aie_heads_varicl_max4_qwen3"
FV_TRAIN_SEL  = ART / "function_vectors/train_selected"
FV_TRAIN_TEST = ART / "function_vectors/train_test_selected"
FV_VARICL  = ART / "function_vectors/train_varicl_top40"
FV_VARICL4 = ART / "function_vectors/train_varicl_max4_top40"
FV_AMBIG   = ART / "function_vectors/ambiguous_constrained"

SPLIT = json.loads((REPO / "task_splits/abstractive_train_test_tasks_29.json").read_text())
TRAIN, TEST = SPLIT["train_tasks"], SPLIT["test_tasks"]
ABS29 = TRAIN + TEST
PAIRED = ["east_neighbor", "west_neighbor", "next_in_period", "next_in_group",
          "next_number", "prev_number", "rhyme"]
DIGITS = ["next_number_digits", "prev_number_digits"]
STD_TASKS = ABS29 + PAIRED + DIGITS
AMBIG_TASKS = ["magnitude", "identity", "past_tense", "past_participle", "first_letter",
               "last_letter", "capital_city", "largest_city", "count_vowels", "count_consonants"]
# rhyme: model can't do it -> 0 ICL-correct; task_specific used --no_filter. Skip filter for it.
NOFILTER_TASKS = {"rhyme"}
# long-context tasks: hard-cap batch so [batch x seq x 152k-vocab] logits can't OOM 32GB even if
# the length probe fails (e.g. transformers missing). The probe normally sets these low anyway.
LONG_CTX = {"ag_news", "commonsense_qa", "sentiment"}


# ---------- adaptive batch via prompt-length probe ----------
def build_batch_map(tasks):
    """Tokenize one 10-shot prompt per task -> pick a CIE/gen batch that bounds
    batch*seq*vocab logits memory (~5GB). Falls back to 8 on any error."""
    bm = {}
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer
        from src.utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
        tok = AutoTokenizer.from_pretrained(MODEL)
        PREF = {"input": "Q:", "output": "A:", "instructions": ""}
        SEP = {"input": "\n", "output": "\n\n", "instructions": ""}
        for t in tasks:
            try:
                ds = load_dataset(t, root_data_dir=str(REPO / "dataset_files"), test_size=0.3, seed=42)
                tr = ds["train"]
                import numpy as np
                idx = np.random.default_rng(0).choice(len(tr), min(10, len(tr)), replace=False)
                pd = word_pairs_to_prompt_data(tr[idx], query_target_pair=ds["valid"][[0]],
                                               prepend_bos_token=False, shuffle_labels=False,
                                               prefixes=PREF, separators=SEP)
                n = len(tok(create_prompt(pd)).input_ids)
                bm[t] = max(2, min(24, 16000 // max(n, 1)))
            except Exception:
                bm[t] = 8
    except Exception as e:
        print(f"[probe] tokenizer/probe unavailable ({e}); default batch=8")
        bm = {t: 8 for t in tasks}
    for t in tasks:
        if t in LONG_CTX:
            bm[t] = min(bm.get(t, 8), 4)  # safety cap for long-context tasks
    return bm


# ---------- dynamic scheduler ----------
def launch(label, gpu, cmd):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME="/workspace/.cache/huggingface",
               HF_HUB_OFFLINE="1", FV_ARTIFACTS_ROOT=str(ART), PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    f = open(LOGDIR / f"{label}.log", "w")
    return subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=f, stderr=subprocess.STDOUT), f


def schedule(jobs, ngpu, dry):
    """jobs: list of dict(label, cmd, done:Path). Dynamic: any free GPU grabs the next job."""
    pending = [j for j in jobs if not Path(j["done"]).exists()]
    print(f"  {len(jobs)} jobs, {len(jobs)-len(pending)} already done, {len(pending)} to run")
    if dry:
        for j in pending[:6]:
            print(f"    [{j['label']}] {' '.join(map(str,j['cmd']))}")
        if len(pending) > 6:
            print(f"    ... (+{len(pending)-6} more)")
        return []
    free = list(range(ngpu))
    running, fails = {}, []
    while pending or running:
        while free and pending:
            gpu = free.pop(0); j = pending.pop(0)
            p, f = launch(j["label"], gpu, j["cmd"])
            running[gpu] = (j, p, f, time.time())
            print(f"  [gpu{gpu}] start {j['label']}")
        time.sleep(3)
        for gpu, (j, p, f, t0) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            f.close(); del running[gpu]; free.append(gpu)
            ok = rc == 0 and Path(j["done"]).exists()
            print(f"  [gpu{gpu}] done  {j['label']} rc={rc} ok={ok} ({time.time()-t0:.0f}s)")
            if not ok:
                fails.append((j["label"], rc))
    if fails:
        print(f"  !! FAILURES: {fails}")
    return fails


def py(script, *a):
    return [sys.executable, script, *map(str, a)]


# ---------- jobs ----------
def reuse_filters():
    """Symlink task_specific's filter into the name the multitask builder reuses, so the
    multitask CIE never recomputes the (slow) genstr filter."""
    n = 0
    for t in ABS29:
        src = TS_ROOT / t / "fs_results_layer_sweep.json"
        dst = TS_ROOT / t / "fs_results_validation_genstr.json"
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src.name); n += 1
            except OSError:
                shutil.copy(src, dst); n += 1
    print(f"  filter-reuse links created: {n}")


def cie_done(root, t):
    return root / t / f"{t}_cie_result.pt"


def phase1_jobs(bm):
    jobs = []
    # task_specific (resume any missing)
    for t in STD_TASKS:
        flt = "--no_filter_to_correct_icl" if t in NOFILTER_TASKS else "--filter_to_correct_icl"
        gen = [] if t in NOFILTER_TASKS else ["--generate_str", "--metric", METRIC]
        jobs.append(dict(label=f"ts_{t}", done=TS_ROOT / t / f"{t}_function_vector.pt",
            cmd=py("src/compute_function_vectors.py", "--dataset_names", t, "--model_name", MODEL,
                   "--n_top_heads", 10, *gen, "--batch_size", bm.get(t, 8),
                   "--n_mean_activations_trials", N_MEAN, "--n_indirect_effect_trials", N_IE,
                   "--save_path_root", TS_ROOT, flt, "--overwrite")))
    # multitask single-task CIE over 29 abstractive (reuse mean-acts + reused filter)
    for t in ABS29:
        gen = [] if t in NOFILTER_TASKS else ["--generate_str", "--metric", METRIC]
        flt = "--no_filter_to_correct_icl" if t in NOFILTER_TASKS else "--filter_to_correct_icl"
        jobs.append(dict(label=f"mt_{t}", done=cie_done(MTALL_ROOT, t),
            cmd=py("src/eval_scripts/compute_multitask_top_aie_heads.py", "--tasks", t,
                   "--abstractive_only", "--model_name", MODEL, "--query_split", "valid",
                   "--demo_split", "train", "--n_top_heads", 40, "--batch_size", bm.get(t, 8),
                   "--batch_size_filter_eval", bm.get(t, 8), flt, *gen,
                   "--mean_activations_root", TS_ROOT, "--save_path_root", MTALL_ROOT,
                   "--save_per_prompt_effects", "--num_shards", 2, "--shard_index", 0)))
    # varicl + varicl_max4 single-task CIE+mean over all 29 (train pool reduces over 20; build over 29)
    for tag, root, mx in (("vc", VC_ROOT, 10), ("vc4", VC4_ROOT, 4)):
        for t in ABS29:
            gen = [] if t in NOFILTER_TASKS else ["--generate_str", "--metric", METRIC]
            flt = "--no_filter_to_correct_icl" if t in NOFILTER_TASKS else "--filter_to_correct_icl"
            jobs.append(dict(label=f"{tag}_{t}", done=root / t / f"{t}_mean_head_activations_varicl.pt",
                cmd=py("src/eval_scripts/compute_multitask_varicl_heads.py", "--tasks", t,
                       "--abstractive_only", "--model_name", MODEL, "--query_split", "valid",
                       "--demo_split", "train", "--n_top_heads", 40, "--batch_size", bm.get(t, 8),
                       "--batch_size_filter_eval", bm.get(t, 8), "--min_shots", 1, "--max_shots", mx,
                       "--max_successful_prompts", 170, flt, *gen, "--save_per_prompt_effects",
                       "--save_path_root", root, "--num_shards", 2, "--shard_index", 0)))
    # ambiguous constrained (one job per task)
    for t in AMBIG_TASKS:
        jobs.append(dict(label=f"amb_{t}", done=FV_AMBIG / t / f"{t}_function_vector.pt",
            cmd=py("src/eval_scripts/compute_ambiguous_constrained_fv.py", "--tasks", t,
                   "--model_name", MODEL, "--n_top_heads", 10, "--n_trials", N_MEAN,
                   "--n_ie_trials", N_IE, "--batch_size", 8, "--output_root", FV_AMBIG)))
    return jobs


def phase2_reduces(dry):
    # all-tasks (train_test) pool over 29
    schedule([dict(label="reduce_all", done=MTALL_ROOT / "multitask_top_aie_heads.pt",
        cmd=py("src/eval_scripts/compute_multitask_top_aie_heads.py", "--all_split_tasks",
               "--model_name", MODEL, "--n_top_heads", 40, "--save_path_root", MTALL_ROOT,
               "--reduce", "--overwrite"))], 1, dry)
    # train pool: copy the 20 train CIEs into MT_ROOT, reduce there
    if not dry:
        MT_ROOT.mkdir(parents=True, exist_ok=True)
        for t in TRAIN:
            s, d = MTALL_ROOT / t, MT_ROOT / t
            if s.exists() and not d.exists():
                shutil.copytree(s, d)
    schedule([dict(label="reduce_train", done=MT_ROOT / "multitask_top_aie_heads.pt",
        cmd=py("src/eval_scripts/compute_multitask_top_aie_heads.py", "--task_split_key",
               "train_tasks", "--model_name", MODEL, "--n_top_heads", 40, "--save_path_root",
               MT_ROOT, "--reduce", "--overwrite"))], 1, dry)
    # varicl reduces (train pool of 20)
    for root in (VC_ROOT, VC4_ROOT):
        schedule([dict(label=f"reduce_{root.name}", done=root / "multitask_top_aie_heads.pt",
            cmd=py("src/eval_scripts/compute_multitask_varicl_heads.py", "--task_split_key",
                   "train_tasks", "--model_name", MODEL, "--n_top_heads", 40, "--save_path_root",
                   root, "--reduce", "--overwrite"))], 1, dry)


def phase3_builds(ngpu, dry):
    # Manifest-granular builds: one model load builds ALL tasks in a manifest (builds are light
    # matmuls, so per-task jobs would waste a model load each). Still dynamically scheduled.
    MANS = [("task_splits/abstractive_train_test_tasks_29.json", "abs29"),
            ("task_splits/paired_tasks_7.json", "paired")]
    jobs = []
    builds = [(MT_ROOT, 10, FV_TRAIN_SEL), (MT_ROOT, 20, Path(f"{FV_TRAIN_SEL}_top20")),
              (MT_ROOT, 40, Path(f"{FV_TRAIN_SEL}_top40")), (MTALL_ROOT, 10, FV_TRAIN_TEST)]
    for heads_root, n, out in builds:
        for man, stem in MANS:
            jobs.append(dict(label=f"build_{out.name}_n{n}_{stem}",
                done=out / f"fv_manifest_{stem}.json",
                cmd=py("src/eval_scripts/compute_all_task_fvs_from_multitask_heads.py",
                       "--task_manifest", man, "--model_name", MODEL, "--n_top_heads", n,
                       "--fv_root", TS_ROOT, "--heads_path", heads_root / "multitask_top_aie_heads.pt",
                       "--output_root", out, "--manifest_name", f"fv_manifest_{stem}.json", "--overwrite")))
    # varicl Stage-2 builds (single process each; fast)
    for root, out in ((VC_ROOT, FV_VARICL), (VC4_ROOT, FV_VARICL4)):
        jobs.append(dict(label=f"build_{out.name}", done=out / "antonym" / "antonym_function_vector.pt",
            cmd=py("src/eval_scripts/compute_all_task_fvs_varicl.py", "--task_manifest",
                   "task_splits/abstractive_train_test_tasks_29.json", "--model_name", MODEL,
                   "--n_top_heads", 40, "--fv_root", root, "--heads_path",
                   root / "multitask_top_aie_heads.pt", "--output_root", out, "--overwrite")))
    schedule(jobs, ngpu, dry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngpu", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--only", choices=["1", "2", "3"], default=None)
    a = ap.parse_args()
    print(f"ART={ART} NGPU={a.ngpu}")
    reuse_filters()
    if a.only in (None, "1"):
        print("\n===== PHASE 1: per-task CIE/mean (dynamic) =====")
        bm = build_batch_map(STD_TASKS)
        print("  batch map:", {k: bm[k] for k in list(bm)[:6]}, "...")
        schedule(phase1_jobs(bm), a.ngpu, a.dry_run)
    if a.only in (None, "2"):
        print("\n===== PHASE 2: reduces =====")
        phase2_reduces(a.dry_run)
    if a.only in (None, "3"):
        print("\n===== PHASE 3: FV builds (dynamic) =====")
        phase3_builds(a.ngpu, a.dry_run)
    print("\nDONE")


if __name__ == "__main__":
    main()
