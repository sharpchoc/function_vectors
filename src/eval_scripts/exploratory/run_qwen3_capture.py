#!/usr/bin/env python3
"""Capture Qwen3-8B residual-stream activations for the activation->FV ridge heatmap.

10 ICL captures over the 29 abstractive split tasks:
  - icl 1-9: token_roles = [pre_label, first_label, last_label]  -> qwen3_29tasks_170prompts_icl{n}_3tokens
  - icl 10 : + last_prompt_token (query)                         -> qwen3_29tasks_170prompts_4tokens
Dynamically scheduled across GPUs (no idle/stragglers); resumable (the capture script skips task dirs
that already have index.json via --no_overwrite_existing, and we skip fully-done chunks).
Usage: python src/eval_scripts/run_qwen3_capture.py --ngpu 4 [--chunk 4] [--dry_run]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path("/workspace/function_vectors")
MODEL = "Qwen/Qwen3-8B"
ART = Path(os.environ.get("FV_ARTIFACTS_ROOT", str(REPO / "artifacts/qwen3-8b")))
RA = ART / "residual_activations"
LOGDIR = ART / "_capture_logs"
SPLIT = json.loads((REPO / "task_splits/abstractive_train_test_tasks_29.json").read_text())
TASKS = SPLIT["train_tasks"] + SPLIT["test_tasks"]            # 29
ROLES3 = ["pre_label_token", "first_label_token", "last_label_token"]
ROLES4 = ROLES3 + ["last_prompt_token"]


def cap_dir(icl):
    name = f"qwen3_29tasks_170prompts_icl{icl}_3tokens" if icl < 10 else "qwen3_29tasks_170prompts_4tokens"
    return RA / name


def task_done(icl, task):
    d = cap_dir(icl) / task
    return (d / "train" / "index.json").exists() and (d / "test" / "index.json").exists()


def chunks(lst, n):
    return [lst[i::n] for i in range(n)]


def env_for(gpu):
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME="/workspace/.cache/huggingface",
             HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return e


def schedule(jobs, ngpu, dry):
    pending = [j for j in jobs if not j["done"]()]
    print(f"  {len(jobs)} jobs, {len(jobs)-len(pending)} already done, {len(pending)} to run", flush=True)
    if dry:
        for j in pending[:4]:
            print("    ", j["label"], " ".join(map(str, j["cmd"])))
        if len(pending) > 4:
            print(f"     ... (+{len(pending)-4} more)")
        return []
    LOGDIR.mkdir(parents=True, exist_ok=True)
    free, running, fails = list(range(ngpu)), {}, []
    while pending or running:
        while free and pending:
            gpu = free.pop(0); j = pending.pop(0)
            f = open(LOGDIR / f"{j['label']}.log", "w")
            p = subprocess.Popen(j["cmd"], cwd=str(REPO), env=env_for(gpu), stdout=f, stderr=subprocess.STDOUT)
            running[gpu] = (j, p, f, time.time())
            print(f"  [gpu{gpu}] start {j['label']}", flush=True)
        time.sleep(3)
        for gpu, (j, p, f, t0) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            f.close(); del running[gpu]; free.append(gpu)
            ok = rc == 0 and j["done"]()
            print(f"  [gpu{gpu}] done {j['label']} rc={rc} ok={ok} ({time.time()-t0:.0f}s)", flush=True)
            if not ok:
                fails.append((j["label"], rc))
    if fails:
        print("  !! FAILURES:", fails, flush=True)
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngpu", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4, help="approx tasks per capture job (amortizes model load)")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    n_chunks = max(1, (len(TASKS) + a.chunk - 1) // a.chunk)
    jobs = []
    for icl in range(1, 11):
        roles = ROLES4 if icl == 10 else ROLES3
        out = cap_dir(icl)
        for ci, chunk in enumerate(chunks(TASKS, n_chunks)):
            if not chunk:
                continue
            tasks = list(chunk)
            cmd = [sys.executable, "src/extract_targeted_residual_stream_activations.py",
                   "--dataset_names", *tasks, "--model_name", MODEL, "--splits", "train", "test",
                   "--seed", "42", "--n_shots", "10", "--max_train_prompts", "130",
                   "--max_test_prompts", "40", "--target_icl_example_index", str(icl),
                   "--token_roles", *roles, "--shard_size", "100", "--storage_dtype", "float16",
                   "--include_embeddings", "--no_overwrite_existing", "--save_path_root", str(out)]
            jobs.append(dict(label=f"cap_icl{icl}_c{ci}", cmd=cmd,
                             done=(lambda tt=tuple(tasks), ic=icl: all(task_done(ic, t) for t in tt))))
    print(f"capture jobs: {len(jobs)} (10 ICL x {n_chunks} chunks); tasks={len(TASKS)}", flush=True)
    schedule(jobs, a.ngpu, a.dry_run)
    print("CAPTURE_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
