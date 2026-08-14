#!/usr/bin/env python
"""SANDBOX: train-side metric-prompt variants for the isolation upper-bound study.

For each task, derives from dataset_files/isolation_prompts/<task>/train_prompts.json
(150 fixed 10-shot clean prompts, demos+query from TRAIN) three fitting-prompt files in the
same folder, one per Lever-2 metric (write_up/isolation_methods_levers.md):

  train_zeroshot.json            the 150 train queries as 0-shot prompts (demos=[])
  train_sametask_shuffled10.json the 150 prompts with demo labels permuted within-prompt
                                 (plain permutation, fixed points allowed)
  train_mixedtask10.json         150 prompts: 10 demos from 10 distinct OTHER tasks'
                                 TRAIN pools (correct pairs) - same construction as
                                 test_mixedtask10 - query = the train prompt's query

Deterministic (seeded per task, distinct stream from the base generator).
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.prompt_utils import load_dataset

N_SHOTS = 10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_split_path", type=Path,
                   default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--root_data_dir", type=str, default=str(REPO_ROOT / "dataset_files"))
    p.add_argument("--prompts_root", type=Path, default=REPO_ROOT / "dataset_files" / "isolation_prompts")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def uniq_pairs(ds):
    seen, out = set(), []
    for i in range(len(ds)):
        if ds[i]["input"] not in seen:
            seen.add(ds[i]["input"])
            out.append({"input": ds[i]["input"], "output": ds[i]["output"]})
    return out


def main():
    args = parse_args()
    split = json.load(open(args.task_split_path))
    tasks = split["train_tasks"] + split["test_tasks"]
    assert len(tasks) == 29

    train_pools = {}
    for t in tasks:
        d = load_dataset(t, root_data_dir=args.root_data_dir, seed=args.seed,
                         merge_valid_into_train=True)
        train_pools[t] = uniq_pairs(d["train"])

    for t in tasks:
        # distinct RNG stream from the base generator (offset 7_000_000)
        rng = np.random.RandomState(args.seed + 7_000_000 + (zlib.crc32(t.encode()) % 100000))
        base = json.load(open(args.prompts_root / t / "train_prompts.json"))
        assert len(base) == 150

        zs = [{"task": t, "setting": "train_zeroshot", "n_shots": 0, "demos": [],
               "query": dict(p["query"]), "prompt_index": p["prompt_index"]} for p in base]

        shuf = []
        for p in base:
            demos = p["demos"]
            perm = rng.permutation(N_SHOTS)
            shuffled = [{"input": demos[i]["input"], "output": demos[perm[i]]["output"]}
                        for i in range(N_SHOTS)]
            shuf.append({"task": t, "setting": "train_sametask_shuffled10", "n_shots": N_SHOTS,
                         "demos": shuffled,
                         "demo_correct_outputs": [d["output"] for d in demos],
                         "query": dict(p["query"]), "prompt_index": p["prompt_index"]})

        others = [o for o in tasks if o != t]
        mixed = []
        for p in base:
            demo_tasks = [others[i] for i in rng.choice(len(others), size=N_SHOTS, replace=False)]
            demos = []
            for dt in demo_tasks:
                pool = train_pools[dt]
                d = dict(pool[int(rng.randint(len(pool)))])
                d["source_task"] = dt
                demos.append(d)
            mixed.append({"task": t, "setting": "train_mixedtask10", "n_shots": N_SHOTS,
                          "demos": demos, "query": dict(p["query"]),
                          "prompt_index": p["prompt_index"]})

        for name, data in (("train_zeroshot", zs), ("train_sametask_shuffled10", shuf),
                           ("train_mixedtask10", mixed)):
            with open(args.prompts_root / t / f"{name}.json", "w") as f:
                json.dump(data, f, indent=1)
        print(f"{t:26s} wrote 3 x 150 train-variant prompts")

    print("done")


if __name__ == "__main__":
    main()
