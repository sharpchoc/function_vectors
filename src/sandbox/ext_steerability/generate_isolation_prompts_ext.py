#!/usr/bin/env python
"""SANDBOX: isolation-prompt sets for the extended_steerable_90 tasks.

Same constructions as dataset_files/isolation_prompts (see generate_isolation_prompts.py),
adapted for the extended pool:
  - datasets load directly from dataset_files/extended_tasks/ (load_dataset() only searches
    abstractive/extractive), example split via split_icl_dataset(merge_valid_into_train=True)
    (train 79% / test 21%, test membership = historical two-stage split);
  - per task: train_prompts.json (min(150, n_train_unique) fixed 10-shot, correct labels,
    distinct queries) + train_zeroshot.json, and 50 paired test queries (min(50, n_test_unique))
    x {test_zeroshot, test_sametask_shuffled10, test_mixedtask10};
  - mixed-task demos: 10 distinct tasks drawn from the 72 TRAIN tasks (minus the prompt's own
    task), correct pairs from their train pools - held-out task examples appear in NO prompt
    except their own task's sets.
Deterministic (seeded per task); pools deduplicated by input.
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
from src.utils.prompt_utils import ICLDataset, split_icl_dataset

N_SHOTS = 10
N_TRAIN_PROMPTS = 150
N_TEST_PROMPTS = 50


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split_path", type=Path, default=REPO_ROOT / "task_splits" / "extended_steerable_90.json")
    p.add_argument("--dataset_root", type=Path, default=REPO_ROOT / "dataset_files" / "extended_tasks")
    p.add_argument("--out_root", type=Path, default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def uniq_pairs(ds):
    seen, out = set(), []
    for i in range(len(ds)):
        if ds[i]["input"] not in seen:
            seen.add(ds[i]["input"])
            out.append({"input": ds[i]["input"], "output": ds[i]["output"]})
    return out


def sample_demos(rng, pool, k, exclude_input):
    idx = [i for i in range(len(pool)) if pool[i]["input"] != exclude_input]
    pick = rng.choice(len(idx), size=k, replace=False)
    return [dict(pool[idx[i]]) for i in pick]


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    train_tasks, heldout_tasks = split["train_tasks"], split["heldout_tasks"]
    tasks = train_tasks + heldout_tasks

    train_pools, test_pools = {}, {}
    for t in tasks:
        d = split_icl_dataset(ICLDataset(str(args.dataset_root / f"{t}.json")),
                              test_size=0.3, seed=args.seed, merge_valid_into_train=True)
        train_pools[t], test_pools[t] = uniq_pairs(d["train"]), uniq_pairs(d["test"])

    capped = {}
    for t in tasks:
        rng = np.random.RandomState(args.seed + 21_000_000 + (zlib.crc32(t.encode()) % 100000))
        tr, te = train_pools[t], test_pools[t]
        n_tp = min(N_TRAIN_PROMPTS, len(tr))
        n_te = min(N_TEST_PROMPTS, len(te))
        if n_tp < N_TRAIN_PROMPTS or n_te < N_TEST_PROMPTS:
            capped[t] = {"train_prompts": n_tp, "test_prompts": n_te}
        assert len(tr) > N_SHOTS + 1, f"{t}: train pool too small ({len(tr)})"
        out_dir = args.out_root / t
        out_dir.mkdir(parents=True, exist_ok=True)

        q_idx = rng.choice(len(tr), size=n_tp, replace=False)
        train_prompts = []
        for j, qi in enumerate(q_idx):
            demos = sample_demos(rng, tr, N_SHOTS, tr[qi]["input"])
            train_prompts.append({"task": t, "setting": "train_10shot", "n_shots": N_SHOTS,
                                  "demos": demos, "query": dict(tr[qi]), "prompt_index": j})
        train_zs = [{"task": t, "setting": "train_zeroshot", "n_shots": 0, "demos": [],
                     "query": dict(p["query"]), "prompt_index": p["prompt_index"]}
                    for p in train_prompts]

        tq_idx = rng.choice(len(te), size=n_te, replace=False)
        test_queries = [dict(te[i]) for i in tq_idx]
        zs = [{"task": t, "setting": "test_zeroshot", "n_shots": 0, "demos": [],
               "query": q, "prompt_index": j} for j, q in enumerate(test_queries)]
        shuf = []
        for j, q in enumerate(test_queries):
            demos = sample_demos(rng, tr, N_SHOTS, q["input"])
            perm = rng.permutation(N_SHOTS)
            shuffled = [{"input": demos[i]["input"], "output": demos[perm[i]]["output"]}
                        for i in range(N_SHOTS)]
            shuf.append({"task": t, "setting": "test_sametask_shuffled10", "n_shots": N_SHOTS,
                         "demos": shuffled,
                         "demo_correct_outputs": [d["output"] for d in demos],
                         "query": q, "prompt_index": j})
        # mixed-task demos: TRAIN tasks only, minus self (heldout examples never leak)
        demo_pool_tasks = [o for o in train_tasks if o != t]
        mixed = []
        for j, q in enumerate(test_queries):
            demo_tasks = [demo_pool_tasks[i]
                          for i in rng.choice(len(demo_pool_tasks), size=N_SHOTS, replace=False)]
            demos = []
            for dt in demo_tasks:
                pool = train_pools[dt]
                d = dict(pool[int(rng.randint(len(pool)))])
                d["source_task"] = dt
                demos.append(d)
            mixed.append({"task": t, "setting": "test_mixedtask10", "n_shots": N_SHOTS,
                          "demos": demos, "query": q, "prompt_index": j})

        for name, data in (("train_prompts", train_prompts), ("train_zeroshot", train_zs),
                           ("test_zeroshot", zs), ("test_sametask_shuffled10", shuf),
                           ("test_mixedtask10", mixed)):
            with open(out_dir / f"{name}.json", "w") as f:
                json.dump(data, f, indent=1)
        print(f"{t:32s} train={n_tp:3d} test={n_te:2d}")

    meta = {
        "created_with": "src/sandbox/ext_steerability/generate_isolation_prompts_ext.py",
        "split_path": str(args.split_path.relative_to(REPO_ROOT)),
        "seed": args.seed,
        "example_split": "split_icl_dataset(merge_valid_into_train=True): train 79% / test 21%",
        "constructions": "identical to dataset_files/isolation_prompts (see its metadata.json), "
                         "except 50 test queries/task and mixed-task demos drawn ONLY from the 72 "
                         "train tasks (minus self)",
        "capped_tasks": capped,
        "n_tasks": len(tasks),
    }
    with open(args.out_root / "metadata.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\ncapped tasks: {capped}")
    print(f"wrote {args.out_root}/metadata.json ({len(tasks)} tasks)")


if __name__ == "__main__":
    main()
