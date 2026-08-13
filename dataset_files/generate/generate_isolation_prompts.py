#!/usr/bin/env python
"""Generate the isolation-methods study prompt sets (see write_up/isolation_methods_levers.md).

For each of the 29 abstractive tasks (train/test task split irrelevant here - every task
gets the same treatment), using the NO-VALIDATION example split
(load_dataset(..., merge_valid_into_train=True): train 79% / test 21%):

  dataset_files/isolation_prompts/<task>/
    train_prompts.json            150 fixed 10-shot prompts; demos + query all from TRAIN
                                  examples, correct labels, 150 distinct queries
    test_zeroshot.json            30 prompts; query from TEST examples, no demos
    test_sametask_shuffled10.json 30 prompts; 10 TRAIN demos with labels permuted within
                                  the prompt (plain permutation, fixed points allowed -
                                  same convention as word_pairs_to_prompt_data), query
                                  from TEST
    test_mixedtask10.json         30 prompts; 10 demos each from a DIFFERENT random other
                                  task (all 28 others eligible), each demo a correct pair
                                  from that task's TRAIN examples; query from TEST

The SAME 30 test queries are used across the three test settings (paired design).
Records store structured pairs (not rendered text): render at use time with
word_pairs_to_prompt_data. Deterministic: RNG seeded per task.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.prompt_utils import load_dataset

N_TRAIN_PROMPTS = 150
N_TEST_PROMPTS = 30
N_SHOTS = 10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_split_path", type=Path,
                   default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--root_data_dir", type=str, default=str(REPO_ROOT / "dataset_files"))
    p.add_argument("--out_root", type=Path, default=REPO_ROOT / "dataset_files" / "isolation_prompts")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pairs(ds):
    """Examples deduplicated by input (first occurrence wins): a few datasets contain
    duplicate inputs (product-company) or same-input-different-output entries
    (national_parks multi-state units), which would otherwise yield repeated queries or
    contradictory demos within one prompt."""
    seen, out = set(), []
    for i in range(len(ds)):
        if ds[i]["input"] not in seen:
            seen.add(ds[i]["input"])
            out.append({"input": ds[i]["input"], "output": ds[i]["output"]})
    return out


def sample_demos(rng, pool, k, exclude_input):
    """k distinct examples from pool whose inputs differ from exclude_input."""
    idx = [i for i in range(len(pool)) if pool[i]["input"] != exclude_input]
    pick = rng.choice(len(idx), size=k, replace=False)
    return [dict(pool[idx[i]]) for i in pick]


def main():
    args = parse_args()
    split = json.load(open(args.task_split_path))
    tasks = split["train_tasks"] + split["test_tasks"]
    assert len(tasks) == 29

    train_pools, test_pools = {}, {}
    for t in tasks:
        d = load_dataset(t, root_data_dir=args.root_data_dir, seed=args.seed,
                         merge_valid_into_train=True)
        train_pools[t], test_pools[t] = pairs(d["train"]), pairs(d["test"])

    summary = {}
    for t in tasks:
        rng = np.random.RandomState(args.seed + (zlib.crc32(t.encode()) % 100000))
        tr, te = train_pools[t], test_pools[t]
        assert len(tr) >= N_TRAIN_PROMPTS, f"{t}: train pool {len(tr)} < {N_TRAIN_PROMPTS}"
        assert len(te) >= N_TEST_PROMPTS, f"{t}: test pool {len(te)} < {N_TEST_PROMPTS}"
        out_dir = args.out_root / t
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- 150 train prompts: distinct train queries, 10 correct train demos each ---
        q_idx = rng.choice(len(tr), size=N_TRAIN_PROMPTS, replace=False)
        train_prompts = []
        for j, qi in enumerate(q_idx):
            demos = sample_demos(rng, tr, N_SHOTS, tr[qi]["input"])
            train_prompts.append({"task": t, "setting": "train_10shot", "n_shots": N_SHOTS,
                                  "demos": demos, "query": dict(tr[qi]), "prompt_index": j})

        # --- 30 shared test queries (paired across the three settings) ---
        tq_idx = rng.choice(len(te), size=N_TEST_PROMPTS, replace=False)
        test_queries = [dict(te[i]) for i in tq_idx]

        zs = [{"task": t, "setting": "test_zeroshot", "n_shots": 0, "demos": [],
               "query": q, "prompt_index": j} for j, q in enumerate(test_queries)]

        shuf = []
        for j, q in enumerate(test_queries):
            demos = sample_demos(rng, tr, N_SHOTS, q["input"])
            perm = rng.permutation(N_SHOTS)  # plain permutation, fixed points allowed
            shuffled = [{"input": demos[i]["input"], "output": demos[perm[i]]["output"]}
                        for i in range(N_SHOTS)]
            shuf.append({"task": t, "setting": "test_sametask_shuffled10", "n_shots": N_SHOTS,
                         "demos": shuffled,
                         "demo_correct_outputs": [d["output"] for d in demos],
                         "query": q, "prompt_index": j})

        others = [o for o in tasks if o != t]
        mixed = []
        for j, q in enumerate(test_queries):
            demo_tasks = [others[i] for i in rng.choice(len(others), size=N_SHOTS, replace=False)]
            demos = []
            for dt in demo_tasks:
                pool = train_pools[dt]
                d = dict(pool[int(rng.randint(len(pool)))])
                d["source_task"] = dt
                demos.append(d)
            mixed.append({"task": t, "setting": "test_mixedtask10", "n_shots": N_SHOTS,
                          "demos": demos, "query": q, "prompt_index": j})

        for name, data in (("train_prompts", train_prompts), ("test_zeroshot", zs),
                           ("test_sametask_shuffled10", shuf), ("test_mixedtask10", mixed)):
            with open(out_dir / f"{name}.json", "w") as f:
                json.dump(data, f, indent=1)
        summary[t] = {"train_pool": len(tr), "test_pool": len(te),
                      "train_prompts": len(train_prompts), "test_prompts_per_setting": N_TEST_PROMPTS}
        print(f"{t:26s} train_pool={len(tr):4d} test_pool={len(te):4d} OK")

    meta = {
        "created_with": "dataset_files/generate/generate_isolation_prompts.py",
        "seed": args.seed,
        "example_split": "load_dataset(merge_valid_into_train=True): train 79% / test 21%, "
                         "test membership identical to the historical two-stage split",
        "train_prompts": f"{N_TRAIN_PROMPTS} fixed {N_SHOTS}-shot, demos+query from TRAIN, "
                         "correct labels, distinct queries",
        "test_settings": {
            "test_zeroshot": "no demos, query from TEST",
            "test_sametask_shuffled10": f"{N_SHOTS} TRAIN demos, labels permuted within prompt "
                                        "(plain permutation, fixed points allowed), query from TEST",
            "test_mixedtask10": f"{N_SHOTS} demos from {N_SHOTS} distinct OTHER tasks (all 28 "
                                "eligible), correct pairs from those tasks' TRAIN pools, "
                                "query from TEST",
        },
        "paired_design": "the SAME 30 test queries are used across the three test settings",
        "per_task": summary,
    }
    with open(args.out_root / "metadata.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nwrote {args.out_root}/metadata.json ({len(tasks)} tasks)")


if __name__ == "__main__":
    main()
