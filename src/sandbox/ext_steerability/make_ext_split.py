#!/usr/bin/env python
"""SANDBOX: build the extended-tasks steerable-90 train/heldout task split.

Filter: 6-shot accuracy >= 0.30 in the stored extended-tasks n-shot sweep
(results/exploratory/general/extended_tasks_nshot_sweep/nshot_accuracy.csv; full-label exact match on
T=1.0 samples, 50 prompts/task). Split: numpy RandomState(seed).permutation of the SORTED
filtered list, first 80% train / rest heldout. Hard gate: filtered count == 90.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import GENERAL_DIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nshot_csv", type=Path,
                   default=GENERAL_DIR / "extended_tasks_nshot_sweep" / "nshot_accuracy.csv")
    p.add_argument("--dataset_root", type=Path, default=REPO_ROOT / "dataset_files" / "extended_tasks")
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--n_shots", type=int, default=6)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expect_n", type=int, default=90)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "task_splits" / "extended_steerable_90.json")
    return p.parse_args()


def main():
    args = parse_args()
    acc6 = {}
    for r in csv.DictReader(open(args.nshot_csv)):
        if int(r["n_shots"]) == args.n_shots:
            acc6[r["task"]] = float(r["accuracy"])
    manifest = json.load(open(args.dataset_root / "manifest.json"))
    tasks_all = sorted(manifest["tasks"])
    missing = [t for t in tasks_all if t not in acc6]
    assert not missing, f"tasks missing from n-shot CSV: {missing}"

    filtered = sorted(t for t in tasks_all if acc6[t] >= args.threshold)
    assert len(filtered) == args.expect_n, \
        f"SPLIT GATE FAILED: {len(filtered)} tasks pass >= {args.threshold} at n={args.n_shots}, expected {args.expect_n}"

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(filtered))
    n_train = int(round(args.train_frac * len(filtered)))
    train = sorted(filtered[i] for i in perm[:n_train])
    heldout = sorted(filtered[i] for i in perm[n_train:])

    counts = {t: len(json.load(open(args.dataset_root / f"{t}.json"))) for t in filtered}
    out = {
        "name": "extended_steerable_90",
        "description": f"Extended tasks with 6-shot sampled-exact-match accuracy >= {args.threshold} "
                       f"(n-shot sweep, T=1.0, 50 prompts/task), split {args.train_frac:.0%} train via "
                       f"np.RandomState({args.seed}).permutation over the sorted filtered list.",
        "source_csv": str(args.nshot_csv.relative_to(REPO_ROOT)),
        "filter": {"n_shots": args.n_shots, "threshold": args.threshold},
        "seed": args.seed,
        "n_filtered": len(filtered),
        "n_train": len(train),
        "n_heldout": len(heldout),
        "train_tasks": train,
        "heldout_tasks": heldout,
        "acc6": {t: acc6[t] for t in filtered},
        "n_examples": counts,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    small = {t: c for t, c in counts.items() if c < 200}
    print(f"filtered {len(filtered)} -> train {len(train)} / heldout {len(heldout)}; wrote {args.out}")
    print(f"tasks with <200 examples (prompt counts will be capped): {small}")


if __name__ == "__main__":
    main()
