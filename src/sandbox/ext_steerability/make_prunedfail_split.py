#!/usr/bin/env python
"""Steering-prune + re-split: drop the parent split's tasks whose best-layer zero-shot FV steering
accuracy is below a threshold, then re-split the survivors 80/20 with a fresh seed.

Re-implementation of the deleted one-off that built task_splits/qwen25_ext_steerable_96_prunedfail.json
(2026-09-12): parent = qwen25_ext_steerable_104.json, summary = round1_104/train_heldout_summary.csv,
rule zs_best < 0.4, seed 43, train_frac 0.8 → 77/19. Regenerating that file with the defaults below
must reproduce it exactly (checked in the base-model port, 2026-09-23).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import REPO_ROOT, QWEN25_FV_DIR  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent_split", type=Path,
                    default=REPO_ROOT / "task_splits" / "qwen25_ext_steerable_104.json")
    ap.add_argument("--summary_csv", type=Path,
                    default=QWEN25_FV_DIR / "round1_104" / "train_heldout_summary.csv",
                    help="aggregate_eval_headset.py output with a zs_best column")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--dataset_root", type=Path, default=REPO_ROOT / "dataset_files" / "extended_tasks")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: task_splits/<parent stem with the count replaced>_<N>_prunedfail.json")
    return ap.parse_args()


def main():
    args = parse_args()
    parent = json.load(open(args.parent_split))
    tasks_all = sorted(parent["train_tasks"] + parent["heldout_tasks"])
    zs = {r["task"]: float(r["zs_best"]) for r in csv.DictReader(open(args.summary_csv))}
    missing = [t for t in tasks_all if t not in zs]
    assert not missing, f"tasks missing from summary: {missing}"
    removed = sorted(t for t in tasks_all if zs[t] < args.threshold)
    kept = sorted(t for t in tasks_all if zs[t] >= args.threshold)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(kept))
    n_train = int(round(args.train_frac * len(kept)))
    train = sorted(kept[i] for i in perm[:n_train])
    heldout = sorted(kept[i] for i in perm[n_train:])
    if args.out is None:
        stem = args.parent_split.stem.rsplit("_", 1)[0]          # e.g. qwen25_ext_steerable
        args.out = args.parent_split.parent / f"{stem}_{len(kept)}_prunedfail.json"
    counts = {t: len(json.load(open(args.dataset_root / f"{t}.json"))) for t in kept}
    out = {
        "name": args.out.stem,
        "description": f"{parent.get('name', args.parent_split.stem)} minus the {len(removed)} tasks whose "
                       f"best-layer zero-shot FV steering accuracy is < {args.threshold} "
                       f"({args.summary_csv.name}); survivors re-split {args.train_frac:.0%} train via "
                       f"np.RandomState({args.seed}).permutation over the sorted list.",
        "source": str(args.summary_csv.relative_to(REPO_ROOT)) if args.summary_csv.is_relative_to(REPO_ROOT) else str(args.summary_csv),
        "parent_split": str(args.parent_split.relative_to(REPO_ROOT)) if args.parent_split.is_relative_to(REPO_ROOT) else str(args.parent_split),
        "removed_failures": removed,
        "seed": args.seed,
        "n_filtered": len(kept),
        "n_train": len(train),
        "n_heldout": len(heldout),
        "train_tasks": train,
        "heldout_tasks": heldout,
        "acc6": {t: parent["acc6"][t] for t in kept} if "acc6" in parent else {},
        "n_examples": counts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"removed {len(removed)}: {removed}\nkept {len(kept)} -> {len(train)} train / {len(heldout)} heldout -> {args.out}")


if __name__ == "__main__":
    main()
