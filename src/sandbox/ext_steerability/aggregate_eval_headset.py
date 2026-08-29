#!/usr/bin/env python
"""Aggregate per-task eval_headset.json files (eval_ext.py output) into a train/heldout
summary CSV — the committed version of the inline aggregation used for
results/69_task_run/FV_train_test_generalisation/train_heldout_summary.csv.

Per task and test setting: baseline, best-layer accuracy (max over the layer sweep), argmax layer.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SETTINGS = {"test_zeroshot": "zs", "test_sametask_shuffled10": "shuf", "test_mixedtask10": "mix"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split_path", type=Path, required=True)
    ap.add_argument("--eval_root", type=Path, required=True,
                    help="dir with <task>/eval_headset.json")
    ap.add_argument("--out_name", type=str, default="eval_headset.json")
    ap.add_argument("--out_csv", type=Path, required=True)
    args = ap.parse_args()

    split = json.load(open(args.split_path))
    rows, missing = [], []
    for group, tasks in (("train", split["train_tasks"]), ("heldout", split["heldout_tasks"])):
        for t in sorted(tasks):
            f = args.eval_root / t / args.out_name
            if not f.exists():
                missing.append(t)
                continue
            d = json.load(open(f))
            row = {"task": t, "group": group, "n_heads": d["n_heads"]}
            for setting, tag in SETTINGS.items():
                s = d["settings"][setting]
                accs = s["acc_by_layer"]
                row[f"{tag}_base"] = round(s["baseline"], 4)
                row[f"{tag}_best"] = round(max(accs), 4)
                row[f"{tag}_bestL"] = int(accs.index(max(accs)))
            rows.append(row)
    if missing:
        print(f"WARNING: {len(missing)} tasks missing eval files: {missing[:8]}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["task", "group", "n_heads"] + [f"{t}_{k}" for t in SETTINGS.values()
                                           for k in ("base", "best", "bestL")]
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    for group in ("train", "heldout"):
        g = [r for r in rows if r["group"] == group]
        if not g:
            continue
        line = f"{group} (n={len(g)}):"
        for tag in SETTINGS.values():
            base = sum(r[f"{tag}_base"] for r in g) / len(g)
            best = sum(r[f"{tag}_best"] for r in g) / len(g)
            line += f"  {tag} {base:.3f}->{best:.3f}"
        print(line)
    print(f"wrote {args.out_csv} ({len(rows)} tasks)")


if __name__ == "__main__":
    main()
