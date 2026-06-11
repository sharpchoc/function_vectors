#!/usr/bin/env python
"""Merge per-task sweep_task_entry.json files into one summary + a best-layer CSV table."""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", type=Path, default=Path("results/heldout_varicl_nheads_sweep"))
    ap.add_argument("--n_values", nargs="+", type=int, default=[10, 20, 30, 40])
    args = ap.parse_args()

    entries = {}
    for f in sorted(args.output_root.glob("*/sweep_task_entry.json")):
        entries[f.parent.name] = json.loads(f.read_text())

    summary = {"tasks": sorted(entries), "n_values": args.n_values, "per_task": entries}
    (args.output_root / "nheads_sweep_summary.json").write_text(json.dumps(summary, indent=2))

    csv_path = args.output_root / "nheads_sweep_best_layer.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["task", "n_filtered"]
        for cond in ("zs", "fs"):
            header += [f"{cond}_top{n}" for n in args.n_values]
            header += [f"{cond}_mt_fixedicl", f"{cond}_task_specific"]
        w.writerow(header)
        for task in sorted(entries):
            e = entries[task]
            row = [task, e["n_filtered_test_examples"]]
            b = e["best_by_n"]
            bl = e.get("baselines", {})
            for cond, key, mtk, tsk in (
                ("zs", "best_zs_intervention_top1", "multitask_fixed_icl_best_zs", "task_specific_best_zs"),
                ("fs", "best_fs_shuffled_intervention_top1", "multitask_fixed_icl_best_fs", "task_specific_best_fs"),
            ):
                row += [round(b[str(n)][key], 4) for n in args.n_values]
                row += [bl.get(mtk), bl.get(tsk)]
            w.writerow(row)
    print(f"wrote {csv_path}")
    print(f"wrote {args.output_root / 'nheads_sweep_summary.json'}")


if __name__ == "__main__":
    main()
