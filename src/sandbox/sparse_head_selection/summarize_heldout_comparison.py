#!/usr/bin/env python
"""SANDBOX: tabulate vanilla_sparse_opt23 vs varicl top-N best-layer results on the 9
held-out test tasks (reads the sweep + sparse summaries; writes a csv + prints)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import STEERING_COMPARISON_DIR

root = STEERING_COMPARISON_DIR / "heldout_varicl_nheads_sweep"
sweep = json.loads((root / "nheads_sweep_summary.json").read_text())
sparse = json.loads((root / "vanilla_sparse_opt23_summary.json").read_text())

out_rows = []
for metric, key in [("zs", "best_zs_intervention_top1"), ("fs_shuffled", "best_fs_shuffled_intervention_top1")]:
    print(f"\n=== best {metric} over layers ===")
    print(f"{'task':24s} {'v10':>6s} {'v20':>6s} {'v30':>6s} {'v40':>6s} {'sparse23':>8s}")
    cols = {n: [] for n in ["10", "20", "30", "40", "s"]}
    for t in sweep["tasks"]:
        b = sweep["per_task"][t]["best_by_n"]
        s = sparse["per_task"][t]
        vals = [b[n][key] for n in ["10", "20", "30", "40"]]
        for n, v in zip(["10", "20", "30", "40"], vals):
            cols[n].append(v)
        cols["s"].append(s[key])
        print(f"{t:24s} " + " ".join(f"{v:6.3f}" for v in vals) + f" {s[key]:8.3f}")
        out_rows.append({"metric": metric, "task": t,
                         **{f"varicl_top{n}": b[n][key] for n in ["10", "20", "30", "40"]},
                         "vanilla_sparse_opt23": s[key]})
    means = [float(np.mean(cols[n])) for n in ["10", "20", "30", "40", "s"]]
    print(f"{'MEAN':24s} " + " ".join(f"{v:6.3f}" for v in means[:4]) + f" {means[4]:8.3f}")
    out_rows.append({"metric": metric, "task": "MEAN",
                     **{f"varicl_top{n}": round(m, 4) for n, m in zip(["10", "20", "30", "40"], means[:4])},
                     "vanilla_sparse_opt23": round(means[4], 4)})

out = root / "vanilla_sparse_opt23_vs_topN.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)
print(f"\nwrote {out}")
