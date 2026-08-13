#!/usr/bin/env python
"""SANDBOX: figures + summary for the task-specific isolation upper-bound sweep (CPU).

Per task: ONE 3x3-grid PNG - rows = train metric {zeroshot, sametask_shuffled10,
mixedtask10}, cols = test setting; each panel: accuracy vs injection layer (alpha=1) with
lines for cie_top10 / cie_top40 / sparse_opt (that row's train metric) and mean_act (no
train metric, identical across rows), plus the dotted unsteered baseline (identical across
rows). Also writes summary.csv with best-layer accuracies for every cell.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT

TRAIN_METRICS = ["zeroshot", "sametask_shuffled10", "mixedtask10"]
TEST_SETTINGS = ["test_zeroshot", "test_sametask_shuffled10", "test_mixedtask10"]
LINES = [("cie10", "tab:blue", "CIE top-10"), ("cie40", "tab:cyan", "CIE top-40"),
         ("sparse", "tab:orange", "sparse opt (c>0.8)")]
MEAN_COLOR, MEAN_LABEL = "tab:green", "mean act (per-layer)"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_root", type=Path, default=ARTIFACTS_ROOT / "sandbox" / "isolation_upper_bound")
    p.add_argument("--out_root", type=Path, default=RESULTS_ROOT / "sandbox" / "isolation_upper_bound")
    p.add_argument("--tasks", nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    tasks = args.tasks or sorted(d.name for d in args.in_root.iterdir()
                                 if (d / "eval_results.json").exists())
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "figures").mkdir(exist_ok=True)
    rows = []
    for task in tasks:
        with open(args.in_root / task / "eval_results.json") as f:
            res = json.load(f)
        layers = res["layers"]
        fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
        for r, m in enumerate(TRAIN_METRICS):
            for c, setting in enumerate(TEST_SETTINGS):
                ax = axes[r, c]
                entry = res["settings"][setting]
                for key, color, label in LINES:
                    accs = entry["products"][f"{key}|{m}"]
                    ax.plot(layers, accs, color=color, lw=1.6,
                            label=label if (r, c) == (0, 0) else None)
                    rows.append({"task": task, "train_metric": m, "test_setting": setting,
                                 "product": key, "best_layer": int(np.argmax(accs)),
                                 "best_acc": max(accs), "acc_L9": accs[9],
                                 "baseline": entry["baseline"]})
                ma = entry["products"]["mean_act"]
                ax.plot(layers, ma, color=MEAN_COLOR, lw=1.6,
                        label=MEAN_LABEL if (r, c) == (0, 0) else None)
                if r == 0:
                    rows.append({"task": task, "train_metric": "-", "test_setting": setting,
                                 "product": "mean_act", "best_layer": int(np.argmax(ma)),
                                 "best_acc": max(ma), "acc_L9": ma[9],
                                 "baseline": entry["baseline"]})
                ax.axhline(entry["baseline"], color="k", ls=":", lw=1.2,
                           label="no steering" if (r, c) == (0, 0) else None)
                if r == 0:
                    ax.set_title(setting.replace("test_", "test: "), fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"train: {m}\nfull-label acc", fontsize=9)
                if r == 2:
                    ax.set_xlabel("injection layer")
                ax.grid(alpha=0.25)
                ax.set_ylim(-0.02, 1.02)
        fig.legend(loc="upper center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(f"{task} — task-specific isolation products, alpha=1, 30 paired test "
                     f"prompts (SANDBOX isolation upper bound)", fontsize=12, y=1.02)
        fig.tight_layout()
        out = args.out_root / "figures" / f"{task}_grid.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.name}")

    with open(args.out_root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "train_metric", "test_setting", "product",
                                          "best_layer", "best_acc", "acc_L9", "baseline"])
        w.writeheader()
        for r in rows:
            r["best_acc"] = f"{r['best_acc']:.4f}"
            r["acc_L9"] = f"{r['acc_L9']:.4f}"
            r["baseline"] = f"{r['baseline']:.4f}"
            w.writerow(r)
    print(f"wrote summary.csv ({len(rows)} rows, {len(tasks)} tasks)")


if __name__ == "__main__":
    main()
