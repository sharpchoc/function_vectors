#!/usr/bin/env python
"""69-task-run train/heldout generalisation bars (CPU).

For each test setting (zeroshot, mixedtask10, sametask_shuffled10): two panels
(55 train / 14 held-out tasks), steered best-layer accuracy ascending with the
37-head pooled FV (prunedfail_seed43 selection, alpha=1), unsteered baselines
as black dashes. Reads results/69_task_run/FV_train_test_generalisation/
train_heldout_summary.csv (from eval_headset.json aggregation).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import TASK69_RUN_DIR  # noqa: E402

OUT_DIR = TASK69_RUN_DIR / "FV_train_test_generalisation"
SETTINGS = [
    ("zs", "zeroshot_bars.png", "zero-shot"),
    ("mix", "mixedtask10_bars.png", "mixed-task mixed-label 10-shot"),
    ("shuf", "shuffled10_bars.png", "same-task shuffled-label 10-shot"),
]


def main():
    rows = list(csv.DictReader(open(OUT_DIR / "train_heldout_summary.csv")))
    groups = {"train": [r for r in rows if r["group"] == "train"],
              "heldout": [r for r in rows if r["group"] == "heldout"]}
    for key, fname, label in SETTINGS:
        fig, axes = plt.subplots(2, 1, figsize=(20, 10.5), dpi=140,
                                 gridspec_kw={"height_ratios": [55, 30]})
        titles = [f"TRAIN tasks (n={len(groups['train'])}) — heads were selected on these",
                  f"HELD-OUT tasks (n={len(groups['heldout'])}) — never seen by the selection"]
        for ax, (grp, rs), title in zip(axes, groups.items(), titles):
            rs = sorted(rs, key=lambda r: float(r[f"{key}_best"]))
            x = np.arange(len(rs))
            mean_steer = np.mean([float(r[f"{key}_best"]) for r in rs])
            mean_base = np.mean([float(r[f"{key}_base"]) for r in rs])
            ax.bar(x, [float(r[f"{key}_best"]) for r in rs], width=0.72, color="tab:blue",
                   label=f"steered, best layer (37-head pooled FV) — mean {mean_steer:.2f}")
            ax.plot(x, [float(r[f"{key}_base"]) for r in rs], "k_", markersize=9,
                    markeredgewidth=1.6, linestyle="none",
                    label=f"no steering — mean {mean_base:.2f}")
            ax.set_xticks(x)
            ax.set_xticklabels([r["task"] for r in rs], rotation=60, ha="right", fontsize=7.5)
            ax.set_ylabel(f"{label} full-label acc")
            ax.set_ylim(0, 1.02)
            ax.set_xlim(-0.7, len(rs) - 0.3)
            ax.grid(alpha=0.25, axis="y")
            ax.legend(fontsize=9, loc="upper left")
            ax.set_title(title, fontsize=11)
        fig.suptitle(f"Test setting: {label} — 69-task pool, pooled sparse head set "
                     "(37 heads, lambda=0.005, fit on the 55 train tasks, seed-43 split), "
                     "alpha=1, 50 queries/task, ascending by steered accuracy", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {OUT_DIR / fname}")


if __name__ == "__main__":
    main()
