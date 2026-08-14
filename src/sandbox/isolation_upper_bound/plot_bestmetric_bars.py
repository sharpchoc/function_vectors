#!/usr/bin/env python
"""SANDBOX: per-task bar charts of best-layer steering accuracy, one figure per test setting.

For each task and intervention type (cie10 / cie40 / sparse), the bar shows the BEST-layer
accuracy under the TRAIN metric that performs best on that test setting; the bar's fill
color encodes which train metric that was. mean_act (no train metric) and the unsteered
baseline get their own bars. Hatch encodes the intervention type; tasks sorted ascending
by their best intervention bar. Reads results/sandbox/isolation_upper_bound/summary.csv.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import RESULTS_ROOT

SETTINGS = ["test_zeroshot", "test_sametask_shuffled10", "test_mixedtask10"]
TM_COLOR = {"zeroshot": "tab:blue", "sametask_shuffled10": "tab:orange",
            "mixedtask10": "tab:green"}
TM_LABEL = {"zeroshot": "trained: zero-shot", "sametask_shuffled10": "trained: same-task shuffled-10",
            "mixedtask10": "trained: mixed-task-10"}
PRODUCTS = [("cie10", ""), ("cie40", "//"), ("sparse", "xx")]
MEAN_COLOR, BASE_COLOR = "tab:purple", "0.35"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary_csv", type=Path,
                   default=RESULTS_ROOT / "sandbox" / "isolation_upper_bound" / "summary.csv")
    p.add_argument("--out_dir", type=Path,
                   default=RESULTS_ROOT / "sandbox" / "isolation_upper_bound" / "figures")
    return p.parse_args()


def main():
    args = parse_args()
    rows = list(csv.DictReader(open(args.summary_csv)))
    data = defaultdict(dict)   # (setting, task) -> {product: (acc, train_metric)}
    baselines = {}
    for r in rows:
        key = (r["test_setting"], r["task"])
        acc = float(r["best_acc"])
        baselines[key] = float(r["baseline"])
        prod, tm = r["product"], r["train_metric"]
        if prod == "mean_act":
            data[key]["mean_act"] = (acc, None)
        else:
            if prod not in data[key] or acc > data[key][prod][0]:
                data[key][prod] = (acc, tm)

    for setting in SETTINGS:
        tasks = sorted({t for s, t in data if s == setting},
                       key=lambda t: max(data[(setting, t)][p][0] for p, _ in PRODUCTS))
        n = len(tasks)
        x = np.arange(n)
        width = 0.17
        fig, ax = plt.subplots(figsize=(21, 6.5), dpi=140)
        for j, (prod, hatch) in enumerate(PRODUCTS):
            for i, t in enumerate(tasks):
                acc, tm = data[(setting, t)][prod]
                ax.bar(x[i] + (j - 2) * width, acc, width=width, color=TM_COLOR[tm],
                       hatch=hatch, edgecolor="white", linewidth=0.4)
        ax.bar(x + width, [data[(setting, t)]["mean_act"][0] for t in tasks], width=width,
               color=MEAN_COLOR, edgecolor="white", linewidth=0.4)
        ax.bar(x + 2 * width, [baselines[(setting, t)] for t in tasks], width=width,
               color=BASE_COLOR, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8.5)
        ax.set_ylabel("best-layer full-label accuracy (alpha=1)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, axis="y")
        color_handles = [Patch(facecolor=c, label=TM_LABEL[m]) for m, c in TM_COLOR.items()]
        color_handles += [Patch(facecolor=MEAN_COLOR, label="mean act (no training)"),
                          Patch(facecolor=BASE_COLOR, label="no-steering baseline")]
        hatch_handles = [Patch(facecolor="0.85", hatch=h, label=p) for p, h in PRODUCTS]
        leg1 = ax.legend(handles=color_handles, loc="upper left", fontsize=8.5,
                         title="bar color = best train metric", title_fontsize=8.5)
        ax.add_artist(leg1)
        ax.legend(handles=hatch_handles, loc="upper left", bbox_to_anchor=(0.24, 1.0),
                  fontsize=8.5, title="hatch = intervention", title_fontsize=8.5)
        ax.set_title(f"{setting.replace('test_', 'test setting: ')} — per task, each intervention "
                     f"at its best train metric; bars per task: cie10 | cie40 | sparse | mean_act | "
                     f"baseline (SANDBOX isolation upper bound)", fontsize=11)
        fig.tight_layout()
        out = args.out_dir / f"bars_{setting}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
