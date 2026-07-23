#!/usr/bin/env python
"""SANDBOX: best-over-layers test R² per token position, canonical vs per-prompt targets.

Line-plot companion to plot_pilot_heatmaps.py's side-by-side heatmaps: for every token
position (icl/role row) take max over layers of test_r2_fv for both the canonical
FV-broadcast run and the SANDBOX per-prompt head-sum run, so the two methods can be
compared at a glance. Output lands next to summary_vs_canonical_all.csv.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_scripts.merge_fulldim_ridge_results import position_key, position_label  # noqa: E402
from utils.paths import RESULTS_ROOT  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_csv", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/gptj_train_varicl_top40/summary_vs_canonical_all.csv")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.summary_csv.parent
    with open(args.summary_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    pos_set = sorted({(int(r["icl_index"]), r["token_role"]) for r in rows}, key=lambda ir: position_key(*ir))
    pos_index = {pos: i for i, pos in enumerate(pos_set)}
    pos_labels = [position_label(icl, role) for icl, role in pos_set]

    best = {k: np.full(len(pos_set), np.nan) for k in ("new", "old")}
    best_layer = {k: np.full(len(pos_set), -1, dtype=int) for k in ("new", "old")}
    for r in rows:
        i = pos_index[(int(r["icl_index"]), r["token_role"])]
        for k, col in (("new", "new_test_r2_fv"), ("old", "old_test_r2_fv")):
            if r[col] in ("", "None"):
                continue
            v = float(r[col])
            if not np.isnan(v) and (np.isnan(best[k][i]) or v > best[k][i]):
                best[k][i] = v
                best_layer[k][i] = int(r["layer"])

    x = np.arange(len(pos_set))
    fig, ax = plt.subplots(figsize=(max(10, len(pos_set) * 0.4), 5.5))
    ax.plot(x, best["old"], marker="o", ms=4, lw=1.5, color="tab:blue",
            label="canonical: FV-broadcast targets")
    ax.plot(x, best["new"], marker="s", ms=4, lw=1.5, color="tab:orange",
            label="SANDBOX: per-prompt head-sum targets")
    for xb in x[:-1]:
        if pos_set[xb + 1][0] != pos_set[xb][0]:  # shot boundary
            ax.axvline(xb + 0.5, color="0.85", lw=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(pos_labels, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("token position (icl/role)")
    ax.set_ylabel("best test R² over layers (vs stored varicl_top40 test FVs)")
    ax.set_title("GPT-J full-dim ridge (4096 → 4096): best-over-layers test R² per token position")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path = out_dir / "best_r2_per_position_lines.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    print(f"{'position':<16}{'canonical':>12}{'@L':>4}{'per-prompt':>12}{'@L':>4}")
    for i, lab in enumerate(pos_labels):
        print(f"{lab:<16}{best['old'][i]:>12.3f}{best_layer['old'][i]:>4d}"
              f"{best['new'][i]:>12.3f}{best_layer['new'][i]:>4d}")


if __name__ == "__main__":
    main()
