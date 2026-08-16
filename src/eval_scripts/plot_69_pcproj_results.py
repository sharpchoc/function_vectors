#!/usr/bin/env python
"""Aggregate + plot the sparse-PC-projection steering results (CPU).

Reads artifacts/69_task_run/pc_sparse/{selection.json, coeffs_final.pt, evals/<task>.json}
and the full-FV reference results (results/69_task_run/train_test_generalisation/
train_heldout_summary.csv). Writes to results/69_task_run/FV_dimensionality_analysis/:
  pc_sparse_summary.csv  per task: base/best/bestL x 3 settings for the PC-projected FV
                         + full-FV zs/mix/shuf best for comparison
  pc_sparse_bars.png     two-panel (train/heldout) zero-shot bars: projected vs full FV
  pc_selection.png       final c vs PC rank (singular-value order), selected PCs marked
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR  # noqa: E402

PC_ROOT = ARTIFACTS_ROOT / "69_task_run" / "pc_sparse"
OUT_DIR = TASK69_RUN_DIR / "FV_dimensionality_analysis"
REF_CSV = TASK69_RUN_DIR / "train_test_generalisation" / "train_heldout_summary.csv"
KEYS = {"zs": "test_zeroshot", "mix": "test_mixedtask10", "shuf": "test_sametask_shuffled10"}


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    sel = json.load(open(PC_ROOT / "selection.json"))
    ref = {r["task"]: r for r in csv.DictReader(open(REF_CSV))}
    rows = []
    for grp, tasks in (("train", split["train_tasks"]), ("heldout", split["heldout_tasks"])):
        for t in tasks:
            s = json.load(open(PC_ROOT / "evals" / f"{t}.json"))["settings"]
            row = {"task": t, "group": grp}
            for k, name in KEYS.items():
                accs = s[name]["acc_by_layer"]
                row[f"{k}_base"] = round(s[name]["baseline"], 4)
                row[f"{k}_best"] = round(max(accs), 4)
                row[f"{k}_bestL"] = int(np.argmax(accs))
                row[f"{k}_full_best"] = float(ref[t][f"{k}_best"])
            rows.append(row)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "pc_sparse_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n_pcs = sel["n_selected"]
    fig, axes = plt.subplots(2, 1, figsize=(20, 10.5), dpi=140,
                             gridspec_kw={"height_ratios": [55, 30]})
    for ax, grp, title in ((axes[0], "train", "TRAIN tasks (n=55)"),
                           (axes[1], "heldout", "HELD-OUT tasks (n=14)")):
        rs = sorted([r for r in rows if r["group"] == grp], key=lambda r: r["zs_best"])
        x = np.arange(len(rs))
        m_proj = np.mean([r["zs_best"] for r in rs])
        m_full = np.mean([r["zs_full_best"] for r in rs])
        m_base = np.mean([r["zs_base"] for r in rs])
        ax.bar(x, [r["zs_best"] for r in rs], width=0.72, color="tab:blue",
               label=f"steered, {n_pcs}-PC projection — mean {m_proj:.2f}")
        ax.plot(x, [r["zs_full_best"] for r in rs], "_", color="tab:orange", markersize=9,
                markeredgewidth=1.8, linestyle="none",
                label=f"full 37-head FV — mean {m_full:.2f}")
        ax.plot(x, [r["zs_base"] for r in rs], "k_", markersize=7, markeredgewidth=1.2,
                linestyle="none", label=f"no steering — mean {m_base:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels([r["task"] for r in rs], rotation=60, ha="right", fontsize=7.5)
        ax.set_ylabel("zero-shot full-label acc")
        ax.set_ylim(0, 1.02); ax.set_xlim(-0.7, len(rs) - 0.3)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_title(title, fontsize=11)
    fig.suptitle(f"Steering with the task FV projected onto {n_pcs} sparse-selected uncentered "
                 f"PCs (lambda={sel['chosen_lambda']}, pooled on 55 train tasks) vs the full "
                 "37-head FV — alpha=1, best layer, 50 queries/task", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pc_sparse_bars.png", bbox_inches="tight")
    plt.close(fig)

    c = torch.load(PC_ROOT / "coeffs_final.pt", map_location="cpu", weights_only=False)["c"].numpy()
    sel_idx = np.array(sel["selected_pcs"])
    fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
    ax.plot(np.arange(len(c)), c, ".", ms=3, color="#b8b8b8", label="final c (all 512 PCs)")
    ax.plot(sel_idx, c[sel_idx], "o", ms=5, color="tab:blue",
            label=f"selected (c > {sel['c_high']}): {n_pcs} PCs")
    ax.axhline(sel["c_high"], ls=":", lw=1, color="#888888")
    ax.set_xlabel("PC index (uncentered singular-value rank, 0 = top)")
    ax.set_ylabel("final coefficient c")
    ax.set_title(f"Which PCs steering selects — lam={sel['chosen_lambda']}, "
                 f"c_max={sel['c_max']}, selected ranks: min {sel_idx.min()}, "
                 f"median {int(np.median(sel_idx))}, max {sel_idx.max()}")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pc_selection.png", bbox_inches="tight")

    for grp in ("train", "heldout"):
        g = [r for r in rows if r["group"] == grp]
        print(f"{grp}: " + "; ".join(
            f"{k} proj {np.mean([r[f'{k}_best'] for r in g]):.3f} vs full "
            f"{np.mean([r[f'{k}_full_best'] for r in g]):.3f}" for k in KEYS))
    print(f"n_selected={n_pcs}; wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
