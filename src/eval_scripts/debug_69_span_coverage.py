#!/usr/bin/env python
"""Debug check: how much of each task FV lies inside the span of the train-task
per-prompt FV PCs (CPU).

For every task (55 train / 14 held-out): v = mean per-prompt FV; coverage_k =
||P_k v||^2 / ||v||^2 where P_k projects onto the top-k uncentered PCs of the TRAIN
per-prompt stack (k grid up to 512), plus coverage in the 46 sparse-SELECTED PCs.
Tests the explanation that held-out steering collapses because held-out FV directions
stick out of the train span. Writes results/69_task_run/FV_dimensionality_reduction/
debugging/{span_coverage.csv, span_coverage.png}.
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
PP_ROOT = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
OUT_DIR = TASK69_RUN_DIR / "FV_dimensionality_reduction" / "train_test_split" / "debugging"
K_GRID = [6, 12, 24, 46, 64, 128, 256, 512]


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    basis = torch.load(PC_ROOT / "pc_basis_uncentered.pt", map_location="cpu", weights_only=False)
    pcs = basis["pcs"].double().numpy()  # (512, 4096) orthonormal rows
    sel_idx = np.array(json.load(open(PC_ROOT / "selection.json"))["selected_pcs"])
    ref = {r["task"]: r for r in csv.DictReader(
        open(TASK69_RUN_DIR / "FV_dimensionality_reduction" / "train_test_split" / "pc_sparse_summary.csv"))}

    rows = []
    for grp, tasks in (("train", split["train_tasks"]), ("heldout", split["heldout_tasks"])):
        for t in tasks:
            v = torch.load(PP_ROOT / f"{t}.pt", map_location="cpu",
                           weights_only=False)["fv"].double().mean(dim=0).numpy()
            e = v @ v
            coef = pcs @ v  # (512,)
            row = {"task": t, "group": grp,
                   "cov_selected46": round(float((coef[sel_idx] ** 2).sum() / e), 4)}
            for k in K_GRID:
                row[f"cov_top{k}"] = round(float((coef[:k] ** 2).sum() / e), 4)
            row["zs_full"] = float(ref[t]["zs_full_best"])
            row["zs_proj"] = float(ref[t]["zs_best"])
            rows.append(row)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "span_coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    tr = [r for r in rows if r["group"] == "train"]
    ho = [r for r in rows if r["group"] == "heldout"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), dpi=150)

    ax = axes[0]  # coverage vs k curves
    for r in tr:
        ax.plot(K_GRID, [r[f"cov_top{k}"] for k in K_GRID], "-", lw=0.7, alpha=0.3,
                color="tab:orange")
    for r in ho:
        ax.plot(K_GRID, [r[f"cov_top{k}"] for k in K_GRID], "-o", ms=3, lw=1.3,
                alpha=0.9, color="tab:blue")
    ax.plot([], [], "-", color="tab:orange", label="train tasks (55)")
    ax.plot([], [], "-o", color="tab:blue", label="held-out tasks (14)")
    ax.set_xscale("log"); ax.set_xticks(K_GRID); ax.set_xticklabels(K_GRID)
    ax.axvline(46, ls=":", lw=1, color="#888888")
    ax.set_xlabel("k (top-k train PCs)"); ax.set_ylabel("energy of task FV inside span")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.set_title("(A) FV energy captured by top-k train-PC span")

    ax = axes[1]  # heldout bars at k=512, sorted
    hs = sorted(ho, key=lambda r: r["cov_top512"])
    x = np.arange(len(hs))
    ax.bar(x, [r["cov_top512"] for r in hs], width=0.55, color="tab:blue",
           label="top-512 span")
    ax.plot(x, [r["cov_selected46"] for r in hs], "_", color="tab:red", markersize=11,
            markeredgewidth=2, linestyle="none", label="46 selected PCs")
    tr512 = [r["cov_top512"] for r in tr]
    ax.axhline(np.mean(tr512), ls="--", lw=1, color="tab:orange",
               label=f"train mean, top-512 ({np.mean(tr512):.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([r["task"] for r in hs], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("energy inside span"); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, axis="y"); ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("(B) held-out FV coverage")

    ax = axes[2]  # coverage vs zs retention scatter
    for rs, color, lbl in ((tr, "tab:orange", "train"), (ho, "tab:blue", "held-out")):
        ret = [r["zs_proj"] - r["zs_full"] for r in rs]
        cov = [r["cov_selected46"] for r in rs]
        ax.plot(cov, ret, "o", ms=5, alpha=0.75, color=color, label=lbl)
    from scipy.stats import spearmanr
    cov_all = [r["cov_selected46"] for r in ho]
    ret_all = [r["zs_proj"] - r["zs_full"] for r in ho]
    rho = spearmanr(cov_all, ret_all).statistic
    ax.set_xlabel("FV energy inside the 46 selected PCs")
    ax.set_ylabel("zs acc change (projected - full)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.set_title(f"(C) coverage vs steering loss (held-out Spearman {rho:.2f})")

    fig.suptitle("Is the held-out collapse explained by FVs sticking out of the train "
                 "per-prompt FV span? (uncentered train-PC basis, energy fractions in "
                 "float64)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "span_coverage.png", bbox_inches="tight")

    print(f"train  cov_top512: mean {np.mean([r['cov_top512'] for r in tr]):.4f} "
          f"min {min(r['cov_top512'] for r in tr):.4f}")
    print(f"heldout cov_top512: mean {np.mean([r['cov_top512'] for r in ho]):.4f} "
          f"min {min(r['cov_top512'] for r in ho):.4f}")
    print(f"heldout cov_selected46: mean {np.mean([r['cov_selected46'] for r in ho]):.4f}")
    print(f"heldout spearman(cov46, zs change) = {rho:.3f}")
    for r in hs:
        print(f"  {r['task']:22s} top512={r['cov_top512']:.3f} sel46={r['cov_selected46']:.3f} "
              f"zs {r['zs_full']:.2f}->{r['zs_proj']:.2f}")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
