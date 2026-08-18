#!/usr/bin/env python
"""Per-task R^2 bar chart for the label-token -> FV ridge (avg-of-10 X, full-dim).

R^2 convention (user spec 2026-08-18): for each task, the denominator variance is computed
against the MEAN OF THE TASK'S SPLIT POOL (all 55 train tasks' pairs for train tasks; all
14 held-out tasks' pairs for held-out tasks), NOT the task's own mean — so credit for
placing a task's predictions in the right region of FV space shows up per task. Uniform
average over the 4096 dims; dims with zero pooled variance dropped.

Model: the CV-chosen avg-X ridge (alpha = 1e3) refit on all train pairs, as in
ridge_labeltoken_to_fv.py.

Outputs in results/69_task_run/labeltoken_fv_ridge/:
  r2_by_task.png, r2_by_task.csv
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

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR, REPO_ROOT  # noqa: E402
from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (  # noqa: E402
    ridge_eig_prep, ridge_predict)

ACTS = ARTIFACTS_ROOT / "69_task_run" / "label_all10_L6_acts"
FVS = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
OUT = TASK69_RUN_DIR / "labeltoken_fv_ridge"
ALPHA = 1000.0  # CV-chosen for the avg-of-10 X variant


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    train_tasks, test_tasks = sorted(split["train_tasks"]), sorted(split["heldout_tasks"])

    def load(t):
        a = torch.load(ACTS / f"{t}.pt", map_location="cpu", weights_only=False)
        f = torch.load(FVS / f"{t}.pt", map_location="cpu", weights_only=False)
        assert list(a["prompt_index"]) == list(f["prompt_index"])
        return a["acts"].double().mean(dim=1), f["fv"].double()

    X, Y, sl = {}, {}, {}
    for grp, tl in (("train", train_tasks), ("heldout", test_tasks)):
        xs, ys, pos = [], [], 0
        for t in tl:
            x, y = load(t)
            xs.append(x); ys.append(y)
            sl[t] = (grp, pos, pos + len(y)); pos += len(y)
        X[grp] = torch.cat(xs); Y[grp] = torch.cat(ys)

    xbar, ybar, ev, evec, c = ridge_eig_prep(X["train"], Y["train"])
    P = {g: ridge_predict(X[g], xbar, ybar, ev, evec, c, ALPHA) for g in X}
    pool_mean = {g: Y[g].mean(dim=0) for g in Y}   # split-pool reference mean
    pool_tot = {g: ((Y[g] - pool_mean[g]) ** 2).sum(dim=0) for g in Y}

    rows = []
    for t in train_tasks + test_tasks:
        g, s, e = sl[t]
        resid = ((Y[g][s:e] - P[g][s:e]) ** 2).sum(dim=0)
        # per-dim total vs the SPLIT-POOL mean, scaled to this task's row count
        tot = ((Y[g][s:e] - pool_mean[g]) ** 2).sum(dim=0)
        ok = tot > 0
        r2 = float((1 - resid[ok] / tot[ok]).mean())
        rows.append((t, g, r2))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "r2_by_task.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "split", "r2_uniform_poolref"])
        w.writerows([(t, g, round(r, 4)) for t, g, r in rows])

    rows.sort(key=lambda r: r[2])
    labels = [t + (" *" if g == "heldout" else "") for t, g, _ in rows]
    vals = [r for _, _, r in rows]
    cols = ["tab:red" if g == "heldout" else "tab:blue" for _, g, _ in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(15, 0.3 * len(rows)), 6.6), dpi=150)
    ax.bar(x, vals, color=cols)
    ax.axhline(0, color="0.4", lw=0.9)
    tr = [r for _, g, r in rows if g == "train"]
    te = [r for _, g, r in rows if g == "heldout"]
    ax.axhline(float(np.mean(tr)), color="tab:blue", ls="--", lw=1.1,
               label=f"train mean = {np.mean(tr):.3f}")
    ax.axhline(float(np.mean(te)), color="tab:red", ls="--", lw=1.1,
               label=f"held-out mean = {np.mean(te):.3f}")
    ax.set_xticks(x, labels, rotation=90, fontsize=6.4)
    ax.set_ylabel("per-task R^2 (uniform over dims; denominator = split-pool mean)")
    ax.set_title("Label-token -> FV ridge (avg-of-10 X, alpha=1e3): per-task R^2\n"
                 "(blue = train tasks, in-sample; red * = held-out tasks)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "r2_by_task.png", bbox_inches="tight")
    print(f"train mean {np.mean(tr):.4f} | heldout mean {np.mean(te):.4f} | "
          f"min task {rows[0][0]} {rows[0][2]:.3f} | max task {rows[-1][0]} {rows[-1][2]:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
