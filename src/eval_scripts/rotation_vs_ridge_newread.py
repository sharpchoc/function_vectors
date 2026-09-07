#!/usr/bin/env python
"""Read→write map (Procrustes rotation vs ridge) for the NEW read-feature objects.

Same protocol as understanding_read_write_linear_map/rotation_vs_ridge.py (55 train /
14 held-out tasks, train-mean centering, ridge with LOO-CV lambda, orthogonal Procrustes,
optional one multiplicative scalar), but with X =
  mbar57  : the bank-(a) read feature averaged over layers 5-7,  mbar_A = mean_l m_A(l)
  unique  : the task-unique part n_A * v1_A  (v1 = top SVD dir of the carrier-removed L5-7
            means, n_A = <mbar_A - c, v1>; sign-invariant), i.e. exactly the object the
            ablation removes and the steering vector adds
  unique_traincarrier : same, but v1/n_A rebuilt with the carrier estimated from the 55
            TRAIN tasks only (leakage check: the default carrier uses all 69)
and the existing m_A(L6) reproduced as a reference row. Y = task FV (mean of 150 per-prompt FVs).

Writes results/69_task_run/understanding_read_write_linear_map/newread/
  fits_summary.csv, rotation_simple_{mbar57,unique}.png (4-bar main-text style)
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
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR  # noqa: E402

RM = ARTIFACTS_ROOT / "69_task_run" / "label_resid_means"
FV = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
BASES = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA" / "L5to7_top1_bases.pt"
OUT = TASK69_RUN_DIR / "understanding_read_write_linear_map" / "newread"
LAMBDA_GRID = np.logspace(-4, 4, 33)


def r2(pred, true, ref):
    resid = ((true - pred) ** 2).sum(0)
    tot = ((true - ref) ** 2).sum(0)
    ok = tot > 0
    return float((1 - resid[ok] / tot[ok]).mean())


def fit_all(X, Y, is_train):
    Xtr, Ytr, Xte, Yte = X[is_train], Y[is_train], X[~is_train], Y[~is_train]
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc, Xtec = Xtr - xm, Ytr - ym, Xte - xm
    n = len(Xc)
    K = Xc @ Xc.T
    best = (None, np.inf)
    for lam in LAMBDA_GRID:
        H = K @ np.linalg.inv(K + lam * np.eye(n))
        E = (Yc - H @ Yc) / (1.0 - np.diag(H))[:, None]
        m = (E ** 2).mean()
        if m < best[1]:
            best = (lam, m)
    lam = best[0]
    A = np.linalg.solve(K + lam * np.eye(n), Yc)
    pred_ridge = Xtec @ Xc.T @ A + ym
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Uy, Sy, Vty = np.linalg.svd(Yc, full_matrices=False)
    small = (S[:, None] * (U.T @ Uy)) * Sy[None, :]
    Us, Ss, Vts = np.linalg.svd(small)
    R = (Vt.T @ Us) @ (Vts @ Vty)
    scale = Ss.sum() / (S ** 2).sum()
    pred_rot = Xtec @ R + ym
    pred_rs = scale * (Xtec @ R) + ym
    ref = Yte.mean(0)
    return {"trainmean_baseline": r2(np.repeat(ym[None], len(Yte), 0), Yte, ref),
            "rotation": r2(pred_rot, Yte, ref), "rotation+scale": r2(pred_rs, Yte, ref),
            "ridge": r2(pred_ridge, Yte, ref), "lambda": lam, "scale": scale}


def build_unique(rm, tasks, carrier_tasks):
    """v1 (top SVD of per-layer carrier-removed unit residuals, layers 5-7) and n_A, with the
    per-layer carrier direction estimated over `carrier_tasks`."""
    Xl = np.stack([rm[t][[5, 6, 7]] for t in tasks])                    # (69, 3, d)
    Cl = np.stack([rm[t][[5, 6, 7]] for t in carrier_tasks]).mean(0)      # (3, d)
    cdirs = Cl / np.linalg.norm(Cl, axis=1, keepdims=True)
    Rl = Xl - (Xl * cdirs).sum(-1, keepdims=True) * cdirs
    mbar = Xl.mean(1)                                                    # (69, d)
    c = np.stack([rm[t][[5, 6, 7]].mean(0) for t in carrier_tasks]).mean(0)
    out = np.zeros_like(mbar)
    for i in range(len(tasks)):
        Un = Rl[i] / np.linalg.norm(Rl[i], axis=1, keepdims=True)
        _, _, Vh = np.linalg.svd(Un, full_matrices=False)
        v1 = Vh[0]
        nA = float((mbar[i] - c) @ v1)
        out[i] = nA * v1
    return out, mbar


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    is_train = np.array([t in set(split["train_tasks"]) for t in tasks])
    rm = {t: torch.load(RM / f"{t}.pt", map_location="cpu", weights_only=False)["resid_means"]
          .double().numpy() for t in tasks}
    Y = np.stack([torch.load(FV / f"{t}.pt", map_location="cpu", weights_only=False)["fv"]
                  .double().mean(0).numpy() for t in tasks])

    # sanity: stored bankA v1 agrees with our rebuild (all-69 carrier)
    stored = torch.load(BASES, map_location="cpu", weights_only=False)["tasks"]
    uniq_all, mbar = build_unique(rm, tasks, tasks)
    cos_chk = np.median([abs(uniq_all[i] / np.linalg.norm(uniq_all[i]) @ stored[t]["V"][0].double().numpy())
                         for i, t in enumerate(tasks)])
    print(f"rebuild check: median |cos(v1_rebuilt, v1_stored)| = {cos_chk:.4f}")
    uniq_tr, _ = build_unique(rm, tasks, [t for t in tasks if t in set(split["train_tasks"])])

    Xs = {"m_A(L6)": np.stack([rm[t][6] for t in tasks]),
          "mbar57": mbar,
          "unique": uniq_all,
          "unique_traincarrier": uniq_tr}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    with open(OUT / "fits_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["X", "trainmean_baseline", "rotation", "rotation+scale", "ridge", "lambda", "scale"])
        for name, X in Xs.items():
            r = fit_all(X, Y, is_train); rows[name] = r
            w.writerow([name] + [f"{r[k]:.4f}" for k in ("trainmean_baseline", "rotation", "rotation+scale", "ridge")]
                       + [f"{r['lambda']:.3g}", f"{r['scale']:.3f}"])
            print(f"{name:20s} base={r['trainmean_baseline']:+.3f} rot={r['rotation']:.3f} "
                  f"rot+s={r['rotation+scale']:.3f} (s={r['scale']:.2f}) ridge={r['ridge']:.3f} "
                  f"| rot+s/ridge={100*r['rotation+scale']/r['ridge']:.0f}%")

    order = [("trainmean_baseline", "mean shift only\n(no map)", "0.72"),
             ("rotation", "rotation", "#7c3aad"),
             ("rotation+scale", "rotation\n+ one scalar", "#7c3aad"),
             ("ridge", "unconstrained\nlinear map", "#2a78d6")]
    titles = {"mbar57": "Predicting write features from L5–7 read features",
              "unique": "Predicting write features from the task-unique direction ($n_A v_1$)"}
    for name, title in titles.items():
        r = rows[name]
        fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=150)
        fig.patch.set_facecolor("white"); ax.set_facecolor("white")
        for x, (k, lab, col) in enumerate(order):
            v = r[k]
            ax.bar([x], [max(v, 0)], width=0.6, color=col, zorder=3)
            ax.text(x, max(v, 0) + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=12.5,
                    fontweight="bold", color="#181c1e")
        ax.set_xticks(range(len(order)), [lab for _, lab, _ in order], fontsize=10.5)
        ax.set_ylim(0, 0.78); ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_ylabel("held-out $R^2$ (14 tasks)", fontsize=11)
        ax.set_title(title, fontsize=12, loc="left", pad=10)
        ax.grid(axis="y", color="#e8eae6", lw=0.8, zorder=0)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / f"rotation_simple_{name}.png", facecolor="white")
        plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
