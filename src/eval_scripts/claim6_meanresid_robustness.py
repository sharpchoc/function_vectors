#!/usr/bin/env python
"""Appendix E companions for the u_A -> v_A linear map (same data/protocol as claim6_meanresid_map.py).

1. Split robustness: 10 random 55/14 task splits (seeds 0-9) + the canonical split; held-out R^2
   (test-mean reference, task-FV targets) of ridge and rotation+scale.
2. Per-task transfer: for every task, the cosine between predicted and true FV averaged over the
   random splits in which it was held out (the canonical split's 14 are in claim6_meanresid_map).
3. Centroid vs within-task: score against PER-PROMPT FV targets v_A^j on the canonical held-out
   tasks — (a) map applied to per-prompt u_A^j, (b) map applied to the task centroid u_A (same
   prediction for all prompts), (c) oracle = each prompt's own task-mean FV (leave-one-prompt-out).
   (b) vs (a) is the within-task share the map captures; (c) is the ceiling for any centroid predictor.

Writes results/69_task_run/understanding_read_write_linear_map/meanresid_map/
  robustness_seeds.csv, per_task_transfer.csv, centroid_decomposition.csv, seed_r2.png
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
from src.eval_scripts.claim6_meanresid_map import (FV, OUT, SPLIT, load, procrustes, r2,  # noqa: E402
                                                   ridge_dual, style, TEAL, PURPLE, INK)

N_SEEDS, N_HELDOUT = 10, 14


def fit_eval(X, Y, is_train, Xpp=None, tasks=None):
    Xtr, Ytr, Xte, Yte = X[is_train], Y[is_train], X[~is_train], Y[~is_train]
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc, Xtec = Xtr - xm, Ytr - ym, Xte - xm
    A, lam = ridge_dual(Xc, Yc)
    R, sc, _ = procrustes(Xc, Yc)
    p_ridge, p_rot = Xtec @ Xc.T @ A + ym, sc * (Xtec @ R) + ym
    ref = Yte.mean(0)
    cos = np.einsum("ij,ij->i", p_ridge, Yte) / (np.linalg.norm(p_ridge, axis=1) * np.linalg.norm(Yte, axis=1))
    return {"ridge": r2(p_ridge, Yte, ref), "rotscale": r2(p_rot, Yte, ref), "lam": lam, "scale": sc,
            "cos": cos, "A": A, "xm": xm, "ym": ym, "Xc": Xc}


def main():
    tasks, is_train_can, X, Xpp, Y = load()
    n = len(tasks)
    rows, per_task = [], {t: [] for t in tasks}
    can = fit_eval(X, Y, is_train_can)
    rows.append({"split": "canonical", "ridge_r2": round(can["ridge"], 4), "rotscale_r2": round(can["rotscale"], 4),
                 "lambda": can["lam"], "scale": round(can["scale"], 3),
                 "heldout_tasks": " ".join(t for t, tr in zip(tasks, is_train_can) if not tr)})
    rng_rows = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        te = set(rng.choice(n, N_HELDOUT, replace=False).tolist())
        is_tr = np.array([i not in te for i in range(n)])
        r = fit_eval(X, Y, is_tr)
        rng_rows.append(r)
        for t, c in zip([tasks[i] for i in range(n) if not is_tr[i]], r["cos"]):
            per_task[t].append(float(c))
        rows.append({"split": f"seed{seed}", "ridge_r2": round(r["ridge"], 4), "rotscale_r2": round(r["rotscale"], 4),
                     "lambda": r["lam"], "scale": round(r["scale"], 3),
                     "heldout_tasks": " ".join(tasks[i] for i in range(n) if not is_tr[i])})
    rr = np.array([r["ridge"] for r in rng_rows]); rs = np.array([r["rotscale"] for r in rng_rows])
    with open(OUT / "robustness_seeds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"canonical: ridge {can['ridge']:.3f} rot+scale {can['rotscale']:.3f} | 10 seeds: ridge {rr.mean():.3f} ± {rr.std():.3f} "
          f"[{rr.min():.3f}, {rr.max():.3f}]  rot+scale {rs.mean():.3f} ± {rs.std():.3f}")

    # per-task transfer over random splits
    pt = [{"task": t, "n_heldout": len(v), "mean_cos_pred_true": round(float(np.mean(v)), 4) if v else ""}
          for t, v in per_task.items()]
    pt.sort(key=lambda d: (d["mean_cos_pred_true"] == "", d["mean_cos_pred_true"]))
    with open(OUT / "per_task_transfer.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pt[0].keys())); w.writeheader(); w.writerows(pt)
    covered = [d for d in pt if d["mean_cos_pred_true"] != ""]
    print(f"per-task (over splits held out; {len(covered)}/69 tasks covered): worst 8:")
    for d in covered[:8]:
        print(f"   {d['task']:26s} cos {d['mean_cos_pred_true']:.3f} (n={d['n_heldout']})")
    print(f"   median {np.median([d['mean_cos_pred_true'] for d in covered]):.3f}")

    # centroid vs within-task, canonical held-out tasks, per-prompt FV targets
    te_tasks = [t for t, tr in zip(tasks, is_train_can) if not tr]
    Ypp = {t: torch.load(FV / f"{t}.pt", map_location="cpu", weights_only=False)["fv"].double().numpy() for t in te_tasks}
    Yj = np.concatenate([Ypp[t] for t in te_tasks]); ref = Yj.mean(0)
    A, xm, ym, Xc = can["A"], can["xm"], can["ym"], can["Xc"]
    pred_pp = np.concatenate([(Xpp[t] - xm) @ Xc.T @ A + ym for t in te_tasks])
    pred_cen = np.concatenate([np.repeat(((X[tasks.index(t)] - xm) @ Xc.T @ A + ym)[None], len(Ypp[t]), 0) for t in te_tasks])
    oracle = np.concatenate([(Ypp[t].sum(0)[None] - Ypp[t]) / (len(Ypp[t]) - 1) for t in te_tasks])   # leave-one-prompt-out task mean
    dec = {"map_on_perprompt_u": r2(pred_pp, Yj, ref), "map_on_task_centroid_u": r2(pred_cen, Yj, ref),
           "oracle_task_mean_fv_loo": r2(oracle, Yj, ref), "trainmean_fv": r2(np.repeat(ym[None], len(Yj), 0), Yj, ref)}
    with open(OUT / "centroid_decomposition.csv", "w") as fh:
        fh.write("predictor,heldout_r2_vs_perprompt_fv\n")
        for k, v in dec.items():
            fh.write(f"{k},{v:.4f}\n")
    print("per-prompt-FV targets (14 held-out tasks):", {k: round(v, 3) for k, v in dec.items()})

    # simple figure: seeds vs canonical
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150); fig.patch.set_facecolor("white"); style(ax)
    ax.scatter(np.zeros(N_SEEDS) + np.linspace(-0.12, 0.12, N_SEEDS), rr, color=TEAL, s=36, zorder=3, label="10 random 55/14 splits")
    ax.scatter([0], [can["ridge"]], color="#c2410c", marker="D", s=60, zorder=4, label="canonical split (main text)")
    ax.scatter(np.ones(N_SEEDS) + np.linspace(-0.12, 0.12, N_SEEDS), rs, color=PURPLE, s=36, zorder=3)
    ax.scatter([1], [can["rotscale"]], color="#c2410c", marker="D", s=60, zorder=4)
    ax.set_xticks([0, 1], ["unconstrained linear\n(ridge)", "rotation + one scalar"], fontsize=10.5)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 0.85)
    ax.set_ylabel("held-out $R^2$", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    ax.set_title("Read→write map: held-out $R^2$ across task splits", loc="left", fontsize=12, color=INK, pad=10)
    fig.tight_layout(); fig.savefig(OUT / "seed_r2.png", facecolor="white"); plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
