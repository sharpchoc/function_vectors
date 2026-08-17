#!/usr/bin/env python
"""Rank-reduced ridge: label-token L6 activations -> per-prompt FV, X projected onto the
top-90%-energy UNCENTERED PCs of the train-set n=10 activations.

Basis (fixed once, user spec 2026-08-18): all 8250 train-set n=10 rows (55 tasks x 150
prompts, block-6 output at the last token of demo 10's label), UNCENTERED SVD, k = smallest
rank with cum sigma^2 >= 0.90. The SAME k right-singular vectors reduce EVERY X variant
(n = 1..10 and the avg-of-10), train and held-out alike; then the same ridge protocol as
ridge_labeltoken_to_fv.py (lambda from logspace(-1,8,19), 5-fold CV over train tasks,
pooled MSE; R^2 uniform-average + variance-weighted; test = 14 held-out tasks).

Output: artifacts/69_task_run/labeltoken_fv_ridge/pca90_n10.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
try:
    from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (
        ridge_eig_prep, ridge_predict)
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from regress_activation_to_fv_fulldim_ridge import ridge_eig_prep, ridge_predict

ACTS = ARTIFACTS_ROOT / "69_task_run" / "label_all10_L6_acts"
FVS = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
OUT = ARTIFACTS_ROOT / "69_task_run" / "labeltoken_fv_ridge"
KFOLDS, SEED = 5, 42


def r2_scores(y, pred):
    resid = ((y - pred) ** 2).sum(dim=0)
    tot = ((y - y.mean(dim=0)) ** 2).sum(dim=0)
    ok = tot > 0
    return float((1 - resid[ok] / tot[ok]).mean()), float(1 - resid.sum() / tot.sum())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    train_tasks, test_tasks = sorted(split["train_tasks"]), sorted(split["heldout_tasks"])

    def load(task):
        a = torch.load(ACTS / f"{task}.pt", map_location="cpu", weights_only=False)
        f = torch.load(FVS / f"{task}.pt", map_location="cpu", weights_only=False)
        assert list(a["prompt_index"]) == list(f["prompt_index"])
        return a["acts"].to(torch.float64), f["fv"].to(torch.float64)

    A, Y, slices = {}, {}, {}
    for grp, tl in (("train", train_tasks), ("test", test_tasks)):
        xs, ys = [], []
        pos = 0
        for t in tl:
            acts, y = load(t)
            xs.append(acts); ys.append(y)
            slices[t] = (pos, pos + len(y)); pos += len(y)
        A[grp] = torch.cat(xs).to(device)      # (N, 10, 4096)
        Y[grp] = torch.cat(ys).to(device)

    # --- uncentered PCA basis from the n=10 TRAIN rows (fixed for all variants) ---
    _, S, Vh = torch.linalg.svd(A["train"][:, 9], full_matrices=False)
    energy = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
    k = int(torch.searchsorted(energy, 0.90).item()) + 1
    V = Vh[:k]                                            # (k, 4096)
    print(f"uncentered PCA basis: k={k} at 90% energy (of {len(S)}); "
          f"top-1 energy {float(S[0]**2/(S**2).sum()):.3f}", flush=True)

    alphas = list(np.logspace(-1, 8, 19))
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(train_tasks))
    folds = [sorted(train_tasks[i] for i in fold) for fold in np.array_split(order, KFOLDS)]

    variants = [(str(n), lambda g, n=n: A[g][:, n - 1]) for n in range(1, 11)]
    variants.append(("avg", lambda g: A[g].mean(dim=1)))

    results = {}
    for xname, getx in variants:
        Ztr = getx("train") @ V.T                         # (8250, k) — X reduced; Y is NOT
        Zte = getx("test") @ V.T
        cv = torch.zeros(len(alphas), dtype=torch.float64, device=device)
        for fold in folds:
            m = torch.zeros(len(Ztr), dtype=torch.bool)
            for t in fold:
                s, e = slices[t]
                m[s:e] = True
            m = m.to(device)
            xbar, ybar, ev, evec, c = ridge_eig_prep(Ztr[~m], Y["train"][~m])
            a_val = (Ztr[m] - xbar) @ evec
            for ai, al in enumerate(alphas):
                pred = (a_val / (ev + al)) @ c + ybar
                cv[ai] += ((pred - Y["train"][m]) ** 2).sum()
        bi = int(torch.argmin(cv))
        best_alpha = float(alphas[bi])
        xbar, ybar, ev, evec, c = ridge_eig_prep(Ztr, Y["train"])
        r2u_tr, r2w_tr = r2_scores(Y["train"], ridge_predict(Ztr, xbar, ybar, ev, evec, c, best_alpha))
        r2u_te, r2w_te = r2_scores(Y["test"], ridge_predict(Zte, xbar, ybar, ev, evec, c, best_alpha))
        results[xname] = {"k": k, "best_alpha": best_alpha,
                          "alpha_pinned": bi in (0, len(alphas) - 1),
                          "r2_train_uniform": round(r2u_tr, 4),
                          "r2_test_uniform": round(r2u_te, 4),
                          "r2_train_weighted": round(r2w_tr, 4),
                          "r2_test_weighted": round(r2w_te, 4)}
        print(f"{xname}: k={k} alpha={best_alpha:g}"
              f"{' PIN' if results[xname]['alpha_pinned'] else ''} | "
              f"R2 train {r2u_tr:.4f} test {r2u_te:.4f} "
              f"(weighted {r2w_tr:.4f}/{r2w_te:.4f})", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pca90_n10.json", "w") as f:
        json.dump({"basis": "uncentered PCA of n=10 train rows, cum sigma^2 >= 0.90",
                   "k": k, "results": results}, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
