#!/usr/bin/env python
"""Seed-split robustness of the avg-label-token -> per-prompt-FV ridge.

Same estimator as ridge_labeltoken_to_fv.py's 'avg' variant (full-dim 4096 -> 4096, X = the
mean of the 10 demo-label L6 activations, Y = per-prompt FV, lambda from logspace(-1,8,19)
by 5-fold CV over the split's TRAIN tasks, pooled MSE), but the 55/14 train/test TASK split
is drawn at random per --split_seed instead of the canonical seed-43 split. Purpose: test
whether held-out R^2 depends on the task-family composition of the split (the coverage /
task-level-overfitting hypothesis).

Output: artifacts/69_task_run/labeltoken_fv_ridge/seedsplits/seed<S>.json
  pooled train (in-sample) and held-out R^2 (uniform + weighted), chosen lambda, the test
  task list, and per-held-out-task R^2 under BOTH references (own-task mean, and the
  held-out split-pool mean as in r2_by_task.png).
"""
import argparse
import json
import random
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
OUT = ARTIFACTS_ROOT / "69_task_run" / "labeltoken_fv_ridge" / "seedsplits"
N_TEST, KFOLDS, CV_SEED = 14, 5, 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split_seed", type=int, required=True)
    return p.parse_args()


def r2_scores(y, pred):
    resid = ((y - pred) ** 2).sum(dim=0)
    tot = ((y - y.mean(dim=0)) ** 2).sum(dim=0)
    ok = tot > 0
    return float((1 - resid[ok] / tot[ok]).mean()), float(1 - resid.sum() / tot.sum())


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    all_tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    test_tasks = sorted(random.Random(args.split_seed).sample(all_tasks, N_TEST))
    train_tasks = [t for t in all_tasks if t not in test_tasks]
    print(f"seed {args.split_seed}: test = {test_tasks}", flush=True)

    def load(t):
        a = torch.load(ACTS / f"{t}.pt", map_location="cpu", weights_only=False)
        f = torch.load(FVS / f"{t}.pt", map_location="cpu", weights_only=False)
        assert list(a["prompt_index"]) == list(f["prompt_index"])
        return a["acts"].double().mean(dim=1), f["fv"].double()

    Xtr, Ytr, sl = [], [], {}
    pos = 0
    for t in train_tasks:
        x, y = load(t)
        Xtr.append(x); Ytr.append(y)
        sl[t] = (pos, pos + len(y)); pos += len(y)
    Xtr = torch.cat(Xtr).to(device); Ytr = torch.cat(Ytr).to(device)

    alphas = list(np.logspace(-1, 8, 19))
    order = np.random.RandomState(CV_SEED).permutation(len(train_tasks))
    folds = [sorted(train_tasks[i] for i in f) for f in np.array_split(order, KFOLDS)]
    cv = torch.zeros(len(alphas), dtype=torch.float64, device=device)
    for fold in folds:
        m = torch.zeros(len(Xtr), dtype=torch.bool)
        for t in fold:
            s, e = sl[t]
            m[s:e] = True
        m = m.to(device)
        xbar, ybar, ev, evec, c = ridge_eig_prep(Xtr[~m], Ytr[~m])
        a_val = (Xtr[m] - xbar) @ evec
        for ai, al in enumerate(alphas):
            cv[ai] += (((a_val / (ev + al)) @ c + ybar - Ytr[m]) ** 2).sum()
    bi = int(torch.argmin(cv))
    best_alpha = float(alphas[bi])

    xbar, ybar, ev, evec, c = ridge_eig_prep(Xtr, Ytr)
    r2u_tr, r2w_tr = r2_scores(Ytr, ridge_predict(Xtr, xbar, ybar, ev, evec, c, best_alpha))

    Xte, Yte, tsl = [], [], {}
    pos = 0
    for t in test_tasks:
        x, y = load(t)
        Xte.append(x); Yte.append(y)
        tsl[t] = (pos, pos + len(y)); pos += len(y)
    Xte = torch.cat(Xte).to(device); Yte = torch.cat(Yte).to(device)
    Pte = ridge_predict(Xte, xbar, ybar, ev, evec, c, best_alpha)
    r2u_te, r2w_te = r2_scores(Yte, Pte)

    pool_mean = Yte.mean(dim=0)
    per_task = {}
    for t in test_tasks:
        s, e = tsl[t]
        own_u, _ = r2_scores(Yte[s:e], Pte[s:e])
        tot = ((Yte[s:e] - pool_mean) ** 2).sum(dim=0)
        ok = tot > 0
        poolref = float((1 - ((Yte[s:e] - Pte[s:e]) ** 2).sum(dim=0)[ok] / tot[ok]).mean())
        per_task[t] = {"r2_ownmean": round(own_u, 4), "r2_poolref": round(poolref, 4)}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"seed{args.split_seed}.json", "w") as f:
        json.dump({"split_seed": args.split_seed, "test_tasks": test_tasks,
                   "best_alpha": best_alpha,
                   "alpha_pinned": bi in (0, len(alphas) - 1),
                   "r2_train_uniform": round(r2u_tr, 4),
                   "r2_train_weighted": round(r2w_tr, 4),
                   "r2_test_uniform": round(r2u_te, 4),
                   "r2_test_weighted": round(r2w_te, 4),
                   "per_test_task": per_task}, f, indent=1)
    print(f"seed {args.split_seed}: alpha={best_alpha:g} | R2 train {r2u_tr:.4f} "
          f"test {r2u_te:.4f} (weighted {r2w_tr:.4f}/{r2w_te:.4f})", flush=True)


if __name__ == "__main__":
    main()
