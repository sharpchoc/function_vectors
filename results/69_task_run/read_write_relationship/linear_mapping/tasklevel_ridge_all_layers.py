#!/usr/bin/env python3.12
"""Task-level ridge per layer: task-mean label-token activation at layer L -> task FV.

For each of GPT-J's 28 layers: 55 train tasks as samples, lambda by
leave-one-task-out CV on train, held-out R^2 on the 14 test tasks.
Dual (kernel) ridge, train-centered features/targets (intercept).
"""
import csv
import json
import numpy as np
import torch

ART = "/workspace/function_vectors/artifacts/69_task_run"
SPLIT = "/workspace/function_vectors/task_splits/extended_steerable_69_prunedfail.json"
TMP = "/root/.claude/jobs/6a46ec85/tmp"

split = json.load(open(SPLIT))
train_tasks, test_tasks = split["train_tasks"], split["heldout_tasks"]

def load(t):
    r = torch.load(f"{ART}/label_resid_means/{t}.pt", map_location="cpu", weights_only=False)
    f = torch.load(f"{ART}/perprompt_fvs/{t}.pt", map_location="cpu", weights_only=False)
    return r["resid_means"].float().numpy(), f["fv"].float().mean(0).numpy()  # (28,4096), (4096,)

Xall_tr, Ytr = map(np.stack, zip(*[load(t) for t in train_tasks]))   # (55,28,4096), (55,4096)
Xall_te, Yte = map(np.stack, zip(*[load(t) for t in test_tasks]))    # (14,28,4096), (14,4096)

def fit_predict(Xc_f, Yc_f, Xq_c, lam):
    K = Xc_f @ Xc_f.T
    A = np.linalg.solve(K + lam * np.eye(len(K)), Yc_f)
    return (Xq_c @ Xc_f.T) @ A

lambdas = np.logspace(-2, 6, 33)
n = len(train_tasks)
ym = Ytr.mean(0)

rows = []
for L in range(28):
    Xtr, Xte = Xall_tr[:, L], Xall_te[:, L]
    xm = Xtr.mean(0)
    # LOO over lambda grid
    loo_sse = np.zeros(len(lambdas))
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        xm_i, ym_i = Xtr[m].mean(0), Ytr[m].mean(0)
        Xc_i, Yc_i = Xtr[m] - xm_i, Ytr[m] - ym_i
        K = Xc_i @ Xc_i.T
        kq = (Xtr[i] - xm_i) @ Xc_i.T
        for j, lam in enumerate(lambdas):
            A = np.linalg.solve(K + lam * np.eye(n - 1), Yc_i)
            loo_sse[j] += ((kq @ A + ym_i - Ytr[i]) ** 2).sum()
    base_sse = sum((((Ytr[np.arange(n) != i].mean(0)) - Ytr[i]) ** 2).sum() for i in range(n))
    jbest = int(loo_sse.argmin())
    lam = lambdas[jbest]
    loo_r2 = 1 - loo_sse[jbest] / base_sse

    pred = fit_predict(Xtr - xm, Ytr - ym, Xte - xm, lam) + ym
    r2_trainmean = 1 - ((Yte - pred) ** 2).sum() / ((Yte - ym) ** 2).sum()
    r2_testmean = 1 - ((Yte - pred) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()
    cos = float(np.mean([(p @ y) / (np.linalg.norm(p) * np.linalg.norm(y)) for p, y in zip(pred, Yte)]))
    rows.append(dict(layer=L, best_lambda=lam, loo_train_r2=round(loo_r2, 4),
                     heldout_r2_trainmean=round(r2_trainmean, 4),
                     heldout_r2_testmean=round(r2_testmean, 4), heldout_mean_cos=round(cos, 4)))
    print(f"L{L:2d}  lam={lam:9.3g}  looR2={loo_r2:+.4f}  heldout R2(trainmean)={r2_trainmean:+.4f}  "
          f"R2(testmean)={r2_testmean:+.4f}  cos={cos:.4f}")

with open(f"{TMP}/tasklevel_ridge_all_layers.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"\nsaved {TMP}/tasklevel_ridge_all_layers.csv")
