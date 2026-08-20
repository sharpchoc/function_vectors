#!/usr/bin/env python3.12
"""Task-level ridge: task-mean L6 label activation (read feature) -> task FV.

55 train tasks as samples (4096 features -> 4096 targets), lambda by
leave-one-task-out CV on train, scored on the 14 held-out tasks.
Dual (kernel) form since n=55 << d=4096. Features and targets are centered
on TRAIN statistics (ridge with intercept).
"""
import json
import numpy as np
import torch

ART = "/workspace/function_vectors/artifacts/69_task_run"
SPLIT = "/workspace/function_vectors/task_splits/extended_steerable_69_prunedfail.json"
LAYER = 6

split = json.load(open(SPLIT))
train_tasks, test_tasks = split["train_tasks"], split["heldout_tasks"]

def load_task(t):
    r = torch.load(f"{ART}/label_resid_means/{t}.pt", map_location="cpu", weights_only=False)
    x = r["resid_means"][LAYER].float().numpy()
    f = torch.load(f"{ART}/perprompt_fvs/{t}.pt", map_location="cpu", weights_only=False)
    y = f["fv"].float().mean(0).numpy()
    return x, y

Xtr, Ytr = map(np.stack, zip(*[load_task(t) for t in train_tasks]))
Xte, Yte = map(np.stack, zip(*[load_task(t) for t in test_tasks]))
print(f"train X {Xtr.shape} Y {Ytr.shape} | test X {Xte.shape} Y {Yte.shape}")

xm, ym = Xtr.mean(0), Ytr.mean(0)
Xc, Yc = Xtr - xm, Ytr - ym

def fit_predict(Xc_f, Yc_f, Xq_c, lam):
    # dual ridge: alpha = (K + lam I)^-1 Yc ; pred = Kq alpha
    K = Xc_f @ Xc_f.T
    A = np.linalg.solve(K + lam * np.eye(len(K)), Yc_f)
    return (Xq_c @ Xc_f.T) @ A

lambdas = np.logspace(-2, 10, 49)
loo = []
for lam in lambdas:
    sse = 0.0
    for i in range(len(Xc)):
        m = np.ones(len(Xc), bool); m[i] = False
        xm_i, ym_i = Xtr[m].mean(0), Ytr[m].mean(0)
        pred = fit_predict(Xtr[m] - xm_i, Ytr[m] - ym_i, Xtr[i] - xm_i, lam) + ym_i
        sse += ((pred - Ytr[i]) ** 2).sum()
    loo.append(sse)
loo = np.array(loo)
best = lambdas[loo.argmin()]
# LOO R^2 vs LOO centroid baseline (predicting the other-54 mean)
base_sse = sum((((Ytr[np.arange(len(Ytr)) != i].mean(0)) - Ytr[i]) ** 2).sum() for i in range(len(Ytr)))
print(f"best lambda (LOO on train): {best:.3g}   LOO R^2 vs LOO-mean baseline: {1 - loo.min()/base_sse:.4f}")
for lam, s in zip(lambdas, loo):
    print(f"  lam={lam:9.3g}  looR2={1 - s/base_sse:+.4f}")

pred = fit_predict(Xc, Yc, Xte - xm, best) + ym

def r2(y, p, ref):
    return 1 - ((y - p) ** 2).sum() / ((y - ref) ** 2).sum()

r2_trainmean = r2(Yte, pred, ym)                 # denominator: around train-mean FV (repo convention: split-average predictor)
r2_testmean = r2(Yte, pred, Yte.mean(0))         # denominator: around test-set mean (standard R^2)
cos = [(p @ y) / (np.linalg.norm(p) * np.linalg.norm(y)) for p, y in zip(pred, Yte)]
cos_base = [(ym @ y) / (np.linalg.norm(ym) * np.linalg.norm(y)) for y in Yte]

print(f"\nHELD-OUT (14 tasks), lambda={best:.3g}")
print(f"  R^2 (baseline = train-mean FV):  {r2_trainmean:.4f}")
print(f"  R^2 (baseline = test-set mean):  {r2_testmean:.4f}")
print(f"  mean cos(pred, true FV):         {np.mean(cos):.4f}   (train-mean baseline: {np.mean(cos_base):.4f})")
print("\nper-task cos / R^2-vs-trainmean:")
for t, p, y in sorted(zip(test_tasks, pred, Yte), key=lambda z: -(z[1] @ z[2]) / (np.linalg.norm(z[1]) * np.linalg.norm(z[2]))):
    c = (p @ y) / (np.linalg.norm(p) * np.linalg.norm(y))
    rr = 1 - ((y - p) ** 2).sum() / ((y - ym) ** 2).sum()
    print(f"  {t:28s} cos={c:+.3f}  r2={rr:+.3f}")

# also: sweep a few lambdas on test to show sensitivity (context only, not selection)
print("\nheld-out R^2 (train-mean baseline) across lambdas [context only]:")
for lam in lambdas[::4]:
    p = fit_predict(Xc, Yc, Xte - xm, lam) + ym
    print(f"  lam={lam:9.3g}  R2={r2(Yte, p, ym):+.4f}")
