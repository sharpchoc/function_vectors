#!/usr/bin/env python
"""Lasso in the train-PCA basis: avg label-token L6 activation -> per-prompt FV.

User variant (2026-08-18): rotate X into the PCA basis fit on the train set (ALL 4096
components — a pure rotation, not rank reduction), then fit a multi-output linear map with
L1 penalty. Ridge is rotation-invariant, so this is only meaningful with L1: sparsity in
the PCA basis lets the fit keep train-aligned directions and zero the tail, a soft,
data-adaptive alternative to the (unhelpful) hard PCA-90 cut.

Details:
  X = mean of the 10 demo-label L6 activations per prompt (label_all10_L6_acts);
  Y = per-prompt FV (perprompt_fvs), NOT reduced, NOT rotated.
  PCA basis: centered SVD of the 55-train-task X pool; Z = (X - xbar) @ V^T (all 4096).
  Model: Y - ybar ~ Z W, minimize (1/2n)||Y - ybar - Z W||_F^2 + lam * ||W||_1,
  solved by FISTA (fp32, fixed step 1/L, warm-started down the lam path).
  lam grid: lam_max * {0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003} with
  lam_max = max|Z^T (Y-ybar)|/n; chosen by 5-fold CV over TRAIN TASKS (pooled MSE).
  R^2 reported as before (uniform + weighted; test = 14 held-out tasks), plus sparsity
  stats (fraction of nonzero weights; number of input PCs with any nonzero weight).

Output: artifacts/69_task_run/labeltoken_fv_ridge/lasso_pca_avg.json
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

ACTS = ARTIFACTS_ROOT / "69_task_run" / "label_all10_L6_acts"
FVS = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
OUT = ARTIFACTS_ROOT / "69_task_run" / "labeltoken_fv_ridge"
KFOLDS, CV_SEED = 5, 42
LAM_FRACS = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
N_ITER = 400

torch.backends.cuda.matmul.allow_tf32 = True


def fista(Z, Yc, lam, L, W0=None, n_iter=N_ITER):
    """min (1/2n)||Yc - Z W||^2 + lam ||W||_1 ; returns W (D_in, D_out) fp32."""
    n = Z.shape[0]
    W = torch.zeros(Z.shape[1], Yc.shape[1], device=Z.device) if W0 is None else W0.clone()
    Wy = W.clone()
    t = 1.0
    step = 1.0 / L
    thr = lam * step
    for it in range(n_iter):
        grad = Z.T @ (Z @ Wy - Yc) / n
        W_new = Wy - step * grad
        W_new = torch.sign(W_new) * torch.clamp(W_new.abs() - thr, min=0.0)
        t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
        Wy = W_new + ((t - 1) / t_new) * (W_new - W)
        W, t = W_new, t_new
    return W


def r2_scores(y, pred):
    resid = ((y - pred) ** 2).sum(dim=0)
    tot = ((y - y.mean(dim=0)) ** 2).sum(dim=0)
    ok = tot > 0
    return float((1 - resid[ok] / tot[ok]).mean()), float(1 - resid.sum() / tot.sum())


def main():
    device = "cuda"
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    train_tasks, test_tasks = sorted(split["train_tasks"]), sorted(split["heldout_tasks"])

    def load(t):
        a = torch.load(ACTS / f"{t}.pt", map_location="cpu", weights_only=False)
        f = torch.load(FVS / f"{t}.pt", map_location="cpu", weights_only=False)
        assert list(a["prompt_index"]) == list(f["prompt_index"])
        return a["acts"].float().mean(dim=1), f["fv"].float()

    Xtr, Ytr, sl = [], [], {}
    pos = 0
    for t in train_tasks:
        x, y = load(t)
        Xtr.append(x); Ytr.append(y)
        sl[t] = (pos, pos + len(y)); pos += len(y)
    Xtr = torch.cat(Xtr).to(device); Ytr = torch.cat(Ytr).to(device)
    Xte, Yte = [], []
    for t in test_tasks:
        x, y = load(t)
        Xte.append(x); Yte.append(y)
    Xte = torch.cat(Xte).to(device); Yte = torch.cat(Yte).to(device)

    # centered PCA rotation from the train pool (all components — pure rotation)
    xbar = Xtr.mean(dim=0)
    U, S, Vh = torch.linalg.svd(Xtr - xbar, full_matrices=False)
    V = Vh                                   # (4096, 4096) rotation
    Ztr = (Xtr - xbar) @ V.T
    Zte = (Xte - xbar) @ V.T
    ybar = Ytr.mean(dim=0)
    L_lip = float(S[0] ** 2) / len(Ztr)
    lam_max = float((Ztr.T @ (Ytr - ybar)).abs().max()) / len(Ztr)
    lams = [lam_max * f for f in LAM_FRACS]
    print(f"n={len(Ztr)} lam_max={lam_max:.4f} L={L_lip:.4f}", flush=True)

    rng = np.random.RandomState(CV_SEED)
    order = rng.permutation(len(train_tasks))
    folds = [sorted(train_tasks[i] for i in f) for f in np.array_split(order, KFOLDS)]
    cv = np.zeros(len(lams))
    for fi, fold in enumerate(folds):
        m = torch.zeros(len(Ztr), dtype=torch.bool)
        for t in fold:
            s, e = sl[t]
            m[s:e] = True
        m = m.to(device)
        Zf, Yf = Ztr[~m], Ytr[~m] - ybar
        Zv, Yv = Ztr[m], Ytr[m] - ybar
        Lf = float(torch.linalg.matrix_norm(Zf, 2) ** 2) / len(Zf)
        W = None
        for li, lam in enumerate(lams):
            W = fista(Zf, Yf, lam, Lf, W0=W)
            cv[li] += float(((Zv @ W - Yv) ** 2).sum())
        print(f"fold {fi} done", flush=True)
    bi = int(np.argmin(cv))
    best_lam = lams[bi]
    print(f"chosen lam = lam_max * {LAM_FRACS[bi]} (idx {bi}"
          f"{', PINNED' if bi in (0, len(lams) - 1) else ''})", flush=True)

    W = None
    for lam in lams[:bi + 1]:                # warm-start path down to the chosen lam
        W = fista(Ztr, Ytr - ybar, lam, L_lip, W0=W)
    pred_tr = Ztr @ W + ybar
    pred_te = Zte @ W + ybar
    r2u_tr, r2w_tr = r2_scores(Ytr.double(), pred_tr.double())
    r2u_te, r2w_te = r2_scores(Yte.double(), pred_te.double())
    nz = W != 0
    frac_nz = float(nz.float().mean())
    pcs_used = int((nz.any(dim=1)).sum())
    # where do the surviving weights sit along the PC spectrum?
    row_nz = nz.sum(dim=1).cpu().numpy()
    print(f"R2 train {r2u_tr:.4f} test {r2u_te:.4f} (weighted {r2w_tr:.4f}/{r2w_te:.4f}) | "
          f"nonzero {frac_nz:.4%} of weights | input PCs used {pcs_used}/4096", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "lasso_pca_avg.json", "w") as f:
        json.dump({"variant": "avg", "basis": "centered train-PCA rotation (all 4096)",
                   "lam_max": lam_max, "lam_fracs": LAM_FRACS, "chosen_frac": LAM_FRACS[bi],
                   "lam_pinned": bi in (0, len(lams) - 1),
                   "cv_sqerr": list(cv), "n_iter": N_ITER,
                   "r2_train_uniform": round(r2u_tr, 4), "r2_test_uniform": round(r2u_te, 4),
                   "r2_train_weighted": round(r2w_tr, 4), "r2_test_weighted": round(r2w_te, 4),
                   "frac_nonzero_weights": frac_nz, "input_pcs_used": pcs_used,
                   "nonzero_per_pc_first50": [int(x) for x in row_nz[:50]],
                   "ref": {"fulldim_ridge_avg": {"train": 0.7871, "test": 0.4641}}}, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
