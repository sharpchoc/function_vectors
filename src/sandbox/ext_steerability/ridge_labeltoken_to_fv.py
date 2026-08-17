#!/usr/bin/env python
"""Ridge regression: nth-demo-label L6 activation -> per-prompt FV (69-task pool).

X (one variant per invocation, --variant in {1..10, avg}):
  n    the block-6 output at the LAST token of demo n's label, per prompt
       (capture_all10_label_L6.py; (150, 10, 4096) per task)
  avg  the mean of the 10 label activations, per prompt
Y: the per-prompt FV (artifacts/69_task_run/perprompt_fvs/<task>.pt, 'fv' (150, 4096)).

Fit ONE full-dim (4096 -> 4096) ridge on the 55 TRAIN tasks' 8250 pairs; lambda from
np.logspace(-1, 8, 19) by 5-fold CV over TASKS (seed-42 fold split, pooled held-out-fold
MSE); refit on all train pairs; report R^2 on the train pairs (in-sample) and on the 14
HELD-OUT tasks' pairs. R^2 conventions: headline = uniform average of per-dim R^2 over the
4096 dims (eval-set mean as the reference); variance-weighted also reported. Per-task R^2
saved for both splits. Solver: centered Gram eigendecomposition reused from
src/eval_scripts/regress_activation_to_fv_fulldim_ridge.py, fp64 on GPU.

Output: artifacts/69_task_run/labeltoken_fv_ridge/variant_<v>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# local bootstrap for in-repo runs; a PYTHONPATH-supplied repo also works (staged copies)
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
try:
    from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (
        ridge_eig_prep, ridge_predict)
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from regress_activation_to_fv_fulldim_ridge import ridge_eig_prep, ridge_predict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True,
                   choices=[str(n) for n in range(1, 11)] + ["avg"])
    p.add_argument("--acts_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_all10_L6_acts")
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "labeltoken_fv_ridge")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def r2_scores(y_true, y_pred):
    """(uniform-average per-dim R^2, variance-weighted R^2), eval-set mean reference."""
    resid = ((y_true - y_pred) ** 2).sum(dim=0)               # (D,)
    tot = ((y_true - y_true.mean(dim=0)) ** 2).sum(dim=0)     # (D,)
    ok = tot > 0
    r2_dim = 1 - resid[ok] / tot[ok]
    uniform = float(r2_dim.mean())
    weighted = float(1 - resid.sum() / tot.sum())
    return uniform, weighted


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.load(open(args.split_path))
    train_tasks = sorted(split["train_tasks"])
    test_tasks = sorted(split["heldout_tasks"])

    def load_xy(task):
        a = torch.load(args.acts_root / f"{task}.pt", map_location="cpu", weights_only=False)
        f = torch.load(args.fv_root / f"{task}.pt", map_location="cpu", weights_only=False)
        assert list(a["prompt_index"]) == list(f["prompt_index"]), f"{task}: prompt order mismatch"
        acts = a["acts"].to(torch.float64)                    # (150, 10, 4096)
        x = acts.mean(dim=1) if args.variant == "avg" else acts[:, int(args.variant) - 1]
        y = f["fv"].to(torch.float64)
        return x, y

    X_tr, Y_tr, tr_slices = [], [], {}
    pos = 0
    for t in train_tasks:
        x, y = load_xy(t)
        X_tr.append(x); Y_tr.append(y)
        tr_slices[t] = (pos, pos + len(x)); pos += len(x)
    X_tr = torch.cat(X_tr).to(device); Y_tr = torch.cat(Y_tr).to(device)
    print(f"variant {args.variant}: train X {tuple(X_tr.shape)}", flush=True)

    alphas = list(np.logspace(-1, 8, 19))
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(train_tasks))
    folds = [sorted(train_tasks[i] for i in fold)
             for fold in np.array_split(order, args.kfolds)]

    cv_sqerr = torch.zeros(len(alphas), dtype=torch.float64, device=device)
    for fi, fold in enumerate(folds):
        val_mask = torch.zeros(len(X_tr), dtype=torch.bool)
        for t in fold:
            s, e = tr_slices[t]
            val_mask[s:e] = True
        xf, yf = X_tr[~val_mask.to(device)], Y_tr[~val_mask.to(device)]
        xv, yv = X_tr[val_mask.to(device)], Y_tr[val_mask.to(device)]
        xbar, ybar, evals, evecs, c = ridge_eig_prep(xf, yf)
        a_val = (xv - xbar) @ evecs
        for ai, alpha in enumerate(alphas):
            pred = (a_val / (evals + alpha)) @ c + ybar
            cv_sqerr[ai] += ((pred - yv) ** 2).sum()
        print(f"  fold {fi} done", flush=True)
    best_idx = int(torch.argmin(cv_sqerr))
    best_alpha = float(alphas[best_idx])
    pinned = best_idx in (0, len(alphas) - 1)

    xbar, ybar, evals, evecs, c = ridge_eig_prep(X_tr, Y_tr)
    pred_tr = ridge_predict(X_tr, xbar, ybar, evals, evecs, c, best_alpha)
    r2u_tr, r2w_tr = r2_scores(Y_tr, pred_tr)

    per_task = {}
    for t in train_tasks:
        s, e = tr_slices[t]
        u, w = r2_scores(Y_tr[s:e], pred_tr[s:e])
        per_task[t] = {"split": "train", "r2_uniform": round(u, 4), "r2_weighted": round(w, 4)}

    X_te, Y_te, te_slices = [], [], {}
    pos = 0
    for t in test_tasks:
        x, y = load_xy(t)
        X_te.append(x); Y_te.append(y)
        te_slices[t] = (pos, pos + len(x)); pos += len(x)
    X_te = torch.cat(X_te).to(device); Y_te = torch.cat(Y_te).to(device)
    pred_te = ridge_predict(X_te, xbar, ybar, evals, evecs, c, best_alpha)
    r2u_te, r2w_te = r2_scores(Y_te, pred_te)
    for t in test_tasks:
        s, e = te_slices[t]
        u, w = r2_scores(Y_te[s:e], pred_te[s:e])
        per_task[t] = {"split": "heldout", "r2_uniform": round(u, 4), "r2_weighted": round(w, 4)}

    args.out_root.mkdir(parents=True, exist_ok=True)
    out = {"variant": args.variant, "best_alpha": best_alpha, "alpha_pinned": pinned,
           "cv_sqerr_by_alpha": {f"{a:g}": float(v) for a, v in zip(alphas, cv_sqerr.cpu())},
           "r2_train_uniform": round(r2u_tr, 4), "r2_train_weighted": round(r2w_tr, 4),
           "r2_test_uniform": round(r2u_te, 4), "r2_test_weighted": round(r2w_te, 4),
           "n_train_pairs": int(len(X_tr)), "n_test_pairs": int(len(X_te)),
           "per_task": per_task,
           "config": {"layer": 6, "x_site": "last token of demo label(s), block-6 output",
                      "cv": f"{args.kfolds}-fold over train tasks, pooled MSE",
                      "r2_reference": "eval-set mean per dim"}}
    with open(args.out_root / f"variant_{args.variant}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"variant {args.variant}: alpha={best_alpha:g}{' (PINNED)' if pinned else ''} | "
          f"R2 train {r2u_tr:.4f} (w {r2w_tr:.4f}) | test {r2u_te:.4f} (w {r2w_te:.4f})",
          flush=True)


if __name__ == "__main__":
    main()
