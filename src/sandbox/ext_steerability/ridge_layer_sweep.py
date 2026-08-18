#!/usr/bin/env python
"""Layer sweep of the label-token -> per-prompt-FV ridge (avg-of-10 X, one layer/invocation).

Same protocol as ridge_labeltoken_to_fv.py's avg variant, with X = the mean label-token
activation at block --layer (capture_avg10_label_multilayer.py). Reported per layer:
  r2_train_insample      in-sample fit on all 55 train tasks (optimistic, kept for
                         comparability)
  r2_train_unseenprompts fit on 120 prompts/task, eval on the other 30 (honest prompt-level
                         generalization; fair oracle on those rows also reported)
  r2_test                the 14 held-out tasks (fit on all train pairs)
lambda from logspace(-1,8,19) by 5-fold CV over train tasks (pooled MSE), independently per
layer. R^2 uniform-average over dims (variance-weighted also stored).

Output: artifacts/69_task_run/labeltoken_fv_ridge_layer_sweep/layer_<L>.json
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
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--acts_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_avg10_L5-15_acts")
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "labeltoken_fv_ridge_layer_sweep")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt_split_seed", type=int, default=43)
    return p.parse_args()


def r2_scores(y, pred):
    resid = ((y - pred) ** 2).sum(dim=0)
    tot = ((y - y.mean(dim=0)) ** 2).sum(dim=0)
    ok = tot > 0
    return float((1 - resid[ok] / tot[ok]).mean()), float(1 - resid.sum() / tot.sum())


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.load(open(args.split_path))
    train_tasks, test_tasks = sorted(split["train_tasks"]), sorted(split["heldout_tasks"])

    def load(t):
        a = torch.load(args.acts_root / f"{t}.pt", map_location="cpu", weights_only=False)
        f = torch.load(args.fv_root / f"{t}.pt", map_location="cpu", weights_only=False)
        assert list(a["prompt_index"]) == list(f["prompt_index"])
        li = a["layers"].index(args.layer)
        return a["acts"][:, li].to(torch.float64), f["fv"].to(torch.float64)

    Xtr, Ytr, sl = [], [], {}
    pos = 0
    for t in train_tasks:
        x, y = load(t)
        Xtr.append(x); Ytr.append(y)
        sl[t] = (pos, pos + len(x)); pos += len(x)
    Xtr = torch.cat(Xtr).to(device); Ytr = torch.cat(Ytr).to(device)

    alphas = list(np.logspace(-1, 8, 19))
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(train_tasks))
    folds = [sorted(train_tasks[i] for i in fold)
             for fold in np.array_split(order, args.kfolds)]
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
    pinned = bi in (0, len(alphas) - 1)

    # final fit on all train pairs; in-sample + held-out-task evals
    xbar, ybar, ev, evec, c = ridge_eig_prep(Xtr, Ytr)
    r2u_tr, r2w_tr = r2_scores(Ytr, ridge_predict(Xtr, xbar, ybar, ev, evec, c, best_alpha))
    Xte, Yte = [], []
    for t in test_tasks:
        x, y = load(t)
        Xte.append(x); Yte.append(y)
    Xte = torch.cat(Xte).to(device); Yte = torch.cat(Yte).to(device)
    r2u_te, r2w_te = r2_scores(Yte, ridge_predict(Xte, xbar, ybar, ev, evec, c, best_alpha))

    # honest prompt-level generalization: 120 fit / 30 eval per train task, same alpha
    prng = np.random.RandomState(args.prompt_split_seed)
    fit_rows, ev_rows = [], []
    for t in train_tasks:
        s, e = sl[t]
        perm = prng.permutation(e - s) + s
        fit_rows += list(perm[:120]); ev_rows += list(perm[120:])
    fit_rows = torch.tensor(fit_rows); ev_rows = torch.tensor(ev_rows)
    xb2, yb2, ev2, evec2, c2 = ridge_eig_prep(Xtr[fit_rows], Ytr[fit_rows])
    pe = ridge_predict(Xtr[ev_rows], xb2, yb2, ev2, evec2, c2, best_alpha)
    r2u_up, r2w_up = r2_scores(Ytr[ev_rows], pe)
    # fair oracle on the same eval rows (task mean from the fit rows)
    orc = torch.zeros_like(pe)
    k = 0
    for ti, t in enumerate(train_tasks):
        tm = Ytr[fit_rows[ti * 120:(ti + 1) * 120]].mean(dim=0)
        orc[k:k + 30] = tm
        k += 30
    r2u_orc, _ = r2_scores(Ytr[ev_rows], orc)

    args.out_root.mkdir(parents=True, exist_ok=True)
    out = {"layer": args.layer, "best_alpha": best_alpha, "alpha_pinned": pinned,
           "r2_train_insample_uniform": round(r2u_tr, 4),
           "r2_train_unseenprompts_uniform": round(r2u_up, 4),
           "r2_train_unseenprompts_oracle": round(r2u_orc, 4),
           "r2_test_uniform": round(r2u_te, 4),
           "r2_train_insample_weighted": round(r2w_tr, 4),
           "r2_train_unseenprompts_weighted": round(r2w_up, 4),
           "r2_test_weighted": round(r2w_te, 4)}
    with open(args.out_root / f"layer_{args.layer}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"L{args.layer}: alpha={best_alpha:g}{' PIN' if pinned else ''} | "
          f"insample {r2u_tr:.4f} | unseen-prompts {r2u_up:.4f} (oracle {r2u_orc:.4f}) | "
          f"heldout-tasks {r2u_te:.4f}", flush=True)


if __name__ == "__main__":
    main()
