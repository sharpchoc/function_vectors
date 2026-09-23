#!/usr/bin/env python
"""Matched identification -> execution maps for GPT-J and Qwen2.5-7B-Instruct (reviewer fix 2026-09-23).

Runs the SAME two protocols on either model and scores them with the SAME metrics, so the paper's
"not a matched comparison" caveats can go:

  T  task-level map  u_A -> v_A  (the GPT-J Claim-6 protocol, claim6_meanresid_map.py):
     u_A = per-prompt mean of [z_l - <z_l, g_hat_l> g_hat_l] averaged over the extraction band
     (GPT-J 5-7, Qwen 11-13); g_l = mean of task means m_A(l) over the CARRIER POOL.
     Carrier pool = "all" (the original pooled construction, reproduced as a check) or
     "train" (training tasks only -> no held-out task touches any preprocessing step).
     Dual ridge, training-mean centering, lambda by closed-form LOO over logspace(-2, 6, 17).
  P  per-prompt map  z_{L_id} -> v_A^j  (the Qwen protocol, ridge_readwrite_generic.py):
     raw residual at the last token of the 10th demo label, block L_id (GPT-J 6, Qwen 12);
     primal ridge, lambda by 5-fold task CV (seed 42) over logspace(-1, 8, 19).

Metrics (all on the held-out tasks):
  pooled_r2            1 - sum_A ||v_A - v_hat_A||^2 / sum_A ||v_A - mean_test v||^2 on task centroids (GPT-J headline)
  coord_r2             coordinate-averaged R^2 on task centroids, held-out mean reference (Qwen headline)
  pooled_r2_trainref   pooled_r2 with the training-mean FV as the reference
  perprompt_coord_r2   (P only) coordinate-averaged R^2 against per-prompt FVs (Qwen secondary)
For P the task-centroid prediction is the mean of the per-prompt predictions.

Writes <out>/matched_maps.csv (one row per protocol x carrier pool) and prints it. CPU, fp64.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import (ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR, QWEN25_FV_DIR,  # noqa: E402
                             QWEN25_READ_ARTIFACTS_DIR, QWEN25_SPLIT)
from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import ridge_eig_prep, ridge_predict  # noqa: E402

CFG = {
    "gptj": dict(read=ARTIFACTS_ROOT / "69_task_run" / "label_resid_perprompt",
                 means=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means",
                 fv=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs",
                 split=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json",
                 band=(5, 6, 7), id_block=6,
                 out=TASK69_RUN_DIR / "understanding_read_write_linear_map" / "matched_maps"),
    "qwen25": dict(read=QWEN25_READ_ARTIFACTS_DIR / "label_resid_perprompt",
                   means=QWEN25_READ_ARTIFACTS_DIR / "label_resid_means",
                   fv=QWEN25_READ_ARTIFACTS_DIR / "perprompt_fvs",
                   split=QWEN25_SPLIT, band=(11, 12, 13), id_block=12,
                   out=QWEN25_FV_DIR / "read_write_map" / "matched_maps"),
}
LOO_GRID = np.logspace(-2, 6, 17)
CV_GRID = list(np.logspace(-1, 8, 19))


def pooled_r2(pred, true, ref):
    return float(1.0 - ((true - pred) ** 2).sum() / ((true - ref) ** 2).sum())


def coord_r2(pred, true):
    resid = ((true - pred) ** 2).sum(0)
    tot = ((true - true.mean(0)) ** 2).sum(0)
    ok = tot > 0
    return float((1 - resid[ok] / tot[ok]).mean())


def ridge_dual_loo(Xc, Yc):
    K = Xc @ Xc.T
    n = len(Xc)
    best = (None, np.inf)
    for lam in LOO_GRID:
        H = K @ np.linalg.inv(K + lam * np.eye(n))
        E = (Yc - H @ Yc) / (1.0 - np.diag(H))[:, None]
        m = (E ** 2).mean()
        if m < best[1]:
            best = (lam, m)
    A = np.linalg.solve(K + best[0] * np.eye(n), Yc)
    return A, float(best[0])


def procrustes(Xc, Yc, tol=1e-10):
    """Orthogonal Procrustes as in claim6_meanresid_map.py: R (partial isometry on span(Xc)) and the
    trace-formula global scale."""
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    keep = S > tol * S[0]
    U, S, Vt = U[:, keep], S[keep], Vt[keep]
    Uy, Sy, Vty = np.linalg.svd(Yc, full_matrices=False)
    small = (S[:, None] * (U.T @ Uy)) * Sy[None, :]
    Us, Ss, Vts = np.linalg.svd(small, full_matrices=False)
    R = (Vt.T @ Us) @ (Vts @ Vty)
    return R, float(Ss.sum() / (S ** 2).sum())


def load(cfg):
    split = json.load(open(cfg["split"]))
    train, test = sorted(split["train_tasks"]), sorted(split["heldout_tasks"])
    band = list(cfg["band"])
    keep = sorted(set(band) | {cfg["id_block"]})     # only the blocks the fits use (16 GB container limit)
    cfg["pos"] = {l: i for i, l in enumerate(keep)}
    acts, fvs, means = {}, {}, {}
    for t in train + test:
        x = torch.load(cfg["read"] / f"{t}.pt", map_location="cpu", weights_only=False)
        f = torch.load(cfg["fv"] / f"{t}.pt", map_location="cpu", weights_only=False)
        a = x["acts"][:, keep].double()
        fv = f["fv"].double()
        if "prompt_index" in x:   # Qwen read capture may drop prompts; pair on prompt_index
            fmap = {p: i for i, p in enumerate(list(f["prompt_index"]))}
            fv_pp = fv[[fmap[p] for p in list(x["prompt_index"])]]
        else:
            assert len(a) == len(fv)
            fv_pp = fv
        acts[t] = a                                   # (N, len(keep), d)
        del x
        fvs[t] = (fv_pp, fv.mean(0))                  # per-prompt (paired), task FV (all prompts)
        means[t] = torch.load(cfg["means"] / f"{t}.pt", map_location="cpu",
                              weights_only=False)["resid_means"][band].double()   # (|B|, d)
    return train, test, acts, fvs, means


def protocol_T(cfg, train, test, acts, fvs, means, pool):
    band = list(cfg["band"])
    pool_tasks = train if pool == "train" else train + test
    g = torch.stack([means[t] for t in pool_tasks]).mean(0)
    gh = g / g.norm(dim=1, keepdim=True)                       # per-block carrier direction

    def u_task(t):
        z = acts[t][:, [cfg["pos"][l] for l in band]]          # (N, |B|, d)
        r = z - (z * gh).sum(-1, keepdim=True) * gh
        return r.mean(1).mean(0).numpy()
    X = {t: u_task(t) for t in train + test}
    Xtr = np.stack([X[t] for t in train]); Xte = np.stack([X[t] for t in test])
    Ytr = np.stack([fvs[t][1].numpy() for t in train]); Yte = np.stack([fvs[t][1].numpy() for t in test])
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    A, lam = ridge_dual_loo(Xtr - xm, Ytr - ym)
    pred = (Xte - xm) @ (Xtr - xm).T @ A + ym
    R, scale = procrustes(Xtr - xm, Ytr - ym)
    p_rot = (Xte - xm) @ R + ym
    p_rs = scale * ((Xte - xm) @ R) + ym
    return {"protocol": "T_task_uA", "carrier_pool": pool, "block": f"{band[0]}-{band[-1]}", "lambda": lam,
            "rotation_pooled_r2": pooled_r2(p_rot, Yte, Yte.mean(0)),
            "rotscale_pooled_r2": pooled_r2(p_rs, Yte, Yte.mean(0)), "rot_scale": scale,
            "pooled_r2": pooled_r2(pred, Yte, Yte.mean(0)), "coord_r2": coord_r2(pred, Yte),
            "pooled_r2_trainref": pooled_r2(pred, Yte, ym), "perprompt_coord_r2": "",
            "trainmean_pooled_r2": pooled_r2(np.repeat(ym[None], len(Yte), 0), Yte, Yte.mean(0)),
            "n_train": len(train), "n_test": len(test)}


def protocol_P(cfg, train, test, acts, fvs):
    L = cfg["id_block"]

    def stack(ts):
        X = torch.cat([acts[t][:, cfg["pos"][L]] for t in ts]); Y = torch.cat([fvs[t][0] for t in ts])
        sl, p = {}, 0
        for t in ts:
            sl[t] = (p, p + len(acts[t])); p += len(acts[t])
        return X, Y, sl
    Xtr, Ytr, sl = stack(train)
    rng = np.random.RandomState(42); order = rng.permutation(len(train))
    folds = [sorted(train[i] for i in f) for f in np.array_split(order, 5)]
    cv = torch.zeros(len(CV_GRID), dtype=torch.float64)
    for fold in folds:
        m = torch.zeros(len(Xtr), dtype=torch.bool)
        for t in fold:
            s, e = sl[t]; m[s:e] = True
        xbar, ybar, evv, evec, c = ridge_eig_prep(Xtr[~m], Ytr[~m]); av = (Xtr[m] - xbar) @ evec
        for ai, al in enumerate(CV_GRID):
            cv[ai] += (((av / (evv + al)) @ c + ybar - Ytr[m]) ** 2).sum()
    best = float(CV_GRID[int(torch.argmin(cv))])
    xbar, ybar, evv, evec, c = ridge_eig_prep(Xtr, Ytr)
    Xte, Yte_pp, tsl = stack(test)
    pred_pp = ridge_predict(Xte, xbar, ybar, evv, evec, c, best)
    cen_pred = torch.stack([pred_pp[tsl[t][0]:tsl[t][1]].mean(0) for t in test]).numpy()
    cen_true = np.stack([fvs[t][1].numpy() for t in test])
    ym = torch.stack([fvs[t][1] for t in train]).mean(0).numpy()
    return {"protocol": "P_perprompt_raw", "carrier_pool": "none", "block": str(L), "lambda": best,
            "rotation_pooled_r2": "", "rotscale_pooled_r2": "", "rot_scale": "",
            "pooled_r2": pooled_r2(cen_pred, cen_true, cen_true.mean(0)), "coord_r2": coord_r2(cen_pred, cen_true),
            "pooled_r2_trainref": pooled_r2(cen_pred, cen_true, ym),
            "perprompt_coord_r2": coord_r2(pred_pp.numpy(), Yte_pp.numpy()),
            "trainmean_pooled_r2": pooled_r2(np.repeat(ym[None], len(cen_true), 0), cen_true, cen_true.mean(0)),
            "n_train": len(train), "n_test": len(test)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(CFG), required=True)
    ap.add_argument("--skip_P", action="store_true")
    ap.add_argument("--out_name", default="matched_maps.csv")
    a = ap.parse_args()
    cfg = CFG[a.model]
    torch.set_num_threads(4)
    train, test, acts, fvs, means = load(cfg)
    print(f"{a.model}: {len(train)} train / {len(test)} held-out tasks", flush=True)
    rows = [protocol_T(cfg, train, test, acts, fvs, means, "all"),
            protocol_T(cfg, train, test, acts, fvs, means, "train")]
    for r in rows:
        print(r, flush=True)
    if not a.skip_P:
        rows.append(protocol_P(cfg, train, test, acts, fvs)); print(rows[-1], flush=True)
    cfg["out"].mkdir(parents=True, exist_ok=True)
    with open(cfg["out"] / a.out_name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", cfg["out"] / a.out_name)


if __name__ == "__main__":
    main()
