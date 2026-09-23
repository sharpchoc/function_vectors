#!/usr/bin/env python
"""Per-prompt read->write ridge map over all read layers (model-agnostic; moved into the repo from
/workspace/msj_icl/icl_69/readfeat/qwen_ridge_readwrite.py on 2026-09-23, paths as arguments).

Reuses the GPT-J solver + protocol verbatim (ridge_eig_prep/ridge_predict, logspace(-1,8,19) lambda by
5-fold task CV seed 42, r2 uniform per-dim eval-mean ref, honest 120/30 prompt split seed 43). X = per-prompt
read acts[:, L]; Y = per-prompt FV; paired per task on prompt_index (the read capture drops label-gate
failures, the FV capture keeps all prompts). Also the task-FV centroid R². Writes <out>/{summary_all28.csv,
taskfv_r2_all28.csv, .ridge_done} (file names keep the historical 'all28' suffix; NL comes from the data).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import ridge_eig_prep, ridge_predict  # noqa: E402
from src.utils.paths import QWEN25_FV_DIR, QWEN25_READ_ARTIFACTS_DIR, QWEN25_SPLIT  # noqa: E402

ALPHAS = list(np.logspace(-1, 8, 19))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read_root", type=Path, default=QWEN25_READ_ARTIFACTS_DIR / "label_resid_perprompt")
    ap.add_argument("--fv_root", type=Path, default=QWEN25_READ_ARTIFACTS_DIR / "perprompt_fvs")
    ap.add_argument("--split_path", type=Path, default=QWEN25_SPLIT)
    ap.add_argument("--out", type=Path, default=QWEN25_FV_DIR / "read_write_map")
    return ap.parse_args()


def r2_uniform(y, pred):
    resid = ((y - pred) ** 2).sum(0); tot = ((y - y.mean(0)) ** 2).sum(0); ok = tot > 0
    return float((1 - resid[ok] / tot[ok]).mean()), float(1 - resid.sum() / tot.sum())


def main():
    a = parse_args()
    READ, FV, SPLIT, OUT = a.read_root, a.fv_root, a.split_path, a.out
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.load(open(SPLIT)); train = sorted(split["train_tasks"]); test = sorted(split["heldout_tasks"])

    def load(t):
        x = torch.load(READ / f"{t}.pt", weights_only=False); f = torch.load(FV / f"{t}.pt", weights_only=False)
        apx = list(x["prompt_index"]); fpx = list(f["prompt_index"]); fmap = {p: i for i, p in enumerate(fpx)}
        sel = [fmap[p] for p in apx]
        return x["acts"].float(), f["fv"].float()[sel]   # (N, NL, resid), (N, resid)
    data = {t: load(t) for t in train + test}
    NL = int(data[train[0]][0].shape[1])
    print("loaded", len(data), "tasks; train rows", sum(data[t][0].shape[0] for t in train), "layers", NL, flush=True)

    summ = open(OUT / "summary_all28.csv", "w"); summ.write("layer,best_alpha,alpha_pinned,r2_train_insample_uniform,r2_train_unseenprompts_uniform,r2_train_unseenprompts_oracle,r2_test_uniform,r2_test_weighted\n")
    tfv = open(OUT / "taskfv_r2_all28.csv", "w"); tfv.write("layer,test_perprompt,test_centroid,train_centroid\n")
    taskFV = {t: data[t][1].mean(0) for t in train + test}
    for L in range(NL):
        Xtr = []; Ytr = []; sl = {}; pos = 0
        for t in train:
            x = data[t][0][:, L]; y = data[t][1]; Xtr.append(x); Ytr.append(y); sl[t] = (pos, pos + len(x)); pos += len(x)
        Xtr = torch.cat(Xtr).to(dev).double(); Ytr = torch.cat(Ytr).to(dev).double()
        rng = np.random.RandomState(42); order = rng.permutation(len(train))
        folds = [sorted(train[i] for i in fold) for fold in np.array_split(order, 5)]
        cv = torch.zeros(len(ALPHAS), dtype=torch.float64, device=dev)
        for fold in folds:
            m = torch.zeros(len(Xtr), dtype=torch.bool)
            for t in fold:
                s, e = sl[t]; m[s:e] = True
            m = m.to(dev); xbar, ybar, evv, evec, c = ridge_eig_prep(Xtr[~m], Ytr[~m]); av = (Xtr[m] - xbar) @ evec
            for ai, al in enumerate(ALPHAS):
                cv[ai] += (((av / (evv + al)) @ c + ybar - Ytr[m]) ** 2).sum()
        bi = int(torch.argmin(cv)); best = float(ALPHAS[bi]); pinned = bi in (0, len(ALPHAS) - 1)
        xbar, ybar, evv, evec, c = ridge_eig_prep(Xtr, Ytr)
        r2u_tr, _ = r2_uniform(Ytr, ridge_predict(Xtr, xbar, ybar, evv, evec, c, best))
        Xte = []; Yte = []; tsl = {}; p = 0
        for t in test:
            x = data[t][0][:, L]; y = data[t][1]; Xte.append(x); Yte.append(y); tsl[t] = (p, p + len(x)); p += len(x)
        Xte = torch.cat(Xte).to(dev).double(); Yte = torch.cat(Yte).to(dev).double()
        pred_te = ridge_predict(Xte, xbar, ybar, evv, evec, c, best)
        r2u_te, r2w_te = r2_uniform(Yte, pred_te)
        prng = np.random.RandomState(43); fit_rows = []; ev_rows = []
        for t in train:
            s, e = sl[t]; perm = prng.permutation(e - s) + s
            n = e - s; k = min(120, int(n * 0.8)); fit_rows += list(perm[:k]); ev_rows += list(perm[k:])
        fit_rows = torch.tensor(fit_rows); ev_rows = torch.tensor(ev_rows)
        xb2, yb2, ev2, evec2, c2 = ridge_eig_prep(Xtr[fit_rows], Ytr[fit_rows])
        pe = ridge_predict(Xtr[ev_rows], xb2, yb2, ev2, evec2, c2, best); r2u_up, _ = r2_uniform(Ytr[ev_rows], pe)
        orc = torch.zeros_like(pe); kk = 0; fitset = set(fit_rows.tolist())
        for t in train:
            s, e = sl[t]; tf = [r for r in range(s, e) if r in fitset]; te_ = [r for r in range(s, e) if r not in fitset]
            if te_:
                orc[kk:kk + len(te_)] = Ytr[torch.tensor(tf)].mean(0); kk += len(te_)
        r2u_orc, _ = r2_uniform(Ytr[ev_rows], orc)
        cen_pred = torch.stack([pred_te[tsl[t][0]:tsl[t][1]].mean(0) for t in test])
        cen_true = torch.stack([taskFV[t].to(dev).double() for t in test])
        r2_cen, _ = r2_uniform(cen_true, cen_pred)
        pp_true = torch.cat([taskFV[t].to(dev).double().expand(tsl[t][1] - tsl[t][0], -1) for t in test])
        r2_pp, _ = r2_uniform(pp_true, pred_te)
        ctr_pred = torch.stack([ridge_predict(Xtr[sl[t][0]:sl[t][1]], xbar, ybar, evv, evec, c, best).mean(0) for t in train])
        ctr_true = torch.stack([taskFV[t].to(dev).double() for t in train]); r2_ctr, _ = r2_uniform(ctr_true, ctr_pred)
        summ.write(f"{L},{best:g},{pinned},{r2u_tr:.4f},{r2u_up:.4f},{r2u_orc:.4f},{r2u_te:.4f},{r2w_te:.4f}\n"); summ.flush()
        tfv.write(f"{L},{r2_pp:.4f},{r2_cen:.4f},{r2_ctr:.4f}\n"); tfv.flush()
        print(f"L{L}: a={best:g}{' PIN' if pinned else ''} test_perprompt_uniform={r2u_te:.4f} test_centroid={r2_cen:.4f}", flush=True)
    summ.close(); tfv.close(); open(OUT / ".ridge_done", "w").write("ok\n"); print("RIDGE DONE", flush=True)


if __name__ == "__main__":
    main()
