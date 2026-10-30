#!/usr/bin/env python
"""Is the learned (shared, task-level) read->write map close to the FV heads' summed OV circuit?
(USER REQUEST 2026-10-30)

Learned map M_L: protocol T of matched_readwrite_maps.py (u_A -> v_A, dual ridge on the training tasks,
training-only carrier, training-mean centering); as a column map M_L = A^T Xc, rank <= n_train - 1, defined
only on S = span(centered training u_A).
OV map: M_OV = sum_{h in FV heads} W_O^h W_V^{kv(h)} diag(gamma_l(h)) ("gamma"), or without diag(gamma)
("nogamma"). Linear part only: biases are constants and drop out on centered inputs; the norm's scalar
normalisation does not change directions.
Measures (Q = orthonormal basis of S; held-out tasks unless stated):
  mat_cos        Frobenius cos(M_L Q, M_OV Q)                    (the two maps on the input span)
  mat_cos_Yproj  same with M_OV's output projected onto span(centered training FVs)
  fn_cos_test    mean_A cos(M_L x_A, M_OV x_A), x_A = u_A - mean_train u   (also fn_cos_train)
  ov_fv_cos      mean_A cos(M_OV x_A, v_A - mean_train v)        (ridge's own value: ridge_fv_cos)
  ov_r2          pooled R^2 of alpha M_OV x_A + mean_train v, alpha least-squares on the training tasks
                 (ridge: ridge_r2 = the matched-map headline)
Null: --n_null draws of random non-FV heads, layer-matched (same count per layer as the FV heads).
Writes <out>/ov_vs_learned_map.csv (rows: fv / null_mean / null_sd / null_max per gamma variant). CPU.
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
from src.eval_scripts.matched_readwrite_maps import CFG as MAP_CFG, load, ridge_dual_loo, pooled_r2  # noqa: E402
from src.eval_scripts.ov_circuit_readwrite import MODELS as OV_CFG, Weights, layer_weights  # noqa: E402

KEYS = {"gptj": ("gptj", "gptj"), "qwen": ("qwen25", "qwen")}


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(KEYS), required=True)
    ap.add_argument("--n_null", type=int, default=20)
    a = ap.parse_args()
    torch.set_num_threads(4)
    mcfg, ocfg = MAP_CFG[KEYS[a.model][0]], OV_CFG[KEYS[a.model][1]]
    out_dir = ocfg["out"]

    # ---- learned map (protocol T, training-only carrier), as in matched_readwrite_maps.protocol_T ----
    train, test, acts, fvs, means = load(mcfg)
    band = list(mcfg["band"])
    g = torch.stack([means[t] for t in train]).mean(0)
    gh = g / g.norm(dim=1, keepdim=True)

    def u_task(t):
        z = acts[t][:, [mcfg["pos"][l] for l in band]]
        r = z - (z * gh).sum(-1, keepdim=True) * gh
        return r.mean(1).mean(0).numpy()
    Xtr = np.stack([u_task(t) for t in train]); Xte = np.stack([u_task(t) for t in test])
    Ytr = np.stack([fvs[t][1].numpy() for t in train]); Yte = np.stack([fvs[t][1].numpy() for t in test])
    del acts
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc, Xtc, Ytc = Xtr - xm, Ytr - ym, Xte - xm, Yte - ym
    A, lam = ridge_dual_loo(Xc, Yc)
    ML = lambda X: X @ Xc.T @ A          # row-vector form: rows of X -> predicted centered FVs
    ridge_r2 = pooled_r2(ML(Xtc) + ym, Yte, Yte.mean(0))
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Q = Vt[S > 1e-10 * S[0]]             # (k, d) orthonormal rows spanning the training inputs
    UY, SY, _ = np.linalg.svd(Yc.T, full_matrices=False)
    PY = UY[:, SY > 1e-10 * SY[0]]       # (d, k') output span
    MLQ = ML(Q)
    print(f"{a.model}: {len(train)} train / {len(test)} test tasks, ridge lambda {lam:g}, held-out R^2 {ridge_r2:.4f}, "
          f"input span rank {len(Q)}", flush=True)

    # ---- OV maps ----
    W = Weights(ocfg["ckpt"])
    H, nkv = ocfg["n_heads"], ocfg["n_kv"]
    per_kv = H // nkv
    sel = sorted(int(f) for f in json.load(open(ocfg["selection"]))["selected_flat"])
    by_layer = {}
    for f in sel:
        by_layer.setdefault(f // H, []).append(f % H)
    LW = {l: layer_weights(W, KEYS[a.model][1], l) for l in by_layer}
    hd = LW[next(iter(LW))]["wo"].shape[0] // H
    Xall = torch.from_numpy(np.concatenate([Q, Xc, Xtc])).float()      # every input we need, rows
    nq, ntr = len(Q), len(Xc)

    def apply_ov(heads_by_layer, use_gamma):
        Yo = torch.zeros(len(Xall), Xall.shape[1])
        for l, hs in heads_by_layer.items():
            lw = LW[l]
            Xin = Xall * lw["nw"] if use_gamma else Xall
            for h in hs:
                gk = h // per_kv
                Yo += (Xin @ lw["wv"][gk * hd:(gk + 1) * hd].T) @ lw["wo"][:, h * hd:(h + 1) * hd].T
        Yo = Yo.double().numpy()
        return Yo[:nq], Yo[nq:nq + ntr], Yo[nq + ntr:]

    def measures(heads_by_layer, use_gamma):
        MQ, Otr, Ote = apply_ov(heads_by_layer, use_gamma)
        alpha = float((Otr * Yc).sum() / (Otr * Otr).sum())
        return {"mat_cos": cos(MLQ, MQ), "mat_cos_Yproj": cos(MLQ, (MQ @ PY) @ PY.T),
                "fn_cos_test": float(np.mean([cos(p, o) for p, o in zip(ML(Xtc), Ote)])),
                "fn_cos_train": float(np.mean([cos(p, o) for p, o in zip(ML(Xc), Otr)])),
                "ov_fv_cos": float(np.mean([cos(o, y) for o, y in zip(Ote, Ytc)])),
                "ov_r2": pooled_r2(alpha * Ote + ym, Yte, Yte.mean(0)), "alpha": alpha}

    ridge_fv_cos = float(np.mean([cos(p, y) for p, y in zip(ML(Xtc), Ytc)]))
    rng = np.random.default_rng(0)
    rows = []
    for use_gamma in (True, False):
        var = "gamma" if use_gamma else "nogamma"
        fv_m = measures(by_layer, use_gamma)
        nulls = []
        for i in range(a.n_null):
            rnd = {l: sorted(rng.choice([h for h in range(H) if h not in hs], len(hs), replace=False).tolist())
                   for l, hs in by_layer.items()}
            nulls.append(measures(rnd, use_gamma))
        keys = list(fv_m)
        rows.append({"variant": var, "heads": "fv", **fv_m})
        for stat, fn in (("null_mean", np.mean), ("null_sd", np.std), ("null_max", np.max)):
            rows.append({"variant": var, "heads": stat, **{k: float(fn([n[k] for n in nulls])) for k in keys}})
        print(var, {k: round(v, 3) for k, v in fv_m.items()}, flush=True)
        print(var, "null mean", {k: round(float(np.mean([n[k] for n in nulls])), 3) for k in keys}, flush=True)
    for r in rows:
        r.update(ridge_r2=ridge_r2, ridge_fv_cos=ridge_fv_cos, n_fv_heads=len(sel), n_null=a.n_null)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ov_vs_learned_map.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"ridge: held-out R^2 {ridge_r2:.4f}, cos(pred, v_A - mean) {ridge_fv_cos:.4f}; wrote {out_dir}")


if __name__ == "__main__":
    main()
