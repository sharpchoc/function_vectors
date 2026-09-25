#!/usr/bin/env python
"""Backup coding identification->execution map WITHOUT double normalisation (not in the paper; 2026-09-25).

v11 (paper, R^2 = .237 +- .030) unit-normalises every (document, k) pair contrast delta^q_fj before fitting, and at scoring
averages the unit contrasts per family and normalises the average again. Here the contrasts are normalised ONCE:
  c^q_f = normalize( mean_j delta^q_fj )        (raw pair contrasts averaged, then one normalisation)
Rows written (one per variant x split, same split files as v11):
  v11_ref          v11 fits scored with v11's double-norm centroids (must reproduce v11_pairdiff_unitnorm_summary.csv)
  v11_on_single    v11 fits scored on single-norm centroids (isolates the scoring change)
  A_rawpairs       ridge fitted on RAW per-pair contrasts (read_write_map_code.py without --unit_norm, tag v12a_rawpairs_*);
                   inputs are the raw family-mean id contrasts (scale-consistent). Reported: r2_raw (raw family-mean exec
                   contrasts, train-mean ref), r2_train_mean_denom / r2_test_mean_denom after normalising the prediction
                   vs unit single-norm targets, and cos.
  B_centroids      ridge fitted on the training families' single-norm centroids (one example per family, dual form,
                   lambda by leave-one-family-out on SSE), scored like v11 (prediction not renormalised).
Output: results/code_styles/read_write_map/sandbox/v12_single_norm_summary.csv. CPU, fp64.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2]
ROOT = R / "results" / "code_styles" / "read_write_map" / "sandbox"
SPLITS = R / "results" / "code_styles" / "read_write_map"
PP = R / "artifacts" / "style_translation" / "qwen25_code" / "prompt_pairs"
B_LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]


def nrm(v):
    return v / np.linalg.norm(v)


def family_stats(family):
    """same pair keys as score_code_map_centroids.centroid; per site: double-norm centroid, single-norm centroid, raw mean"""
    d = np.load(PP / f"{family}.npz")
    ix = {(v, str(s), int(k)): i for i, (v, s, k) in enumerate(zip(d["doc_id"], d["pole"], d["k"]))}
    keys = sorted({(v, k) for v, s, k in ix if (v, "nat", k) in ix and (v, "alt", k) in ix})
    a = [ix[(v, "nat", k)] for v, k in keys]; b = [ix[(v, "alt", k)] for v, k in keys]
    out = {}
    for site in ("read", "write"):
        delta = d[site][a].astype(np.float64) - d[site][b].astype(np.float64)
        n = np.linalg.norm(delta, axis=1, keepdims=True); assert (n > 0).all()
        raw = delta.mean(0)
        out[site] = dict(double=nrm((delta / n).mean(0)), single=nrm(raw), raw=raw)
    return out


def r2(Y, P, ref):
    return float(1 - ((Y - P) ** 2).sum() / ((Y - ref) ** 2).sum())


def cosm(P, Y):
    return float(np.mean((P * Y).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(Y, axis=1))))


def dual_ridge(X, Y, lam):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    A = np.linalg.solve(Xc @ Xc.T + lam * np.eye(len(X)), Yc)
    return lambda Z: (Z - mx) @ Xc.T @ A + my


def main():
    split_files = sorted(SPLITS.glob("split_v10_*_s*.json"))
    fams = sorted({f for sf in split_files for part in ("train", "test") for f in json.load(open(sf))[part]})
    S = {f: family_stats(f) for f in fams}
    get = lambda fs, site, kind: np.array([S[f][site][kind] for f in fs])
    rows = []
    for sf in split_files:
        sp = json.load(open(sf)); tr, te = sorted(sp["train"]), sorted(sp["test"])
        split = "66/34" if "_66_" in sf.name else "80/20"; s = sf.stem.split("_")[-1]
        base = dict(split=split, seed=s, n_train=len(tr), n_test=len(te))
        # v11 fits: reference + scored on single-norm centroids
        v11 = ROOT / f"v11_pairdiff_unitnorm_split{sf.stem.split('_')[2]}_{s}" / "fit.json"
        x = json.loads(v11.read_text()); assert sorted(x["test_families"]) == te
        z = np.load(x["model_file"]); W, xm, ym = z["W"].astype(np.float64), z["x_mean"].astype(np.float64), z["y_mean"].astype(np.float64)
        for tag, kind in (("v11_ref", "double"), ("v11_on_single", "single")):
            X, Y, Ytr = get(te, "read", kind), get(te, "write", kind), get(tr, "write", kind)
            P = (X - xm) @ W + ym
            rows.append(dict(variant=tag, **base, lam=x["lambda_selected"], r2_train_mean_denom=r2(Y, P, Ytr.mean(0)),
                             r2_test_mean_denom=r2(Y, P, Y.mean(0)), cos_mean=cosm(P, Y), r2_raw=np.nan))
        # variant A: raw-pair fits
        fa = ROOT / f"v12a_rawpairs_split{sf.stem.split('_')[2]}_{s}" / "fit.json"
        if fa.exists():
            xa = json.loads(fa.read_text()); assert not xa["unit_norm"] and xa["pair_diff"] and xa["select"] == "all" and sorted(xa["test_families"]) == te
            za = np.load(xa["model_file"]); W, xm, ym = (za[k].astype(np.float64) for k in ("W", "x_mean", "y_mean"))
            Praw = (get(te, "read", "raw") - xm) @ W + ym
            Yraw, Ytr_raw = get(te, "write", "raw"), get(tr, "write", "raw")
            Pn = Praw / np.linalg.norm(Praw, axis=1, keepdims=True)
            Y, Ytr = get(te, "write", "single"), get(tr, "write", "single")
            rows.append(dict(variant="A_rawpairs", **base, lam=xa["lambda_selected"], r2_train_mean_denom=r2(Y, Pn, Ytr.mean(0)),
                             r2_test_mean_denom=r2(Y, Pn, Y.mean(0)), cos_mean=cosm(Praw, Y), r2_raw=r2(Yraw, Praw, Ytr_raw.mean(0))))
        # variant B: fit on training-family single-norm centroids
        Xtr, Ytr = get(tr, "read", "single"), get(tr, "write", "single")
        sse = {}
        for lam in B_LAMS:
            e = 0.0
            for i in range(len(tr)):
                m = np.arange(len(tr)) != i
                e += float(((dual_ridge(Xtr[m], Ytr[m], lam)(Xtr[i:i + 1]) - Ytr[i:i + 1]) ** 2).sum())
            sse[lam] = e
        lam = min(sse, key=sse.get)
        X, Y = get(te, "read", "single"), get(te, "write", "single")
        P = dual_ridge(Xtr, Ytr, lam)(X)
        rows.append(dict(variant="B_centroids", **base, lam=lam, r2_train_mean_denom=r2(Y, P, Ytr.mean(0)),
                         r2_test_mean_denom=r2(Y, P, Y.mean(0)), cos_mean=cosm(P, Y), r2_raw=np.nan,
                         lam_at_grid_edge=lam in (B_LAMS[0], B_LAMS[-1])))
    d = pd.DataFrame(rows).sort_values(["variant", "split", "seed"]).round(4)
    out = ROOT / "v12_single_norm_summary.csv"; d.to_csv(out, index=False)
    print(d.to_string(index=False))
    for (v, sp), g in d.groupby(["variant", "split"]):
        print(f"{v:14s} {sp}: R2 train-mean {g.r2_train_mean_denom.mean():.3f} +- {g.r2_train_mean_denom.std(ddof=1):.3f} | "
              f"R2 test-mean {g.r2_test_mean_denom.mean():.3f} | cos {g.cos_mean.mean():.3f} | r2_raw {g.r2_raw.mean():.3f} | n={len(g)}")
    print("wrote", out)


if __name__ == "__main__":
    main()
