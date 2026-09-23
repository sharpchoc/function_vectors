#!/usr/bin/env python
"""Score code-style identification->execution map fits on held-out family centroids (paper protocol).

Same scoring as paper_materials/summarize_coding_map.py: per family, unit-normalise each (document, k) pair difference at the
identification (read, L8) and execution (write, L24) sites, average within the family and re-normalise the centroid; predict
held-out execution centroids from identification centroids with each saved ridge model. Reports R^2 with the mean training-family
execution centroid as reference (paper headline), R^2 with the held-out mean as reference, and mean cos(pred, true).
Usage: score_code_map_centroids.py --prefix v11_pairdiff_unitnorm  -> results/code_styles/read_write_map/sandbox/<prefix>_summary.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2]
ROOT = R / "results" / "code_styles" / "read_write_map" / "sandbox"
PP = R / "artifacts" / "style_translation" / "qwen25_code" / "prompt_pairs"


def centroid(family):
    d = np.load(PP / f"{family}.npz")
    ix = {(v, str(s), int(k)): i for i, (v, s, k) in enumerate(zip(d["doc_id"], d["pole"], d["k"]))}
    keys = sorted({(v, k) for v, s, k in ix if (v, "nat", k) in ix and (v, "alt", k) in ix})
    a = [ix[(v, "nat", k)] for v, k in keys]; b = [ix[(v, "alt", k)] for v, k in keys]
    out = []
    for site in ("read", "write"):
        delta = d[site][a].astype(np.float32) - d[site][b].astype(np.float32)
        n = np.linalg.norm(delta, axis=1, keepdims=True); assert (n > 0).all()
        v = (delta / n).mean(0); out.append(v / np.linalg.norm(v))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="v11_pairdiff_unitnorm"); a = ap.parse_args()
    runs = sorted(p for p in ROOT.glob(f"{a.prefix}_split*_s*") if (p / "fit.json").exists())
    fits = {p.name: json.loads((p / "fit.json").read_text()) for p in runs}
    fams = sorted({f for x in fits.values() for f in x["train_families"] + x["test_families"]})
    C = {f: centroid(f) for f in fams}
    rows = []
    for tag, x in fits.items():
        assert x["select"] == "all" and x["pair_diff"] and x["unit_norm"] and x["read_layer"] == 8
        z = np.load(x["model_file"]); te, tr = x["test_families"], x["train_families"]
        X = np.array([C[f][0] for f in te]); Y = np.array([C[f][1] for f in te]); Ytr = np.array([C[f][1] for f in tr])
        P = (X - z["x_mean"]) @ z["W"] + z["y_mean"]
        r2_tr = 1 - ((Y - P) ** 2).sum() / ((Y - Ytr.mean(0)) ** 2).sum()
        r2_te = 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
        cos = np.mean((P * Y).sum(1) / (np.linalg.norm(P, axis=1) * np.linalg.norm(Y, axis=1)))
        split = "66/34" if "split66" in tag else "80/20"
        rows.append(dict(tag=tag, split=split, n_train=len(tr), n_test=len(te), lam=x["lambda_selected"],
                         lofo_r2=round(x["lofo_r2_at_selected"], 4), r2_train_mean_denom=round(float(r2_tr), 4),
                         r2_test_mean_denom=round(float(r2_te), 4), cos_mean=round(float(cos), 4),
                         js_hungarian_in_split="js_hungarian" in tr + te))
    d = pd.DataFrame(rows).sort_values("tag"); out = ROOT / f"{a.prefix}_summary.csv"; d.to_csv(out, index=False)
    print(d.to_string(index=False))
    for s, g in d.groupby("split"):
        print(f"{s}: R2 train-mean {g.r2_train_mean_denom.mean():.4f} ± {g.r2_train_mean_denom.std(ddof=1):.4f} (sample SD, n={len(g)}) | "
              f"R2 test-mean {g.r2_test_mean_denom.mean():.4f} | cos {g.cos_mean.mean():.4f} | families {sorted(set(g.n_train + g.n_test))}")
    print("wrote", out)


if __name__ == "__main__":
    main()
