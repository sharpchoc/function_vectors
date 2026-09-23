#!/usr/bin/env python
"""Summarize the coding-style identification->execution readout v2 (read_causal.py --controls, 53-family pool, 2026-09-23).

Per family: k = 3 prompts with the model's default (natural) convention demonstrated, 40 held-out documents; the identification
style-contrast vector r (block 7 output, evidence-token mean) is added as -alpha*r at every evidence token (towards the alternative
convention), prefill only. At the query cue (hidden state 24 = output of block 23, the paper's execution block) we compare the
steered shift h_alpha - h_0 with the real shift h_cf - h_0 obtained by demonstrating the alternative convention instead.
Arms: read (own family) | ctrl_family (another pool family's r, seeded, same language when possible) | ctrl_random (random
direction with |r|). Metrics: dircos = cos(h_alpha - h_0, h_cf - h_0); proj_frac = share of the real shift's projection onto the
unit execution direction reproduced by steering (median over prompts); alt_top = next-token top choice is the alternative
convention's first token. Family bootstrap 10,000 resamples, seed 20260923.
Writes results/code_styles/read_causal_v2/{per_family.csv, summary.csv}.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2]
SRC = R / "artifacts" / "style_translation" / "qwen25_code" / "read_causal_v2"
OUT = R / "results" / "code_styles" / "read_causal_v2"
RNG = np.random.default_rng(20260923)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = sorted(json.load(open(R / "results" / "code_styles" / "code_pool.json"))["pool"])
    rows = []
    for fam in pool:
        z = np.load(SRC / f"{fam}.npz", allow_pickle=True)
        layers = list(z["layers"]); li = layers.index(24)
        m0 = z["lp0"][:, 1] - z["lp0"][:, 0]; mcf = z["lpcf"][:, 1] - z["lpcf"][:, 0]
        for ai, arm in enumerate(z["arms"]):
            name, alpha = str(arm).split("_a")[0], float(str(arm).split("_a")[1])
            ma = z["lp"][:, ai, 1] - z["lp"][:, ai, 0]
            rows.append(dict(family=fam, arm=name, alpha=alpha, ctrl_family=str(z["ctrl_family"]),
                             dircos=float(np.nanmean(z["dircos"][:, ai, li])), proj_frac=float(np.nanmedian(z["proj_frac"][:, ai, li])),
                             cos_w=float(np.nanmean(z["cos_w"][:, ai, li])), norm_ratio=float(np.nanmean(z["norm_ratio"][:, ai, li])),
                             cf_cos_w=float(np.nanmean(z["cf_cos_w"][:, li])),
                             alt_top_steered=float((ma > 0).mean()), alt_top_unsteered=float((m0 > 0).mean()), alt_top_cf=float((mcf > 0).mean()),
                             margin_steered=float(ma.mean()), margin_unsteered=float(m0.mean()), margin_cf=float(mcf.mean())))
    d = pd.DataFrame(rows); d.to_csv(OUT / "per_family.csv", index=False)
    summ = []
    for (arm, alpha), g in d.groupby(["arm", "alpha"]):
        g = g.sort_values("family")
        for m in ("dircos", "proj_frac", "cos_w", "norm_ratio", "alt_top_steered", "margin_steered"):
            x = g[m].to_numpy(); b = x[RNG.integers(0, len(x), (10000, len(x)))].mean(1)
            summ.append(dict(arm=arm, alpha=alpha, metric=m, mean=x.mean(), lo=np.quantile(b, .025), hi=np.quantile(b, .975), n_families=len(x)))
    ref = d[(d.arm == "read") & (d.alpha == 2.0)]
    for m in ("cf_cos_w", "alt_top_unsteered", "alt_top_cf", "margin_unsteered", "margin_cf"):
        x = ref[m].to_numpy(); b = x[RNG.integers(0, len(x), (10000, len(x)))].mean(1)
        summ.append(dict(arm="reference", alpha=np.nan, metric=m, mean=x.mean(), lo=np.quantile(b, .025), hi=np.quantile(b, .975), n_families=len(x)))
    s = pd.DataFrame(summ).round(4); s.to_csv(OUT / "summary.csv", index=False)
    piv = s[s.arm != "reference"].pivot_table(index=["arm", "alpha"], columns="metric", values="mean").round(3)
    print(piv[["dircos", "proj_frac", "cos_w", "norm_ratio", "alt_top_steered"]].to_string())
    print(s[s.arm == "reference"].to_string(index=False))
    a2 = d[(d.alpha == 2.0)].pivot_table(index="family", columns="arm", values="dircos")
    print(f"alpha 2: own dircos > both controls in {((a2.read > a2.ctrl_family) & (a2.read > a2.ctrl_random)).sum()}/{len(a2)} families; "
          f"own dircos >= .5 in {(a2.read >= .5).sum()}")


if __name__ == "__main__":
    main()
