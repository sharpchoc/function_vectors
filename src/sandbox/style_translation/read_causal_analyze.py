#!/usr/bin/env python
"""Analysis of read_causal.py outputs -> results/code_styles/read_causal/: by_alpha.csv (family x alpha x layer means), summary.json,
headline.png (L24 metrics by alpha, mean over families with 95 % CI), layer_curves.png (dircos / proj_frac by layer for each alpha),
per_family_L24.csv + per_family_grid.png (dircos and proj_frac per family at every alpha), margin.csv (next-token margin towards alt)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
_BOOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(_BOOT))
from src.sandbox.style_translation.models import paths as model_paths


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--pool", default="results/code_styles/code_pool.json"); a = ap.parse_args()
    MP = model_paths(a.model); SRC = MP["prompt_pairs"].parent / "read_causal"; R = _BOOT / "results/code_styles/read_causal"; R.mkdir(parents=True, exist_ok=True)
    pool = json.load(open(_BOOT / a.pool))["pool"]; rows = []; mrows = []; missing = []
    for fam in pool:
        p = SRC / f"{fam}.npz"
        if not p.exists(): missing.append(fam); continue
        z = np.load(p); layers = z["layers"] if "layers" in z else np.arange(1, z["dircos"].shape[2] + 1); arms = list(z["arms"]); alphas = [float(x) for x in z["alphas"]]   # early files: layers 1..28 + control arms
        for ai, arm in enumerate(arms):
            alpha = float(arm.split("_a")[1])
            for li, L in enumerate(layers):
                rows.append(dict(family=fam, arm=arm.split("_a")[0], alpha=alpha, layer=int(L), n=len(z["doc_id"]),
                                 dircos=float(np.nanmean(z["dircos"][:, ai, li])), proj_frac=float(np.nanmedian(z["proj_frac"][:, ai, li])), proj_frac_mean=float(np.nanmean(z["proj_frac"][:, ai, li])),
                                 cos_w=float(np.nanmean(z["cos_w"][:, ai, li])), norm_ratio=float(np.nanmean(z["norm_ratio"][:, ai, li])), cf_cos_w=float(np.nanmean(z["cf_cos_w"][:, li]))))
            m0 = z["lp0"][:, 1] - z["lp0"][:, 0]; mcf = z["lpcf"][:, 1] - z["lpcf"][:, 0]; ma = z["lp"][:, ai, 1] - z["lp"][:, ai, 0]   # margin towards the alt first token
            mrows.append(dict(family=fam, arm=arm.split("_a")[0], alpha=alpha, margin_unsteered=float(m0.mean()), margin_cf=float(mcf.mean()), margin_steered=float(ma.mean()),
                              alt_top_unsteered=float((m0 > 0).mean()), alt_top_cf=float((mcf > 0).mean()), alt_top_steered=float((ma > 0).mean())))
    df = pd.DataFrame(rows); df = df[df.layer >= 20]; df.to_csv(R / "by_alpha.csv", index=False);   # layers 20..28 (user decision); early files also hold 1..19 md = pd.DataFrame(mrows); md.to_csv(R / "margin.csv", index=False)
    d = df[df.arm == "read"]; L24 = d[d.layer == 24]
    ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    summ = {"n_families": int(L24.family.nunique()), "missing": missing, "layers": sorted(d.layer.unique().tolist()), "alphas": sorted(d.alpha.unique().tolist()),
            "L24_by_alpha": {str(al): {m: round(float(g[m].mean()), 3) for m in ("dircos", "proj_frac", "proj_frac_mean", "cos_w", "norm_ratio")} | {"dircos_ci": round(float(ci(g.dircos)), 3)} for al, g in L24.groupby("alpha")},
            "L24_cf_cos_w_mean": round(float(L24.groupby("family").cf_cos_w.first().mean()), 3),
            "margin_by_alpha": {str(al): {k: round(float(g[k].mean()), 3) for k in ("margin_unsteered", "margin_cf", "margin_steered", "alt_top_unsteered", "alt_top_cf", "alt_top_steered")} for al, g in md[md.arm == "read"].groupby("alpha")}}
    best = max(summ["L24_by_alpha"], key=lambda k: summ["L24_by_alpha"][k]["dircos"]); summ["best_alpha_by_L24_dircos"] = float(best)
    json.dump(summ, open(R / "summary.json", "w"), indent=1)
    # headline: L24 metrics by alpha
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8)); g = L24.groupby("alpha")
    for k, (m, lab) in enumerate([("dircos", "cos(Δ steered, Δ counterfactual)"), ("proj_frac", "fraction of counterfactual shift along ŵ (median)"), ("norm_ratio", "|Δ steered| / |Δ counterfactual|")]):
        mu = g[m].mean(); e = g[m].apply(ci); ax[k].errorbar(mu.index, mu.values, yerr=e.values, marker="o", color="#1f5f8b", capsize=3); ax[k].set_xscale("log", base=2); ax[k].set_xticks(mu.index); ax[k].set_xticklabels([f"{x:g}" for x in mu.index])
        ax[k].set_xlabel("α (read feature at L8, every evidence token)"); ax[k].set_title(lab, fontsize=10); ax[k].grid(alpha=.3)
        if m != "norm_ratio": ax[k].set_ylim(0, 1.05 if m == "dircos" else max(1.3, mu.max() * 1.1))
        if m == "proj_frac": ax[k].axhline(1, color="grey", ls="--", lw=1)
    ax[0].axhline(summ["L24_cf_cos_w_mean"], color="grey", ls=":", lw=1); fig.suptitle(f"Read feature → write site (L24 cue), {summ['n_families']} families, k = 3 natural-context prompts steered towards the alternative convention", fontsize=10)
    fig.tight_layout(); fig.savefig(R / "headline.png", dpi=150); plt.close(fig)
    # layer curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for al, g in d.groupby("alpha"):
        gl = g.groupby("layer"); ax[0].plot(gl.dircos.mean().index, gl.dircos.mean().values, marker="o", label=f"α {al:g}"); ax[1].plot(gl.proj_frac.mean().index, gl.proj_frac.mean().values, marker="o", label=f"α {al:g}")
    ax[0].set_title("dircos by cue layer"); ax[1].set_title("fraction of counterfactual shift along ŵ (median over prompts, mean over families)"); ax[1].axhline(1, color="grey", ls="--", lw=1)
    for x in ax: x.set_xlabel("layer at the cue token"); x.grid(alpha=.3); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(R / "layer_curves.png", dpi=150); plt.close(fig)
    # per family at L24
    pf = L24.pivot(index="family", columns="alpha", values="dircos"); pp = L24.pivot(index="family", columns="alpha", values="proj_frac")
    tab = pd.concat({"dircos": pf, "proj_frac": pp}, axis=1); tab["cf_cos_w"] = L24.groupby("family").cf_cos_w.first(); tab.round(3).to_csv(R / "per_family_L24.csv")
    order = pf[float(best)].sort_values(ascending=False).index
    fig, ax = plt.subplots(1, 2, figsize=(12, max(6, .22 * len(order))))
    for k, (P, lab, vmax) in enumerate([(pf, "dircos", 1), (pp.clip(-0.5, 2), "proj_frac (clipped to [-.5, 2])", 2)]):
        im = ax[k].imshow(P.loc[order].values, aspect="auto", cmap="viridis", vmin=0, vmax=vmax); ax[k].set_yticks(range(len(order))); ax[k].set_yticklabels(order, fontsize=7); ax[k].set_xticks(range(P.shape[1])); ax[k].set_xticklabels([f"{c:g}" for c in P.columns]); ax[k].set_xlabel("α"); ax[k].set_title(lab); plt.colorbar(im, ax=ax[k])
    fig.tight_layout(); fig.savefig(R / "per_family_grid.png", dpi=150); plt.close(fig)
    print(json.dumps(summ, indent=1)); print("->", R)


if __name__ == "__main__":
    main()
