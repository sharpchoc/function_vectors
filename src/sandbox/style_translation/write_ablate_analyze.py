#!/usr/bin/env python
"""Analysis of write_ablate.py records -> results/code_styles/write_ablation/: full.csv (family x context x arm), summary.json (pooled over
families), headline.png (per context: base / own zero / own mean / cf zero / cf mean; mean ± 95 % CI over families), per_family_grid.png
(Δ success vs base), margin.csv. success = context convention kept AND judge OK (judge_rollouts)."""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
_BOOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(_BOOT))
from src.sandbox.style_translation.models import paths as model_paths
ARMS = ["base", "own_zero", "own_mean", "cf_zero", "cf_mean"]; LAB = {"base": "unablated", "own_zero": "own: zero-proj", "own_mean": "own: mean", "cf_zero": "counterfactual: zero-proj", "cf_mean": "counterfactual: mean"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--pool", default="results/code_styles/code_pool.json"); a = ap.parse_args()
    MP = model_paths(a.model); SRC = MP["steering"].parent / "ablation" / "full_k4"; R = _BOOT / "results/code_styles/write_ablation"; R.mkdir(parents=True, exist_ok=True)
    pool = json.load(open(_BOOT / a.pool))["pool"]; rows = []; missing = []; unj = 0
    for fam in pool:
        ps = glob.glob(str(SRC / "*" / f"{fam}.json"))
        if not ps: missing.append(fam); continue
        recs = json.load(open(ps[0])); unj += sum(1 for r in recs if not r.get("judge"))
        for ctx in ("nat", "alt"):
            for arm in ARMS:
                rr = [r for r in recs if r["context_style"] == ctx and r["arm"] == arm]
                if not rr: continue
                ok = [bool(r["style_ok"]) and bool((r.get("judge") or {}).get("ok")) for r in rr]
                rows.append(dict(family=fam, context=ctx, arm=arm, direction_family=rr[0].get("direction_family"), n=len(rr), success=np.mean(ok), convention_rate=np.mean([bool(r["style_ok"]) for r in rr]),
                                 judge_ok=np.mean([bool((r.get("judge") or {}).get("ok")) for r in rr]), unscorable=np.mean([r["decision"] is None for r in rr]), margin_ctx=np.mean([r["margin_ctx"] for r in rr]),
                                 top1_ctx=np.mean([r["top1"] == "ctx" for r in rr]), proj_before=np.mean([r["proj_before"] for r in rr if r["proj_before"] is not None]) if arm != "base" else np.nan))
    df = pd.DataFrame(rows); df.to_csv(R / "full.csv", index=False)
    piv = df.pivot_table(index=["family", "context"], columns="arm", values="success")[ARMS]; piv.round(3).to_csv(R / "success_by_arm.csv")
    ci = lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))
    summ = {"n_families": int(df.family.nunique()), "missing": missing, "unjudged": unj}
    for ctx in ("nat", "alt", "both"):
        p = piv.xs(ctx, level="context") if ctx != "both" else piv.groupby(level="family").mean()
        summ[ctx] = {arm: {"success": round(float(p[arm].mean()), 3), "ci": round(float(ci(p[arm])), 3)} for arm in ARMS}
        d = p.sub(p["base"], axis=0); summ[ctx]["drop_own_zero"] = round(float(-d["own_zero"].mean()), 3); summ[ctx]["drop_own_mean"] = round(float(-d["own_mean"].mean()), 3)
        summ[ctx]["drop_cf_zero"] = round(float(-d["cf_zero"].mean()), 3); summ[ctx]["drop_cf_mean"] = round(float(-d["cf_mean"].mean()), 3)
        summ[ctx]["families_own_zero_drop_ge_.20_and_cf_lt_.10"] = int(((-d["own_zero"] >= .20) & (-d["cf_zero"] < .10)).sum())
        summ[ctx]["families_own_mean_drop_ge_.20_and_cf_lt_.10"] = int(((-d["own_mean"] >= .20) & (-d["cf_mean"] < .10)).sum())
    m = df.pivot_table(index=["family", "context"], columns="arm", values="margin_ctx")[ARMS]; m.round(3).to_csv(R / "margin.csv")
    summ["margin_ctx_mean"] = {arm: round(float(m[arm].mean()), 3) for arm in ARMS}
    json.dump(summ, open(R / "summary.json", "w"), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for k, ctx in enumerate(("nat", "alt")):
        p = piv.xs(ctx, level="context"); mu = [p[arm].mean() for arm in ARMS]; e = [ci(p[arm]) for arm in ARMS]
        cols = ["#6b7280", "#1f5f8b", "#5aa0d0", "#b3261e", "#e08a80"]; ax[k].bar(range(5), mu, yerr=e, color=cols, capsize=3); ax[k].set_xticks(range(5)); ax[k].set_xticklabels([LAB[x] for x in ARMS], rotation=25, ha="right", fontsize=8)
        ax[k].set_title(f"{'natural' if ctx == 'nat' else 'alternative'}-convention demonstrations (k = 4)", fontsize=10); ax[k].set_ylim(0, 1); ax[k].grid(axis="y", alpha=.3)
    ax[0].set_ylabel("keeps the demonstrated convention (judge OK)"); fig.suptitle(f"Write-feature ablation at the L24 cue token, {summ['n_families']} families, 40 held-out documents per pole", fontsize=10)
    fig.tight_layout(); fig.savefig(R / "headline.png", dpi=150); plt.close(fig)
    d = piv.groupby(level="family").mean(); d = d.sub(d["base"], axis=0)[ARMS[1:]]; order = d["own_zero"].sort_values().index
    fig, ax = plt.subplots(figsize=(6, max(6, .2 * len(order)))); im = ax.imshow(d.loc[order].values, aspect="auto", cmap="RdBu", vmin=-.8, vmax=.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7); ax.set_xticks(range(4)); ax.set_xticklabels([LAB[x] for x in ARMS[1:]], rotation=25, ha="right", fontsize=8); ax.set_title("Δ success vs unablated (mean of both contexts)", fontsize=9); plt.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(R / "per_family_grid.png", dpi=150); plt.close(fig)
    print(json.dumps({k: v for k, v in summ.items() if k in ("n_families", "missing", "unjudged", "both", "margin_ctx_mean")}, indent=1)); print("->", R)


if __name__ == "__main__":
    main()
