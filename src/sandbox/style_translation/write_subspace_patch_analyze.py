#!/usr/bin/env python
"""Analyse subspace-patch steering (write_subspace_patch.py) towards the ALTERNATIVE convention over all pool families.
Success = completion uses the alternative convention AND Gemini judge OK (unjudged records are excluded and counted).
Outputs results/code_styles/subspace_patch/: full.csv (family x layer x variant), pooled_by_layer.csv, per_family_best.csv, summary.json,
pooled_by_layer.png, README.md (written by hand afterwards)."""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/style_translation/qwen25_code/steering/subspace_patch_k4"
OUT = ROOT / "results/code_styles/subspace_patch"; OUT.mkdir(parents=True, exist_ok=True)
POOL = json.load(open(ROOT / "results/code_styles/code_pool.json"))["pool"]
VARIANTS = ["patch2", "patch1", "add2"]


def main():
    rows = []; missing = []; unjudged = 0
    for fam in POOL:
        fs = sorted(glob.glob(str(A / "*" / f"{fam}.json")))
        if not fs:
            missing.append(fam); continue
        for r in json.load(open(fs[0])):
            if r["target"] not in (None, "alt"):
                continue
            if r["judge"] is None:
                unjudged += 1; continue
            rows.append(dict(family=fam, layer=r["layer"], variant=r["variant"], doc_id=r["doc_id"], style_ok=bool(r["style_ok"]) if r["target"] else r["decision"] == "alt",
                             judge_ok=bool(r["judge"]["ok"]), unscorable=r["decision"] not in ("nat", "alt"), margin=r["margin_target"] if r["target"] else r["lp_alt"] - r["lp_nat"]))
    d = pd.DataFrame(rows); d["success"] = d.style_ok & d.judge_ok
    g = d.groupby(["family", "layer", "variant"]).agg(n=("success", "size"), success=("success", "mean"), convention_rate=("style_ok", "mean"),
                                                      judge_ok=("judge_ok", "mean"), unscorable=("unscorable", "mean"), margin=("margin", "mean")).reset_index()
    base = g[g.variant == "base"].set_index("family").success
    g["unsteered"] = g.family.map(base); g["gain"] = g.success - g.unsteered
    k4 = pd.read_csv(ROOT / "results/code_styles/write_steering/full56/full.csv").query("target == 'alt'").drop_duplicates("family").set_index("family")
    g["k4_alt"] = g.family.map(k4.k4); g.round(4).to_csv(OUT / "full.csv", index=False)
    st = g[g.variant != "base"]
    pooled = st.groupby(["layer", "variant"]).agg(success=("success", "mean"), convention_rate=("convention_rate", "mean"), judge_ok=("judge_ok", "mean"),
                                                   margin=("margin", "mean"), n_families=("family", "nunique")).reset_index()
    P = pooled.pivot(index="layer", columns="variant", values="success")[VARIANTS]; P.round(4).to_csv(OUT / "pooled_by_layer.csv")
    best_layer = {v: int(P[v].idxmax()) for v in VARIANTS}
    # per family: success of each variant at the pooled best layer of that variant, and at the family's own best layer
    pf = []
    for fam, gf in st.groupby("family"):
        row = dict(family=fam, unsteered=base[fam], k4_alt=k4.k4.get(fam, np.nan))
        for v in VARIANTS:
            s = gf[gf.variant == v].set_index("layer").success
            row[f"{v}_L{best_layer[v]}"] = s.get(best_layer[v], np.nan); row[f"{v}_own_best"] = s.max(); row[f"{v}_own_best_layer"] = int(s.idxmax())
        pf.append(row)
    pf = pd.DataFrame(pf).sort_values(f"patch2_L{best_layer['patch2']}", ascending=False); pf.round(4).to_csv(OUT / "per_family_best.csv", index=False)
    wins = {f"{a}_minus_{b}": {"at_pooled_best": float((pf[f"{a}_L{best_layer[a]}"] - pf[f"{b}_L{best_layer[b]}"]).mean()),
                               "own_best": float((pf[f"{a}_own_best"] - pf[f"{b}_own_best"]).mean()),
                               "families_a_better_by_.10_at_pooled_best": int(((pf[f"{a}_L{best_layer[a]}"] - pf[f"{b}_L{best_layer[b]}"]) >= .10).sum()),
                               "families_b_better_by_.10_at_pooled_best": int(((pf[f"{b}_L{best_layer[b]}"] - pf[f"{a}_L{best_layer[a]}"]) >= .10).sum())}
            for a, b in (("patch2", "patch1"), ("patch2", "add2"), ("add2", "patch1"))}
    summ = dict(n_families=int(st.family.nunique()), missing=missing, unjudged_records=unjudged, unsteered_alt=float(base.mean()), k4_alt=float(pf.k4_alt.mean()),
                best_layer=best_layer, pooled_at_best={v: float(P[v].max()) for v in VARIANTS}, pooled_by_layer={int(l): {v: float(P.loc[l, v]) for v in VARIANTS} for l in P.index},
                own_best_mean={v: float(pf[f"{v}_own_best"].mean()) for v in VARIANTS}, comparisons=wins,
                margin_at_best={v: float(pooled[(pooled.layer == best_layer[v]) & (pooled.variant == v)].margin.iloc[0]) for v in VARIANTS})
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for v, lab in zip(VARIANTS, ["patch PC1 + PC2 to k = 4 mean", "patch PC1 only", "add 2 × mean difference"]):
        ax.plot(P.index, P[v], marker="o", label=lab)
    ax.axhline(base.mean(), color="grey", ls="--", label=f"unsteered 0-shot ({base.mean():.2f})"); ax.axhline(pf.k4_alt.mean(), color="k", ls=":", label=f"k = 4 in context ({pf.k4_alt.mean():.2f})")
    ax.set_xlabel("layer (cue token)"); ax.set_ylabel("success → alternative convention"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title(f"Cue-token subspace patch vs additive steering\n{st.family.nunique()} code-convention families, 40 held-out 0-shot prompts each", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "pooled_by_layer.png", dpi=150)
    print(json.dumps({k: summ[k] for k in ("n_families", "missing", "unjudged_records", "unsteered_alt", "k4_alt", "best_layer", "pooled_at_best", "own_best_mean", "comparisons", "margin_at_best")}, indent=1))
    print(P.round(3).to_string())


if __name__ == "__main__":
    main()
