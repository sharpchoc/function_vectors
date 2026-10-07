#!/usr/bin/env python
"""Analyse the own-counterpart subspace patch (write_subspace_patch_own.py) next to the mean-target patch (write_subspace_patch.py), alternative
direction, all pool families. Success = alternative convention AND judge OK. Outputs results/code_styles/subspace_patch_own/:
full.csv, pooled_by_layer.csv, per_family_L28.csv, summary.json, pooled_by_layer.png."""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "artifacts/style_translation/qwen25_code/steering"
OUT = ROOT / "results/code_styles/subspace_patch_own"; OUT.mkdir(parents=True, exist_ok=True)
POOL = json.load(open(ROOT / "results/code_styles/code_pool.json"))["pool"]
ARMS = ["mean1", "mean2", "own1", "own2", "own10", "full"]
LABELS = {"mean1": "PC1, training mean", "mean2": "PC1 + PC2, training mean", "own1": "PC1, own counterpart", "own2": "PC1 + PC2, own counterpart",
          "own10": "top 10 PCs, own counterpart", "full": "full activation, own counterpart"}


def load(sub, rename):
    rows = []; unj = 0
    for fam in POOL:
        fs = sorted(glob.glob(str(A / sub / "*" / f"{fam}.json")))
        if not fs:
            continue
        for r in json.load(open(fs[0])):
            if r["target"] not in (None, "alt") or r["variant"] not in rename:
                continue
            if r["judge"] is None:
                unj += 1; continue
            rows.append(dict(family=fam, layer=r["layer"], variant=rename[r["variant"]], doc_id=r["doc_id"], style_ok=bool(r["style_ok"]) if r["target"] else r["decision"] == "alt",
                             judge_ok=bool(r["judge"]["ok"]), unscorable=r["decision"] not in ("nat", "alt"), margin=r["margin_target"] if r["target"] else r["lp_alt"] - r["lp_nat"]))
    return pd.DataFrame(rows), unj


def main():
    d_mean, u1 = load("subspace_patch_k4", {"patch1": "mean1", "patch2": "mean2"})
    d_own, u2 = load("subspace_patch_own_k4", {"own1": "own1", "own2": "own2", "own10": "own10", "full": "full", "base": "base"})
    d = pd.concat([d_mean, d_own]); d["success"] = d.style_ok & d.judge_ok
    fams_own = set(d_own.family); d = d[d.family.isin(fams_own)]
    g = d.groupby(["family", "layer", "variant"]).agg(n=("success", "size"), success=("success", "mean"), convention_rate=("style_ok", "mean"), judge_ok=("judge_ok", "mean"),
                                                      unscorable=("unscorable", "mean"), margin=("margin", "mean")).reset_index()
    base = g[g.variant == "base"].set_index("family").success; g["unsteered"] = g.family.map(base); g.round(4).to_csv(OUT / "full.csv", index=False)
    st = g[g.variant != "base"]
    P = st.groupby(["layer", "variant"]).success.mean().unstack()[ARMS]; P.round(4).to_csv(OUT / "pooled_by_layer.csv")
    M = st.groupby(["layer", "variant"]).margin.mean().unstack()[ARMS]
    best = {v: int(P[v].idxmax()) for v in ARMS}
    pf = st[st.layer == 28].pivot(index="family", columns="variant", values="success")[ARMS]; pf.insert(0, "unsteered", base); pf = pf.sort_values("own2", ascending=False)
    pf.round(4).to_csv(OUT / "per_family_L28.csv")
    k4 = pd.read_csv(ROOT / "results/code_styles/write_steering/full56/full.csv").query("target == 'alt'").drop_duplicates("family").set_index("family").k4
    summ = dict(n_families=int(st.family.nunique()), unjudged=u1 + u2, unsteered=float(base.mean()), k4_alt=float(k4.reindex(pf.index).mean()), best_layer=best,
                pooled_by_layer={int(l): {v: float(P.loc[l, v]) for v in ARMS} for l in P.index}, pooled_L28={v: float(P.loc[28, v]) for v in ARMS},
                margin_L28={v: float(M.loc[28, v]) for v in ARMS}, judge_ok_L28={v: float(st[(st.layer == 28) & (st.variant == v)].judge_ok.mean()) for v in ARMS},
                deltas_L28={"own2_minus_mean2": float((pf.own2 - pf.mean2).mean()), "own2_minus_own1": float((pf.own2 - pf.own1).mean()), "own10_minus_own2": float((pf.own10 - pf.own2).mean()),
                            "full_minus_own10": float((pf.full - pf.own10).mean()), "own1_minus_mean1": float((pf.own1 - pf.mean1).mean())},
                families_own2_better_mean2_by_10=int(((pf.own2 - pf.mean2) >= .1).sum()), families_mean2_better_own2_by_10=int(((pf.mean2 - pf.own2) >= .1).sum()))
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    cols = {"mean1": "#d9a441", "mean2": "#c2541b", "own1": "#7fb3c8", "own2": "#1f6f8b", "own10": "#3b3f8c", "full": "#222222"}
    ls = {"mean1": "--", "mean2": "--", "own1": "-", "own2": "-", "own10": "-", "full": "-"}
    for v in ARMS:
        ax.plot(P.index, P[v], marker="o", ms=6, lw=2.2, color=cols[v], ls=ls[v], label=LABELS[v])
    ax.axhline(summ["k4_alt"], color="black", ls=":", lw=1.5, label=f"k = 4 in context ({summ['k4_alt']:.2f})"); ax.axhline(summ["unsteered"], color="grey", ls="--", lw=1.5, label=f"unsteered 0-shot ({summ['unsteered']:.2f})")
    ax.set_xlabel("layer of the cue-token patch"); ax.set_ylabel("success → alternative convention"); ax.set_ylim(0, 1); ax.set_xticks(range(20, 29))
    ax.set_title(f"Cue-token patch: training mean vs the document's own k = 4 counterpart, {summ['n_families']} families"); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.3)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2); fig.tight_layout(); fig.savefig(OUT / "pooled_by_layer.png", facecolor="white")
    print(json.dumps({k: summ[k] for k in ("n_families", "unjudged", "unsteered", "k4_alt", "best_layer", "pooled_L28", "margin_L28", "judge_ok_L28", "deltas_L28", "families_own2_better_mean2_by_10", "families_mean2_better_own2_by_10")}, indent=1))
    print(P.round(3).to_string())


if __name__ == "__main__":
    main()
