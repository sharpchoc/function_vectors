#!/usr/bin/env python
"""Write-feature steering sweep — hyperparameter selection (CPU). Merges the shard directories of write_sweep.py and scores every cell
(family, target, layer, alpha): success = the completion uses the TARGET convention AND the judge finds it a sensible continuation.
Selection: ONE shared (layer, alpha) = the cell with the highest mean success over families and both directions (ties: smaller alpha,
then earlier layer). Outputs -> <results>/write_steering/<tag>/: sweep.csv, baseline.csv, selected.json, sweep_heatmap.png,
sweep_by_family.png, sweep_margin.png."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths


def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--sweep", default="sweep_k3"); ap.add_argument("--tag", default="sweep10")
    args = ap.parse_args()
    MP = model_paths(args.model); SRC = MP["steering"] / args.sweep; OUT = MP["results"] / "write_steering" / args.tag; OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for f in sorted(SRC.glob("*/*.json")):
        recs += json.load(open(f))
    unj = sum(1 for r in recs if not r.get("judge")); assert unj == 0, f"{unj} unjudged records"
    fams = sorted({r["family"] for r in recs}); layers = sorted({r["layer"] for r in recs if r["target"]}); alphas = sorted({r["alpha"] for r in recs if r["target"]})
    cell = defaultdict(list)
    for r in recs:
        cell[(r["family"], r["target"], r["layer"], r["alpha"])].append(r)
    # unsteered baseline: how often the model already writes each convention (AND judge OK) on the same held-out prompts
    base = {}
    with open(OUT / "baseline.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["family", "n", "nat_and_ok", "alt_and_ok", "unscorable", "judge_ok", "margin_nat_minus_alt"])
        for f in fams:
            b = cell[(f, None, 0, 0.0)]; n = len(b)
            base[f] = {"nat": np.mean([r["decision"] == "nat" and r["judge"]["ok"] for r in b]), "alt": np.mean([r["decision"] == "alt" and r["judge"]["ok"] for r in b]),
                       "unsc": np.mean([r["decision"] is None for r in b]), "jok": np.mean([r["judge"]["ok"] for r in b]), "n": n}
            w.writerow([f, n, round(base[f]["nat"], 3), round(base[f]["alt"], 3), round(base[f]["unsc"], 3), round(base[f]["jok"], 3), round(float(np.mean([r["lp_nat"] - r["lp_alt"] for r in b])), 3)])
    S = {}; rows = []
    for f in fams:
        for t in ("nat", "alt"):
            for L in layers:
                for a in alphas:
                    c = cell[(f, t, L, a)]; n = len(c)
                    suc = np.mean([r["decision"] == t and r["judge"]["ok"] for r in c]); lo, hi = wilson(suc, n)
                    S[(f, t, L, a)] = dict(n=n, success=float(suc), lo=lo, hi=hi, target_rate=float(np.mean([r["decision"] == t for r in c])), judge_ok=float(np.mean([r["judge"]["ok"] for r in c])),
                                           unscorable=float(np.mean([r["decision"] is None for r in c])), margin=float(np.mean([r["margin_target"] for r in c])), top1=float(np.mean([r["top1"] == t for r in c])),
                                           baseline=float(base[f][t]))
                    rows.append(dict(family=f, target=t, layer=L, alpha=a, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in S[(f, t, L, a)].items()}))
    with open(OUT / "sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def grid(metric, targets=("nat", "alt")):
        return np.array([[np.mean([S[(f, t, L, a)][metric] for f in fams for t in targets]) for a in alphas] for L in layers])
    G = grid("success"); J = grid("judge_ok")
    order = sorted(((G[i, j], -alphas[j], -layers[i], layers[i], alphas[j]) for i in range(len(layers)) for j in range(len(alphas))), reverse=True)
    _, _, _, Lb, ab = order[0]
    base_mean = float(np.mean([base[f][t] for f in fams for t in ("nat", "alt")]))
    per_family = {f: {t: max(((S[(f, t, L, a)]["success"], -a, -L, L, a) for L in layers for a in alphas))[3:] for t in ("nat", "alt")} for f in fams}
    sel = dict(layer=int(Lb), alpha=float(ab), mean_success=float(G[layers.index(Lb), alphas.index(ab)]), mean_success_nat=float(grid("success", ("nat",))[layers.index(Lb), alphas.index(ab)]),
               mean_success_alt=float(grid("success", ("alt",))[layers.index(Lb), alphas.index(ab)]), mean_judge_ok=float(J[layers.index(Lb), alphas.index(ab)]), unsteered_mean=base_mean,
               unsteered_judge_ok=float(np.mean([base[f]["jok"] for f in fams])), top5=[dict(layer=int(L), alpha=float(a), mean_success=float(g)) for g, _, _, L, a in order[:5]],
               per_family_at_selected={f: {t: round(S[(f, t, Lb, ab)]["success"], 3) for t in ("nat", "alt")} | {"base_nat": round(float(base[f]["nat"]), 3), "base_alt": round(float(base[f]["alt"]), 3)} for f in fams},
               per_family_best={f: {t: dict(layer=int(per_family[f][t][0]), alpha=float(per_family[f][t][1]), success=round(S[(f, t, per_family[f][t][0], per_family[f][t][1])]["success"], 3)) for t in ("nat", "alt")} for f in fams},
               mean_of_per_family_best=float(np.mean([S[(f, t, per_family[f][t][0], per_family[f][t][1])]["success"] for f in fams for t in ("nat", "alt")])), families=fams, n_per_cell=int(np.mean([v["n"] for v in S.values()])))
    json.dump(sel, open(OUT / "selected.json", "w"), indent=1)

    def heat(ax, M, title, vmin=0, vmax=1, cmap="viridis", mark=True):
        im = ax.imshow(M, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(len(alphas))); ax.set_xticklabels([f"{a:g}" for a in alphas]); ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers)
        for i in range(len(layers)):
            for j in range(len(alphas)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] < (vmin + vmax) / 2 else "black")
        if mark:
            ax.add_patch(plt.Rectangle((alphas.index(ab) - .5, layers.index(Lb) - .5), 1, 1, fill=False, ec="red", lw=2))
        ax.set_title(title, fontsize=10); ax.set_xlabel("strength α (× mean difference)"); return im
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.6))
    heat(axes[0], G, f"success, both directions\n(unsteered {base_mean:.2f})"); heat(axes[1], grid("success", ("nat",)), f"→ natural (unsteered {np.mean([base[f]['nat'] for f in fams]):.2f})")
    heat(axes[2], grid("success", ("alt",)), f"→ alternative (unsteered {np.mean([base[f]['alt'] for f in fams]):.2f})"); heat(axes[3], J, f"judge OK (unsteered {sel['unsteered_judge_ok']:.2f})")
    axes[0].set_ylabel("layer (output of block L)")
    fig.suptitle(f"{MP['label']}: cue-token steering of 0-shot prompts with α · (μ_nat − μ_alt), mean over {len(fams)} families, ~{sel['n_per_cell']} held-out prompts per cell\n"
                 f"success = target convention AND judge OK; red box = selected shared setting (layer {Lb}, α = {ab:g})", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "sweep_heatmap.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(19, 15))
    for r_, t in enumerate(("nat", "alt")):
        for c_, f in enumerate(fams):
            ax = axes[r_ * 2 + c_ // 5][c_ % 5]
            M = np.array([[S[(f, t, L, a)]["success"] for a in alphas] for L in layers]); heat(ax, M, f"{f} → {'natural' if t == 'nat' else 'alternative'} (unsteered {base[f][t]:.2f})")
            if c_ % 5 == 0: ax.set_ylabel("layer")
    fig.suptitle("Per-family success (target convention AND judge OK); red box = the selected shared setting", fontsize=12); fig.tight_layout(); fig.savefig(OUT / "sweep_by_family.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True); cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(layers)))
    for ax, t in zip(axes, ("nat", "alt")):
        for L, col in zip(layers, cmap):
            ax.plot(alphas, [np.mean([S[(f, t, L, a)]["margin"] for f in fams]) for a in alphas], marker="o", ms=3, color=col, label=f"L{L}")
        b = np.mean([(1 if t == "nat" else -1) * np.mean([r["lp_nat"] - r["lp_alt"] for r in cell[(f, None, 0, 0.0)]]) for f in fams])
        ax.axhline(b, color="#9e9e9e", ls="dashed", label="unsteered"); ax.axhline(0, color="black", lw=0.6); ax.set_xscale("log", base=2); ax.set_xticks(alphas); ax.set_xticklabels([f"{a:g}" for a in alphas])
        ax.set_title(f"steering towards the {'natural' if t == 'nat' else 'alternative'} convention"); ax.set_xlabel("strength α"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("first-token margin: lp(target) − lp(other), nats"); axes[1].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Classifier-free check: steered first-token log-prob margin, mean over families", fontsize=11); fig.tight_layout(); fig.savefig(OUT / "sweep_margin.png", dpi=150); plt.close(fig)
    print(json.dumps({k: sel[k] for k in ("layer", "alpha", "mean_success", "mean_success_nat", "mean_success_alt", "mean_judge_ok", "unsteered_mean", "unsteered_judge_ok", "mean_of_per_family_best", "top5")}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
