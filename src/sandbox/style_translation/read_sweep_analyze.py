#!/usr/bin/env python
"""Read-feature steering sweep analysis (CPU). Cells (family, target, layer, alpha) over the shard dirs of read_sweep.py:
success = the completion uses the TARGET convention (the style the demonstrations were NOT in) AND the judge OK. Baseline = the same
prompts unsteered (how often the model writes the target style against its context); ceiling = the step-3 k = 3 accuracy of the target
style. Selection: one shared (layer, alpha) maximising mean success over families and both directions (ties: smaller alpha, earlier layer).
Outputs -> <results>/read_steering/<tag>/: sweep.csv, baseline.csv, selected.json, sweep_heatmap.png, sweep_by_family.png, sweep_margin.png,
read_vs_write.png (best read cell vs best write cell per family, from write_steering/sweep10/sweep.csv)."""
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
from src.sandbox.style_translation.write_sweep_analyze import wilson


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--sweep", default="sweep_k3"); ap.add_argument("--tag", default="sweep10")
    args = ap.parse_args()
    MP = model_paths(args.model); SRC = MP["read_features"].parent / "read_steer" / args.sweep; R = MP["results"]; OUT = R / "read_steering" / args.tag; OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for f in sorted(SRC.glob("*/*.json")):
        recs += json.load(open(f))
    unj = sum(1 for r in recs if not r.get("judge")); assert unj == 0, f"{unj} unjudged records"
    fams = sorted({r["family"] for r in recs}); layers = sorted({r["layer"] for r in recs if r["target"]}); alphas = sorted({r["alpha"] for r in recs if r["target"]})
    k3 = {}
    for row in csv.DictReader(open(R / "summary.csv")):
        if int(row["k"]) == 3: k3[(row["family"], row["style"])] = float(row["accuracy"])
    cell = defaultdict(list)
    for r in recs:
        cell[(r["family"], r["target"], r["layer"], r["alpha"], r["context_style"])].append(r)
    base = {}
    with open(OUT / "baseline.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["family", "context", "target", "n", "target_and_ok", "unscorable", "judge_ok", "k3_accuracy_of_target"])
        for f in fams:
            for ctx, tgt in (("alt", "nat"), ("nat", "alt")):
                b = cell[(f, None, 0, 0.0, ctx)]; n = len(b)
                base[(f, tgt)] = dict(n=n, succ=float(np.mean([r["decision"] == tgt and r["judge"]["ok"] for r in b])) if n else 0.0, unsc=float(np.mean([r["decision"] is None for r in b])) if n else 0.0,
                                     jok=float(np.mean([r["judge"]["ok"] for r in b])) if n else 0.0, ceil=k3.get((f, tgt), float("nan")))
                w.writerow([f, ctx, tgt, n, round(base[(f, tgt)]["succ"], 3), round(base[(f, tgt)]["unsc"], 3), round(base[(f, tgt)]["jok"], 3), round(base[(f, tgt)]["ceil"], 3)])
    S = {}; rows = []
    for f in fams:
        for tgt, ctx in (("nat", "alt"), ("alt", "nat")):
            for L in layers:
                for a in alphas:
                    c = cell[(f, tgt, L, a, ctx)]; n = len(c)
                    suc = float(np.mean([r["decision"] == tgt and r["judge"]["ok"] for r in c])) if n else 0.0; lo, hi = wilson(suc, n)
                    S[(f, tgt, L, a)] = dict(n=n, success=suc, lo=lo, hi=hi, target_rate=float(np.mean([r["decision"] == tgt for r in c])) if n else 0.0,
                                             judge_ok=float(np.mean([r["judge"]["ok"] for r in c])) if n else 0.0, unscorable=float(np.mean([r["decision"] is None for r in c])) if n else 0.0,
                                             margin=float(np.mean([r["margin_target"] for r in c])) if n else 0.0, n_evidence=float(np.mean([r["n_evidence"] for r in c])) if n else 0.0,
                                             baseline=base[(f, tgt)]["succ"], ceiling=base[(f, tgt)]["ceil"])
                    rows.append(dict(family=f, target=tgt, context=ctx, layer=L, alpha=a, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in S[(f, tgt, L, a)].items()}))
    with open(OUT / "sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def grid(metric, targets=("nat", "alt")):
        return np.array([[np.mean([S[(f, t, L, a)][metric] for f in fams for t in targets]) for a in alphas] for L in layers])
    G = grid("success"); J = grid("judge_ok")
    order = sorted(((G[i, j], -alphas[j], -layers[i], layers[i], alphas[j]) for i in range(len(layers)) for j in range(len(alphas))), reverse=True)
    _, _, _, Lb, ab = order[0]; iL, ia = layers.index(Lb), alphas.index(ab)
    base_mean = float(np.mean([base[(f, t)]["succ"] for f in fams for t in ("nat", "alt")])); ceil_mean = float(np.nanmean([base[(f, t)]["ceil"] for f in fams for t in ("nat", "alt")]))
    sel = dict(layer=int(Lb), alpha=float(ab), mean_success=float(G[iL, ia]), mean_success_nat=float(grid("success", ("nat",))[iL, ia]), mean_success_alt=float(grid("success", ("alt",))[iL, ia]),
               mean_judge_ok=float(J[iL, ia]), unsteered_mean=base_mean, k3_ceiling_mean=ceil_mean, top5=[dict(layer=int(L), alpha=float(a), mean_success=float(g)) for g, _, _, L, a in order[:5]],
               per_family_at_selected={f: {t: round(S[(f, t, Lb, ab)]["success"], 3) for t in ("nat", "alt")} | {"base_nat": round(base[(f, "nat")]["succ"], 3), "base_alt": round(base[(f, "alt")]["succ"], 3), "ceil_nat": round(base[(f, "nat")]["ceil"], 3), "ceil_alt": round(base[(f, "alt")]["ceil"], 3)} for f in fams},
               mean_of_per_family_best=float(np.mean([max(S[(f, t, L, a)]["success"] for L in layers for a in alphas) for f in fams for t in ("nat", "alt")])))
    json.dump(sel, open(OUT / "selected.json", "w"), indent=1)
    # heatmaps
    fig, axes = plt.subplots(1, 4, figsize=(20, 6.5))
    for ax, (title, M) in zip(axes, [(f"success, both directions\n(unsteered {base_mean:.2f}, k = 3 right-style ceiling {ceil_mean:.2f})", G), (f"→ natural (alt-context prompts)\nunsteered {np.mean([base[(f,'nat')]['succ'] for f in fams]):.2f}", grid("success", ("nat",))),
                                     (f"→ alternative (nat-context prompts)\nunsteered {np.mean([base[(f,'alt')]['succ'] for f in fams]):.2f}", grid("success", ("alt",))), (f"judge OK (unsteered {np.mean([base[(f,t)]['jok'] for f in fams for t in ('nat','alt')]):.2f})", J)]):
        ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="lower")
        for i in range(len(layers)):
            for j in range(len(alphas)): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if M[i, j] < .55 else "k")
        ax.set_xticks(range(len(alphas))); ax.set_xticklabels([f"{a:g}" for a in alphas]); ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers); ax.set_xlabel("strength α (× read difference)"); ax.set_title(title, fontsize=10)
        ax.add_patch(plt.Rectangle((ia - .5, iL - .5), 1, 1, fill=False, ec="red", lw=2))
    axes[0].set_ylabel("layer (output of block L)")
    fig.suptitle(f"Read-feature steering at every evidence token of k = 3 wrong-style prompts, {len(fams)} families, ~{int(np.mean([S[(f,t,Lb,ab)]['n'] for f in fams for t in ('nat','alt')]))} held-out prompts per cell; red box = selected (layer {Lb}, α = {ab:g})", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "sweep_heatmap.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(4, 5, figsize=(22, 16))
    for i, (f, t) in enumerate([(f, t) for t in ("nat", "alt") for f in fams]):
        ax = axes.flat[i]; M = np.array([[S[(f, t, L, a)]["success"] for a in alphas] for L in layers]); ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="lower")
        for r_ in range(len(layers)):
            for c_ in range(len(alphas)): ax.text(c_, r_, f"{M[r_, c_]:.2f}", ha="center", va="center", fontsize=6.5, color="w" if M[r_, c_] < .55 else "k")
        ax.set_xticks(range(len(alphas))); ax.set_xticklabels([f"{a:g}" for a in alphas], fontsize=7); ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers, fontsize=7)
        ax.set_title(f"{f} → {'natural' if t == 'nat' else 'alternative'} (unsteered {base[(f, t)]['succ']:.2f}, ceiling {base[(f, t)]['ceil']:.2f})", fontsize=8); ax.add_patch(plt.Rectangle((ia - .5, iL - .5), 1, 1, fill=False, ec="red", lw=1.5))
    fig.suptitle("Per-family success (target convention AND judge OK); red box = the selected shared setting", fontsize=12); fig.tight_layout(); fig.savefig(OUT / "sweep_by_family.png", dpi=110); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, t in zip(axes, ("nat", "alt")):
        for L in layers: ax.plot(alphas, [np.mean([S[(f, t, L, a)]["margin"] for f in fams]) for a in alphas], marker="o", label=f"L{L}", color=plt.cm.viridis(layers.index(L) / len(layers)))
        ax.axhline(np.mean([np.mean([r["margin_target"] for r in cell[(f, None, 0, 0.0, 'alt' if t == 'nat' else 'nat')]]) for f in fams]), color="gray", ls="--", label="unsteered"); ax.axhline(0, color="k", lw=.6)
        ax.set_xscale("log", base=2); ax.set_xticks(alphas); ax.set_xticklabels([f"{a:g}" for a in alphas]); ax.set_xlabel("strength α"); ax.set_title(f"steering towards the {'natural' if t == 'nat' else 'alternative'} convention"); ax.grid(alpha=.3)
    axes[0].set_ylabel("first-token margin: lp(target) − lp(context style), nats"); axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle("Classifier-free check: steered first-token margin at the query cue, mean over families"); fig.tight_layout(); fig.savefig(OUT / "sweep_margin.png", dpi=130); plt.close(fig)
    # read vs write: best cell per family and direction
    wp = R / "write_steering" / "sweep10" / "sweep.csv"
    if wp.exists():
        W = defaultdict(float)
        for row in csv.DictReader(open(wp)): W[(row["family"], row["target"])] = max(W[(row["family"], row["target"])], float(row["success"]))
        fig, ax = plt.subplots(figsize=(7, 6))
        for t, c in (("nat", "#1f4f7a"), ("alt", "#8a2b23")):
            xs = [W[(f, t)] for f in fams]; ys = [max(S[(f, t, L, a)]["success"] for L in layers for a in alphas) for f in fams]
            ax.scatter(xs, ys, color=c, label=f"→ {'natural' if t == 'nat' else 'alternative'}")
            for f, x_, y_ in zip(fams, xs, ys): ax.annotate(f, (x_, y_), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.plot([0, 1], [0, 1], "k--", lw=.6); ax.set_xlabel("best write-steering success (0-shot, cue token)"); ax.set_ylabel("best read-steering success (k = 3 wrong-style, evidence tokens)"); ax.legend(); ax.grid(alpha=.3)
        ax.set_title("Read vs write steering, best cell per family and direction"); fig.tight_layout(); fig.savefig(OUT / "read_vs_write.png", dpi=130); plt.close(fig)
    print(json.dumps({k: v for k, v in sel.items() if k not in ("per_family_at_selected",)}, indent=1)); print("->", OUT)


if __name__ == "__main__":
    main()
