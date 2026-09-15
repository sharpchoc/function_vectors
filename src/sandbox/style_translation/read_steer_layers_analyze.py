#!/usr/bin/env python
"""Step 7d — full-layer view of evidence-token steering: the screen over layers 0..27 (one screen dir, or the
9-layer screen merged with screen_extra as in the text study) → target-style rate vs layer for every family,
direction and alpha, plus the mean over cells per family group. Outputs -> the read_steer bucket of the run:
read_steer_layer_alpha_full.png, read_steer_layer_mean.png, screen_full.csv, best_layer_full.csv.
"""
import collections
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, STYLE_TRANSLATION_RESULTS
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.ml_families import ML_FAMILIES, ML_FAMILY
from src.sandbox.style_translation.steer_screen import ALPHAS
from src.sandbox.style_translation.read_steer_screen import DIRECTIONS, K_CTX
from src.sandbox.style_translation.family_groups import grouped_grid, grouped_order, GROUPS

ROOT = ARTIFACTS_ROOT / "style_translation" / "read_steer"
OUT = STYLE_TRANSLATION_RESULTS / "read_steer"


def configure(model="gptj", tag=None, k_ctx=K_CTX):
    global ROOT, OUT
    MP = model_paths(model); ROOT = MP["read_steer"]
    R = MP["results"] / tag if tag else MP["results"]
    OUT = R / ("read_steer" if k_ctx == K_CTX else f"read_steer_k{k_ctx}")


C = {"nat2alt": "#d62728", "alt2nat": "#1f77b4"}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gptj", help="models.MODELS key")
    ap.add_argument("--tag", default=None); ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--k_ctx", type=int, default=K_CTX)
    args = ap.parse_args()
    k = args.k_ctx; configure(args.model, args.tag, k); OUT.mkdir(parents=True, exist_ok=True)
    SCREEN = ROOT / ("screen" if k == K_CTX else f"screen_k{k}"); EXTRA = ROOT / ("screen_extra" if k == K_CTX else f"screen_extra_k{k}")
    universe = [f.name for f in list(FAMILIES) + list(ML_FAMILIES) if args.families is None or f.name in args.families]
    fams = [f for f in universe if (SCREEN / f"{f}.json").exists()]
    grid, base = {}, {}
    for fam in fams:
        recs = json.load(open(SCREEN / f"{fam}.json")) + (json.load(open(EXTRA / f"{fam}.json")) if (EXTRA / f"{fam}.json").exists() else [])
        for d, (ctx, target) in DIRECTIONS.items():
            b = [r for r in recs if r["alpha"] == 0.0 and r["context"] == ctx]
            base[(fam, d)] = float(np.mean([r["decision"] == target for r in b]))
            by = collections.defaultdict(list)
            for r in recs:
                if r["direction"] == d:
                    by[(r["layer"], r["alpha"])].append(r["decision"] == target)
            for kk, v in by.items():
                grid[(fam, d) + kk] = float(np.mean(v))
    layers = sorted({kk[2] for kk in grid})
    fams = [f for f in fams if all((f, d, L, a) in grid for d in DIRECTIONS for L in layers for a in ALPHAS)]   # complete sweeps only
    if layers != list(range(0, len(layers))):
        print(f"note: layers screened = {layers} (not the full 0..n-1 sweep)")
    with open(OUT / "screen_full.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["family", "direction", "layer", "alpha", "target_rate", "unsteered"])
        for (fam, d, L, a), v in sorted(grid.items()):
            if fam in fams:
                w.writerow([fam, d, L, a, round(v, 3), round(base[(fam, d)], 3)])
    nine = [L for L in (2, 4, 6, 8, 10, 12, 16, 20, 24) if L in layers]
    rows = []
    for fam in fams:
        for d in DIRECTIONS:
            cells = {(L, a): grid[(fam, d, L, a)] for L in layers for a in ALPHAS}
            (Lb, ab), vb = max(cells.items(), key=lambda kv: (kv[1], -kv[0][1], -kv[0][0]))
            sub = {kk: v for kk, v in cells.items() if kk[0] in nine} or cells
            (L9, a9), v9 = max(sub.items(), key=lambda kv: (kv[1], -kv[0][1], -kv[0][0]))
            l0 = max(cells[(0, a)] for a in ALPHAS) if 0 in layers else float("nan")
            rows.append(dict(family=fam, direction=d, unsteered=round(base[(fam, d)], 3), best_layer=Lb, best_alpha=ab, best_rate=round(vb, 3),
                             best_layer_9=L9, best_alpha_9=a9, best_rate_9=round(v9, 3), rate_L0=round(l0, 3), gain_full_vs_9=round(vb - v9, 3)))
    with open(OUT / "best_layer_full.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    ticks = [L for L in (0, 4, 8, 12, 16, 20, 24, 27) if L in layers]
    fig, gax = grouped_grid(fams, slots_per_family=2, fam_cols=(2, 2), panel_w=3.1, panel_h=2.3, top=0.915, bottom=0.04, sharex=True, sharey=True)
    acol = dict(zip(ALPHAS, plt.cm.viridis(np.linspace(0.15, 0.9, len(ALPHAS)))))
    for fam in grouped_order(fams):
        for slot, d in enumerate(DIRECTIONS):
            ax = gax[(fam, slot)]
            ax.axhline(base[(fam, d)], color="#9e9e9e", linestyle="dashed", lw=1.2, label="α = 0 (unsteered)")
            for a in ALPHAS:
                ax.plot(layers, [grid[(fam, d, L, a)] for L in layers], marker="o", ms=2.2, lw=1.2, color=acol[a], label=f"α = {a:g}")
            ax.set_title(f"{fam}: {DIRECTIONS[d][0]} ctx → {DIRECTIONS[d][1]}", fontsize=8, color=C[d])
            ax.set_ylim(-0.03, 1.03); ax.set_xticks(ticks); ax.tick_params(labelsize=6.5); ax.grid(alpha=0.3)
    for ax in fig.axes_meta["left"]:
        ax.set_ylabel("target rate", fontsize=8)
    for ax in fig.axes_meta["bottom"]:
        ax.set_xlabel("injection layer (0 = embeddings)", fontsize=8)
    h, l = next(iter(gax.values())).get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(ALPHAS) + 1, fontsize=9, frameon=False)
    fig.suptitle(f"Evidence-token steering of a {k}-shot prompt, every layer {layers[0]}–{layers[-1]}: rate of the target convention at the next decision vs injection layer, one line per α\n"
                 f"style only, 50 prompts per point, 16-token completions; dashed = unsteered {k}-shot rate", fontsize=11, y=0.995)
    fig.savefig(OUT / "read_steer_layer_alpha_full.png", dpi=150); plt.close(fig)

    groups = [(t, [f for f in g if f in fams], tint) for t, g, tint in GROUPS]; groups = [x for x in groups if x[1]]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 4), sharey=True, squeeze=False); axes = axes[0]
    for ax, (title, g, tint) in zip(axes, groups):
        ax.set_facecolor(tint)
        for a in ALPHAS:
            m = [np.mean([grid[(fam, d, L, a)] for fam in g for d in DIRECTIONS]) for L in layers]
            ax.plot(layers, m, marker="o", ms=3, lw=1.6, color=acol[a], label=f"α = {a:g}")
        ax.axhline(np.mean([base[(fam, d)] for fam in g for d in DIRECTIONS]), color="#9e9e9e", linestyle="dashed", lw=1.2, label="unsteered")
        ax.set_title(f"{title.split(' — ')[0]} ({len(g)} families × 2 directions)", fontsize=10); ax.set_xticks(ticks); ax.set_xlabel("injection layer (0 = embeddings)"); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
    axes[0].set_ylabel("mean target-style rate"); axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Evidence-token steering of a {k}-shot prompt: mean target-style rate over families and directions\nvs injection layer (screen, 50 prompts per cell)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "read_steer_layer_mean.png", dpi=150); plt.close(fig)

    # mean over families per DIRECTION, with the unsteered k-shot rate and the genuine k-shot reference (step-3 style rate at the same k)
    step3 = {}
    STEP3 = OUT.parent / "summary.csv"
    if STEP3.exists():
        step3 = {(r["family"], r["style"], int(r["k"])): float(r["style_ok"]) for r in csv.DictReader(open(STEP3))}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, d in zip(axes, DIRECTIONS):
        ctx, target = DIRECTIONS[d]
        for a in ALPHAS:
            m = [np.mean([grid[(fam, d, L, a)] for fam in fams]) for L in layers]
            ax.plot(layers, m, marker="o", ms=3, lw=1.6, color=acol[a], label=f"α = {a:g}")
        ax.axhline(np.mean([base[(fam, d)] for fam in fams]), color="#9e9e9e", linestyle="dashed", lw=1.3, label=f"unsteered {k}-shot prompt ({ctx} context)")
        ref = [step3[(fam, target, k)] for fam in fams if (fam, target, k) in step3]
        if ref:
            ax.axhline(np.mean(ref), color="black", linestyle="dashed", lw=1.3, label=f"genuine {k}-shot prompt ({target} context), step 3")
        ax.set_title(f"{ctx} context → steered toward {target}  ({len(fams)} families)", fontsize=10, color=C[d])
        ax.set_xticks(ticks); ax.set_xlabel("injection layer (0 = embeddings)"); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="center left", frameon=False)
    axes[0].set_ylabel("target-convention rate (style only, mean over families)", fontsize=9)
    fig.suptitle(f"Evidence-token steering of a {k}-shot prompt, by direction: mean target-style rate vs injection layer (screen, 50 prompts per cell)\n"
                 f"dashed grey = same prompt unsteered; dashed black = a real {k}-shot prompt whose context shows the target (style rate, step 3, same k)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "read_steer_layer_mean_by_direction.png", dpi=150); plt.close(fig)

    print(f"{'family':18s} {'dir':8s} {'unst':>5s} {'L0':>5s} {'best9 (L,α)':>13s} {'bestAll (L,α)':>15s} {'gain':>5s}")
    for r in rows:
        print(f"{r['family']:18s} {r['direction']:8s} {r['unsteered']:5.2f} {r['rate_L0']:5.2f} {r['best_rate_9']:5.2f} (L{r['best_layer_9']:<2d},{r['best_alpha_9']:g}) {r['best_rate']:6.2f} (L{r['best_layer']:<2d},{r['best_alpha']:g}) {r['gain_full_vs_9']:+5.2f}")
    print(f"mean rate: L0 {np.nanmean([r['rate_L0'] for r in rows]):.2f} | best of 9 {np.mean([r['best_rate_9'] for r in rows]):.2f} | best of all {np.mean([r['best_rate'] for r in rows]):.2f}")
    print("->", OUT)


if __name__ == "__main__":
    main()
