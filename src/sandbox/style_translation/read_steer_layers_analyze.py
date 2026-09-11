#!/usr/bin/env python
"""Step 7d — full-layer view of evidence-token steering: merge the 9-layer screen with the extra-layer
screen (layers 0..27 covered) and plot the target-style rate vs layer for every family, direction and
alpha, plus the mean over cells per group. Outputs -> results/style_translation/read_steer/:
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
from src.sandbox.style_translation.steer_screen import ALPHAS
from src.sandbox.style_translation.read_steer_screen import DIRECTIONS
from src.sandbox.style_translation.family_groups import grouped_grid, grouped_order, GROUPS

ROOT = ARTIFACTS_ROOT / "style_translation" / "read_steer"
OUT = STYLE_TRANSLATION_RESULTS / "read_steer"
C = {"nat2alt": "#d62728", "alt2nat": "#1f77b4"}


def main():
    fams = [f.name for f in FAMILIES if (ROOT / "screen" / f"{f.name}.json").exists() and (ROOT / "screen_extra" / f"{f.name}.json").exists()]
    grid, base = {}, {}
    for fam in fams:
        recs = json.load(open(ROOT / "screen" / f"{fam}.json")) + json.load(open(ROOT / "screen_extra" / f"{fam}.json"))
        for d, (ctx, target) in DIRECTIONS.items():
            b = [r for r in recs if r["alpha"] == 0.0 and r["context"] == ctx]
            base[(fam, d)] = float(np.mean([r["decision"] == target for r in b]))
            by = collections.defaultdict(list)
            for r in recs:
                if r["direction"] == d:
                    by[(r["layer"], r["alpha"])].append(r["decision"] == target)
            for k, v in by.items():
                grid[(fam, d) + k] = float(np.mean(v))
    layers = sorted({k[2] for k in grid})
    assert layers == list(range(0, 28)), layers
    with open(OUT / "screen_full.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["family", "direction", "layer", "alpha", "target_rate", "unsteered"])
        for (fam, d, L, a), v in sorted(grid.items()):
            w.writerow([fam, d, L, a, round(v, 3), round(base[(fam, d)], 3)])
    # best layer per cell over the full sweep vs over the original 9
    rows = []
    for fam in fams:
        for d in DIRECTIONS:
            cells = {(L, a): grid[(fam, d, L, a)] for L in layers for a in ALPHAS}
            (Lb, ab), vb = max(cells.items(), key=lambda kv: (kv[1], -kv[0][1], -kv[0][0]))
            nine = {k: v for k, v in cells.items() if k[0] in (2, 4, 6, 8, 10, 12, 16, 20, 24)}
            (L9, a9), v9 = max(nine.items(), key=lambda kv: (kv[1], -kv[0][1], -kv[0][0]))
            l0 = max(cells[(0, a)] for a in ALPHAS)
            rows.append(dict(family=fam, direction=d, unsteered=round(base[(fam, d)], 3), best_layer=Lb, best_alpha=ab, best_rate=round(vb, 3),
                             best_layer_9=L9, best_alpha_9=a9, best_rate_9=round(v9, 3), rate_L0=round(l0, 3), gain_full_vs_9=round(vb - v9, 3)))
    with open(OUT / "best_layer_full.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # per-cell curves, all 28 layers
    fig, gax = grouped_grid(fams, slots_per_family=2, fam_cols=(2, 2), panel_w=3.1, panel_h=2.3, top=0.915, bottom=0.04, sharex=True, sharey=True)
    acol = dict(zip(ALPHAS, plt.cm.viridis(np.linspace(0.15, 0.9, len(ALPHAS)))))
    for fam in grouped_order(fams):
        for slot, d in enumerate(DIRECTIONS):
            ax = gax[(fam, slot)]
            ax.axhline(base[(fam, d)], color="#9e9e9e", linestyle="dashed", lw=1.2, label="α = 0 (unsteered)")
            for a in ALPHAS:
                ax.plot(layers, [grid[(fam, d, L, a)] for L in layers], marker="o", ms=2.2, lw=1.2, color=acol[a], label=f"α = {a:g}")
            ax.set_title(f"{fam}: {DIRECTIONS[d][0]} ctx → {DIRECTIONS[d][1]}", fontsize=8, color=C[d])
            ax.set_ylim(-0.03, 1.03); ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 27]); ax.tick_params(labelsize=6.5); ax.grid(alpha=0.3)
    for ax in fig.axes_meta["left"]:
        ax.set_ylabel("target rate", fontsize=8)
    for ax in fig.axes_meta["bottom"]:
        ax.set_xlabel("injection layer (0 = embeddings)", fontsize=8)
    h, l = next(iter(gax.values())).get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(ALPHAS) + 1, fontsize=9, frameon=False)
    fig.suptitle("Evidence-token steering, every layer 0–27: rate of the target convention at the 4th decision vs injection layer, one line per α\n"
                 "style only, 50 texts per point, 16-token completions; dashed = unsteered 3-shot rate", fontsize=11, y=0.995)
    fig.savefig(OUT / "read_steer_layer_alpha_full.png", dpi=150); plt.close(fig)

    # mean over cells per group, per alpha
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, group, tint) in zip(axes, GROUPS):
        g = [f for f in group if f in fams]; ax.set_facecolor(tint)
        for a in ALPHAS:
            m = [np.mean([grid[(fam, d, L, a)] for fam in g for d in DIRECTIONS]) for L in layers]
            ax.plot(layers, m, marker="o", ms=3, lw=1.6, color=acol[a], label=f"α = {a:g}")
        ax.axhline(np.mean([base[(fam, d)] for fam in g for d in DIRECTIONS]), color="#9e9e9e", linestyle="dashed", lw=1.2, label="unsteered")
        ax.set_title(f"{title.split(' — ')[0]} ({len(g)} families × 2 directions)", fontsize=10); ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 27]); ax.set_xlabel("injection layer (0 = embeddings)"); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
    axes[0].set_ylabel("mean target-style rate"); axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Evidence-token steering: mean target-style rate over families and directions vs injection layer (screen, 50 texts per cell)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "read_steer_layer_mean.png", dpi=150); plt.close(fig)

    print(f"{'family':14s} {'dir':8s} {'unst':>5s} {'L0':>5s} {'best9 (L,α)':>13s} {'bestAll (L,α)':>15s} {'gain':>5s}")
    for r in rows:
        print(f"{r['family']:14s} {r['direction']:8s} {r['unsteered']:5.2f} {r['rate_L0']:5.2f} {r['best_rate_9']:5.2f} (L{r['best_layer_9']:<2d},{r['best_alpha_9']:g}) {r['best_rate']:6.2f} (L{r['best_layer']:<2d},{r['best_alpha']:g}) {r['gain_full_vs_9']:+5.2f}")
    print(f"mean rate: L0 {np.mean([r['rate_L0'] for r in rows]):.2f} | best of 9 {np.mean([r['best_rate_9'] for r in rows]):.2f} | best of 28 {np.mean([r['best_rate'] for r in rows]):.2f}")
    print("->", OUT)


if __name__ == "__main__":
    main()
