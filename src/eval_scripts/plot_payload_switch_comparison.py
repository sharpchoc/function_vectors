#!/usr/bin/env python
"""Comparison: payload-subspace REPLACEMENT steering vs the old paired-Delta method.

2x2 grid, rows = directions (synonym->antonym, antonym->synonym), left column = OLD
paired-Delta injection (oneshot_switch_logit, demo-label site), right column = NEW k=4
subspace replacement. Each panel shows its own method's full alpha fan (layer profile of
mean logit(target_gold) - logit(source_gold), ±ci95 band, dashed alpha=0 baseline), plus the
OTHER method's best curve (highest peak over layers) dotted in crimson for direct reading.
Y-limits shared within each row. Both studies use the SAME 100 eval prompts and metric
(clean baselines agree to 0.001), so curves are directly overlayable; note alpha scales
different objects (full-rank Delta vs k=4 subspace coords).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FV_FORMATION_DIR, LABEL_GEOMETRY_DIR

DIRECTIONS = ["synonym_to_antonym", "antonym_to_synonym"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new_root", type=Path,
                   default=FV_FORMATION_DIR / "ablation" / "attention_head_mechanisms" / "payload_switch_steering")
    p.add_argument("--old_root", type=Path, default=LABEL_GEOMETRY_DIR / "oneshot_switch_logit")
    p.add_argument("--arm", type=str, default="replace_both",
                   choices=["replace_both", "replace_target_only"])
    p.add_argument("--directions", nargs="+", default=DIRECTIONS)
    return p.parse_args()


def load_new(root, direction, arm):
    """{alpha: (layers, mean, ci95)}, baseline — new subspace-replacement npz."""
    z = np.load(root / direction / f"{arm}_sweep.npz", allow_pickle=False)
    contrast = z["logit_tgt"] - z["logit_src"]  # (L, A, N)
    out = {}
    for ai, alpha in enumerate(z["alphas"]):
        c = contrast[:, ai]
        out[float(alpha)] = (z["layers"].astype(int), c.mean(axis=-1),
                             1.96 * c.std(axis=-1, ddof=1) / np.sqrt(c.shape[-1]))
    base = float((z["clean_logit_tgt"] - z["clean_logit_src"]).mean())
    return out, base


def load_old(root, direction, site="label"):
    """{alpha: (layers, mean, ci95)}, baseline — old paired-Delta logit_diff.json."""
    data = json.loads((root / direction / "logit_diff.json").read_text())
    by_alpha = {}
    for c in data["conditions"].values():
        if c["site"] == site:
            by_alpha.setdefault(float(c["alpha"]), []).append(c)
    out = {}
    for alpha, recs in by_alpha.items():
        recs = sorted(recs, key=lambda r: r["layer"])
        out[alpha] = (np.array([r["layer"] for r in recs]),
                      np.array([r["mean_logit_diff"] for r in recs]),
                      np.array([r["ci95"] if r["ci95"] is not None else np.nan for r in recs]))
    return out, float(data["baseline_alpha0"]["mean_logit_diff"])


def best_alpha(series):
    return max(series, key=lambda a: np.nanmax(series[a][1]))


def plot_panel(ax, series, baseline, title, neighbour=None):
    alphas = sorted(series)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(alphas)))
    for col, alpha in zip(cmap, alphas):
        x, y, ci = series[alpha]
        ax.plot(x, y, "-o", color=col, ms=3, lw=1.5, label=f"α={alpha:g}")
        ok = ~np.isnan(ci)
        if ok.any():
            ax.fill_between(np.asarray(x)[ok], (y - ci)[ok], (y + ci)[ok], color=col, alpha=0.10)
    ax.axhline(baseline, ls="--", color="grey", lw=1.4, label=f"α=0 baseline ({baseline:.2f})")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    if neighbour is not None:
        nx, ny, nlabel = neighbour
        ax.plot(nx, ny, ls=":", color="crimson", lw=2.0, marker="x", ms=4, label=nlabel)
    ax.set_xlabel("injection layer")
    ax.set_ylabel("logit(target) − logit(source)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)


def main():
    args = parse_args()
    fig, axes = plt.subplots(len(args.directions), 2,
                             figsize=(13.5, 4.6 * len(args.directions)), squeeze=False)
    for i, direction in enumerate(args.directions):
        old_series, old_base = load_old(args.old_root, direction)
        new_series, new_base = load_new(args.new_root, direction, args.arm)

        ba_old, ba_new = best_alpha(old_series), best_alpha(new_series)
        best_old = (old_series[ba_old][0], old_series[ba_old][1], f"best old (α={ba_old:g})")
        best_new = (new_series[ba_new][0], new_series[ba_new][1], f"best new (α={ba_new:g})")

        dname = direction.replace("_", " ")
        plot_panel(axes[i][0], old_series, old_base,
                   f"{dname}\nOLD: paired-Δ injection @ demo label token", neighbour=best_new)
        plot_panel(axes[i][1], new_series, new_base,
                   f"{dname}\nNEW: k=4 subspace replacement @ demo label token", neighbour=best_old)

        # shared y-limits across the row (both methods' bands + baselines + neighbours)
        lo, hi = [], []
        for series, base in ((old_series, old_base), (new_series, new_base)):
            for x, y, ci in series.values():
                ci0 = np.nan_to_num(ci, nan=0.0)
                lo.append(np.nanmin(y - ci0)); hi.append(np.nanmax(y + ci0))
            lo.append(base); hi.append(base)
        pad = 0.06 * (max(hi) - min(lo))
        for ax in axes[i]:
            ax.set_ylim(min(lo) - pad, max(hi) + pad)

    fig.suptitle("Task-switch steering @ demo label token — old paired-Δ injection (left) vs "
                 f"new unpaired k=4 subspace replacement ({args.arm}, right)\n"
                 "identical eval prompts & metric; crimson dotted = the other panel's best curve; "
                 "α scales different objects (full-rank Δ vs k=4 subspace coords)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figdir = args.new_root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / f"fig_payload_switch_vs_paired_{args.arm}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
