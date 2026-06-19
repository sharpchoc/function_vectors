#!/usr/bin/env python
"""Plot logit-readout task-switch steering (reads steer_switch_logit.py outputs).

One figure per (direction, site): x = injection layer, y = mean[ logit(target_gold) - logit(source_gold) ],
one colored line per alpha + a dashed flat alpha=0 baseline (clean contrast). Also a 4x2 aggregate.
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
from utils.paths import LABEL_GEOMETRY_DIR

DIRECTIONS = ["synonym_to_antonym", "antonym_to_synonym",
              "prev_number_digits_to_next_number_digits", "next_number_digits_to_prev_number_digits"]
SITES = ["label", "final"]
SITE_LABEL = {"label": "demo label token", "final": "final prompt token"}


def load(path):
    return json.load(open(path)) if path.exists() else None


def plot_axis(ax, data, site, title):
    conds = [c for c in data["conditions"].values() if c["site"] == site]
    by_alpha = {}
    for c in conds:
        by_alpha.setdefault(c["alpha"], []).append(c)
    alphas = sorted(by_alpha)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(alphas)))
    for col, alpha in zip(cmap, alphas):
        recs = sorted(by_alpha[alpha], key=lambda r: r["layer"])
        x = [r["layer"] for r in recs]
        y = [r["mean_logit_diff"] for r in recs]
        ci = [r["ci95"] if r["ci95"] is not None else np.nan for r in recs]
        ax.plot(x, y, "-o", color=col, ms=3, lw=1.5, label=f"α={alpha:g}")
        y, ci = np.array(y), np.array(ci)
        ok = ~np.isnan(ci)
        if ok.any():
            ax.fill_between(np.array(x)[ok], (y - ci)[ok], (y + ci)[ok], color=col, alpha=0.12)
    base = data["baseline_alpha0"]["mean_logit_diff"]
    ax.axhline(base, ls="--", color="grey", lw=1.4, label=f"α=0 baseline ({base:.2f})")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("injection layer")
    ax.set_ylabel("logit(target) − logit(source)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=LABEL_GEOMETRY_DIR / "oneshot_switch_logit")
    p.add_argument("--figdir", type=Path, default=LABEL_GEOMETRY_DIR / "oneshot_switch_logit" / "figures")
    args = p.parse_args()
    args.figdir.mkdir(parents=True, exist_ok=True)

    fig_agg, axes = plt.subplots(len(DIRECTIONS), len(SITES), figsize=(13, 4 * len(DIRECTIONS)))
    for i, direction in enumerate(DIRECTIONS):
        data = load(args.root / direction / "logit_diff.json")
        if data is None:
            print(f"skip {direction} (no logit_diff.json)")
            continue
        for j, site in enumerate(SITES):
            title = f"{direction.replace('_', ' ')}\nsteer @ {SITE_LABEL[site]}"
            fig, ax = plt.subplots(figsize=(7, 5))
            plot_axis(ax, data, site, title)
            fig.tight_layout()
            out = args.figdir / f"fig_logit_{direction}_{site}.png"
            fig.savefig(out, dpi=140)
            plt.close(fig)
            print(f"wrote {out}")
            plot_axis(axes[i, j], data, site, title)
    fig_agg.tight_layout()
    fig_agg.savefig(args.figdir / "fig_logit_aggregate.png", dpi=130)
    plt.close(fig_agg)
    print(f"wrote {args.figdir/'fig_logit_aggregate.png'}")


if __name__ == "__main__":
    main()
