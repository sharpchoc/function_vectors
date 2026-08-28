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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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


def series_by_alpha(data, site):
    """Return {alpha: (x, y, ci)} sorted, plus baseline value, for one site."""
    by_alpha = {}
    for c in data["conditions"].values():
        if c["site"] == site:
            by_alpha.setdefault(c["alpha"], []).append(c)
    out = {}
    for alpha, recs in by_alpha.items():
        recs = sorted(recs, key=lambda r: r["layer"])
        x = np.array([r["layer"] for r in recs])
        y = np.array([r["mean_logit_diff"] for r in recs])
        ci = np.array([r["ci95"] if r["ci95"] is not None else np.nan for r in recs])
        out[alpha] = (x, y, ci)
    return out, data["baseline_alpha0"]["mean_logit_diff"]


def best_alpha(series):
    """Alpha whose curve reaches the highest peak logit(target)-logit(source)."""
    return max(series, key=lambda a: np.nanmax(series[a][1]))


def plot_axis_pretty(ax, series, baseline, title, neighbour=None):
    alphas = sorted(series)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(alphas)))
    for col, alpha in zip(cmap, alphas):
        x, y, ci = series[alpha]
        ax.plot(x, y, "-o", color=col, ms=3, lw=1.5, label=f"α={alpha:g}")
        ok = ~np.isnan(ci)
        if ok.any():
            ax.fill_between(x[ok], (y - ci)[ok], (y + ci)[ok], color=col, alpha=0.12)
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


def plot_pretty_aggregate(root, figdir):
    """4x2 aggregate: y-axis shared per row, each panel's best curve dotted onto its neighbour."""
    fig, axes = plt.subplots(len(DIRECTIONS), len(SITES), figsize=(13, 4 * len(DIRECTIONS)))
    for i, direction in enumerate(DIRECTIONS):
        data = load(root / direction / "logit_diff.json")
        if data is None:
            print(f"skip {direction} (no logit_diff.json)")
            continue
        # series + best curve for each site
        per_site, baselines, best = {}, {}, {}
        for site in SITES:
            per_site[site], baselines[site] = series_by_alpha(data, site)
            ba = best_alpha(per_site[site])
            bx, by, _ = per_site[site][ba]
            best[site] = (bx, by, ba)
        # shared y-limits across the row (both sites' data + CI bands + baselines)
        lo, hi = [], []
        for site in SITES:
            for x, y, ci in per_site[site].values():
                ci0 = np.nan_to_num(ci, nan=0.0)
                lo.append(np.nanmin(y - ci0)); hi.append(np.nanmax(y + ci0))
            lo.append(baselines[site]); hi.append(baselines[site])
        ymin, ymax = min(lo), max(hi)
        pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        for j, site in enumerate(SITES):
            other = SITES[1 - j]
            obx, oby, oba = best[other]
            neighbour = (obx, oby, f"{SITE_LABEL[other]} best (α={oba:g})")
            title = f"{direction.replace('_', ' ')}\nsteer @ {SITE_LABEL[site]}"
            plot_axis_pretty(axes[i, j], per_site[site], baselines[site], title, neighbour)
            axes[i, j].set_ylim(ymin - pad, ymax + pad)
    fig.tight_layout()
    out = figdir / "fig_logit_aggregate_pretty.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


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

    plot_pretty_aggregate(args.root, args.figdir)


if __name__ == "__main__":
    main()
