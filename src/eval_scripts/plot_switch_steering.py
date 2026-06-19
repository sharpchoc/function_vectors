#!/usr/bin/env python
"""Plot behavioral task-switch steering accuracy curves (reads steer_switch_judge.py outputs).

One figure per (direction, site): x = injection layer, y = target-task judged accuracy, one
colored line per nonzero alpha + a dashed flat alpha=0 baseline (from <direction>__baseline).
Also a 4x2 aggregate (rows=directions, cols=sites).
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
              "prev_number_to_next_number", "next_number_to_prev_number",
              "prev_number_digits_to_next_number_digits", "next_number_digits_to_prev_number_digits"]
SITES = ["label", "final"]
SITE_LABEL = {"label": "demo label token", "final": "final prompt token"}


def load_acc(path):
    if not path.exists():
        return None
    return json.load(open(path))


def baseline_acc(root, direction):
    d = load_acc(root / f"{direction}__baseline" / "accuracy.json")
    if not d:
        return None
    return list(d.values())[0]["accuracy"]


def series_by_alpha(acc):
    """acc dict ('layer|alpha'->rec) -> {alpha: (layers[], accs[], ci[])} sorted by layer."""
    by = {}
    for rec in acc.values():
        by.setdefault(rec["alpha"], []).append(rec)
    out = {}
    for alpha, recs in by.items():
        recs = sorted(recs, key=lambda r: r["layer"])
        out[alpha] = (np.array([r["layer"] for r in recs]),
                      np.array([r["accuracy"] for r in recs]),
                      np.array([r["ci95"] if r["ci95"] is not None else np.nan for r in recs]))
    return out


def plot_axis(ax, acc, base, title):
    series = series_by_alpha(acc)
    alphas = sorted(series)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(alphas)))
    for c, alpha in zip(cmap, alphas):
        layers, accs, ci = series[alpha]
        ax.plot(layers, accs, "-o", color=c, ms=3, lw=1.5, label=f"α={alpha:g}")
        ok = ~np.isnan(ci)
        if ok.any():
            ax.fill_between(layers[ok], (accs - ci)[ok], (accs + ci)[ok], color=c, alpha=0.12)
    if base is not None:
        ax.axhline(base, ls="--", color="grey", lw=1.4, label=f"α=0 baseline ({base:.2f})")
    ax.set_xlabel("injection layer")
    ax.set_ylabel("target-task accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=LABEL_GEOMETRY_DIR / "oneshot_switch_steering")
    p.add_argument("--figdir", type=Path, default=LABEL_GEOMETRY_DIR / "oneshot_switch_steering" / "figures")
    args = p.parse_args()
    args.figdir.mkdir(parents=True, exist_ok=True)

    fig_agg, axes = plt.subplots(len(DIRECTIONS), len(SITES), figsize=(13, 4 * len(DIRECTIONS)))
    for i, direction in enumerate(DIRECTIONS):
        base = baseline_acc(args.root, direction)
        for j, site in enumerate(SITES):
            acc = load_acc(args.root / f"{direction}__{site}" / "accuracy.json")
            if acc is None:
                print(f"skip {direction} {site} (no accuracy.json)")
                continue
            title = f"{direction.replace('_', ' ')}\nsteer @ {SITE_LABEL[site]}"
            # standalone
            fig, ax = plt.subplots(figsize=(7, 5))
            plot_axis(ax, acc, base, title)
            fig.tight_layout()
            out = args.figdir / f"fig_switch_{direction}_{site}.png"
            fig.savefig(out, dpi=140)
            plt.close(fig)
            print(f"wrote {out}")
            # aggregate panel
            plot_axis(axes[i, j], acc, base, title)
    fig_agg.tight_layout()
    agg = args.figdir / "fig_switch_aggregate.png"
    fig_agg.savefig(agg, dpi=130)
    plt.close(fig_agg)
    print(f"wrote {agg}")


if __name__ == "__main__":
    main()
