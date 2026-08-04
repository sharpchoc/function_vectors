#!/usr/bin/env python
"""Plot payload-subspace task-switch steering (reads steer_payload_switch_logit.py npz).

Aggregate figure styled on plot_switch_logit.py: rows = directions, cols = arms
(replace_both, replace_target_only); x = injection layer, y = mean[ logit(target_gold) -
logit(source_gold) ], one viridis line per alpha with a ±ci95 band, dashed grey alpha=0
clean baseline. Second figure: same grid but Δ log p (steered − clean) of the target gold
(solid) and source gold (dashed) per alpha. PNG only.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import FV_FORMATION_DIR

DIRECTIONS = ["synonym_to_antonym", "antonym_to_synonym"]
ARMS = ["replace_both", "replace_target_only"]
ARM_LABEL = {"replace_both": "erase source + write target",
             "replace_target_only": "write target only"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   default=FV_FORMATION_DIR / "ablation" / "attention_head_mechanisms" / "payload_switch_steering")
    p.add_argument("--directions", nargs="+", default=DIRECTIONS)
    p.add_argument("--arms", nargs="+", default=ARMS)
    return p.parse_args()


def load_npz(root, direction, arm):
    path = root / direction / f"{arm}_sweep.npz"
    return np.load(path, allow_pickle=False) if path.exists() else None


def sem_band(a, axis=-1):
    return 1.96 * a.std(axis=axis, ddof=1) / np.sqrt(a.shape[axis])


def plot_contrast_axis(ax, z, title):
    layers, alphas = z["layers"], z["alphas"]
    contrast = z["logit_tgt"] - z["logit_src"]              # (L, A, N)
    base = float((z["clean_logit_tgt"] - z["clean_logit_src"]).mean())
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(alphas)))
    for ai, (col, alpha) in enumerate(zip(cmap, alphas)):
        y = contrast[:, ai].mean(axis=-1)
        ci = sem_band(contrast[:, ai])
        ax.plot(layers, y, "-o", color=col, ms=3, lw=1.5, label=f"α={alpha:g}")
        ax.fill_between(layers, y - ci, y + ci, color=col, alpha=0.12)
    ax.axhline(base, ls="--", color="grey", lw=1.4, label=f"α=0 baseline ({base:.2f})")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("injection layer")
    ax.set_ylabel("logit(target) − logit(source)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)


def plot_dlogp_axis(ax, z, title):
    layers, alphas = z["layers"], z["alphas"]
    dpt = z["logp_tgt"] - z["clean_logp_tgt"][None, None, :]
    dps = z["logp_src"] - z["clean_logp_src"][None, None, :]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(alphas)))
    for ai, (col, alpha) in enumerate(zip(cmap, alphas)):
        ax.plot(layers, dpt[:, ai].mean(axis=-1), "-o", color=col, ms=3, lw=1.5, label=f"α={alpha:g} tgt")
        ax.plot(layers, dps[:, ai].mean(axis=-1), "--", color=col, lw=1.2, label=f"α={alpha:g} src")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("injection layer")
    ax.set_ylabel("Δ log p (steered − clean)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)


def main():
    args = parse_args()
    figdir = args.root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    for name, plot_axis in (("fig_payload_switch_aggregate", plot_contrast_axis),
                            ("fig_payload_switch_dlogp", plot_dlogp_axis)):
        fig, axes = plt.subplots(len(args.directions), len(args.arms),
                                 figsize=(6.5 * len(args.arms), 4 * len(args.directions)),
                                 squeeze=False)
        for i, direction in enumerate(args.directions):
            for j, arm in enumerate(args.arms):
                z = load_npz(args.root, direction, arm)
                if z is None:
                    axes[i][j].set_axis_off()
                    print(f"skip {direction}/{arm} (no npz)")
                    continue
                title = f"{direction.replace('_', ' ')}\n{ARM_LABEL.get(arm, arm)} @ demo label token"
                plot_axis(axes[i][j], z, title)
        fig.suptitle("d_payload k=4 subspace-replacement steering (10-shot mean targets, no paired prompts)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = figdir / f"{name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
