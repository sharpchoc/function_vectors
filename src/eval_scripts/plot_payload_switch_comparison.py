#!/usr/bin/env python
"""Comparison: payload-subspace REPLACEMENT steering vs the old paired-Delta method.

Both studies use the SAME 100 eval prompts, the same demo-label-token site and the same
readout (logit(target_gold) - logit(source_gold) at the final position; clean baselines agree
to 0.001), so curves are directly overlayable. Left column: layer profiles at the shared
alphas (solid = new subspace replacement, dashed = old paired-Delta injection). Right column:
effectiveness-vs-strength (peak over layers vs alpha, including the old study's alpha=8).

Caveat rendered in the suptitle: alpha scales different objects (old: a full-rank 4096-d
difference-of-means Delta_label; new: the k=4 subspace coords of the 10-shot target mean),
so per-alpha overlays are a convenience pairing, not an equivalence — the right column's
best-over-alpha view is the strength-free comparison.
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


def main():
    args = parse_args()
    fig, axes = plt.subplots(len(args.directions), 2,
                             figsize=(13.5, 4.4 * len(args.directions)), squeeze=False)
    for i, direction in enumerate(args.directions):
        new_series, new_base = load_new(args.new_root, direction, args.arm)
        old_series, old_base = load_old(args.old_root, direction)

        # ---- left: layer profiles at the shared alphas ----
        ax = axes[i][0]
        shared = sorted(set(new_series) & set(old_series))
        cmap = plt.cm.viridis(np.linspace(0, 0.9, len(shared)))
        for col, alpha in zip(cmap, shared):
            x, y, ci = new_series[alpha]
            ax.plot(x, y, "-o", color=col, ms=3, lw=1.8, label=f"new α={alpha:g}")
            ax.fill_between(x, y - ci, y + ci, color=col, alpha=0.10)
            ox, oy, _ = old_series[alpha]
            ax.plot(ox, oy, "--s", color=col, ms=3, lw=1.2, alpha=0.85, label=f"old α={alpha:g}")
        ax.axhline(new_base, ls="--", color="grey", lw=1.4, label=f"α=0 baseline ({new_base:.2f})")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_xlabel("injection layer")
        ax.set_ylabel("logit(target) − logit(source)")
        ax.set_title(f"{direction.replace('_', ' ')} — layer profiles @ demo label token\n"
                     "solid = subspace replacement (new), dashed = paired-Δ injection (old)",
                     fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5, ncol=2)

        # ---- right: peak over layers vs alpha (all alphas each method has) ----
        ax = axes[i][1]
        for series, base, col, lab in ((new_series, new_base, "tab:blue", "new: k=4 subspace replacement"),
                                       (old_series, old_base, "tab:red", "old: paired-Δ injection")):
            alphas = sorted(series)
            peaks = [np.nanmax(series[a][1]) for a in alphas]
            ax.plot([0] + alphas, [base] + peaks, "-o", color=col, lw=1.8, ms=5, label=lab)
            for a, pk in zip(alphas, peaks):
                bl = int(series[a][0][np.nanargmax(series[a][1])])
                ax.annotate(f"L{bl}", (a, pk), textcoords="offset points", xytext=(0, 5),
                            fontsize=6.5, color=col, ha="center")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.axhline(new_base, ls="--", color="grey", lw=1.2, label=f"α=0 baseline ({new_base:.2f})")
        ax.set_xlabel("steering strength α")
        ax.set_ylabel("peak-over-layers logit(target) − logit(source)")
        ax.set_title(f"{direction.replace('_', ' ')} — effectiveness vs strength\n"
                     "(best injection layer annotated)", fontsize=10)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0, 0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["0", "0.5", "1", "2", "4", "8"])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    fig.suptitle("Task-switch steering @ demo label token: new unpaired subspace replacement "
                 f"({args.arm}) vs old paired-Δ injection\n"
                 "identical eval prompts & metric; α scales different objects "
                 "(k=4 subspace coords vs full-rank Δ) — right column is the strength-free view",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figdir = args.new_root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / f"fig_payload_switch_vs_paired_{args.arm}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
