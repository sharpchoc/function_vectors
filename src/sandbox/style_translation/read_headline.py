#!/usr/bin/env python
"""Headline bars for the read-steering sweep: per direction, unsteered (wrong-style context) vs steered vs the real k = 3 right-style
accuracy. Outputs -> <results>/read_steering/<tag>/: headline_bars.png (selected cell, pooled over families, per-family dots),
headline_top5.png (one panel per top-5 cell), headline_by_family.png (per family at the selected cell)."""
import argparse
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
from src.sandbox.style_translation.models import paths as model_paths

COL = {"unsteered": "#9e9e9e", "steered": "#1f4f7a", "k3": "#2b7a4b"}


def bars(ax, vals, title, dots=None, ylabel=True):
    """vals: {direction: (unsteered, steered, k3)}; dots: {direction: [(u, s, k) per family]}"""
    x = np.arange(2); w = 0.26
    for i, (key, lab) in enumerate((("unsteered", "unsteered (wrong-style demos)"), ("steered", "steered (read feature)"), ("k3", "real 3-shot, right-style demos"))):
        ys = [vals[d][i] for d in ("nat", "alt")]
        ax.bar(x + (i - 1) * w, ys, w, color=COL[key], label=lab)
        for xi, y in zip(x + (i - 1) * w, ys): ax.text(xi, y + .015, f"{y:.2f}", ha="center", fontsize=8)
        if dots:
            for j, d in enumerate(("nat", "alt")):
                pts = [t[i] for t in dots[d]]; ax.scatter(np.full(len(pts), x[j] + (i - 1) * w) + np.random.default_rng(0).uniform(-.05, .05, len(pts)), pts, s=9, color="k", alpha=.45, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(["→ natural\n(alt-context prompts)", "→ alternative\n(nat-context prompts)"]); ax.set_ylim(0, 1.08); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.3)
    if ylabel: ax.set_ylabel("success: target convention AND judge OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--tag", default="sweep10")
    args = ap.parse_args()
    R = model_paths(args.model)["results"] / "read_steering" / args.tag
    sel = json.load(open(R / "selected.json")); rows = list(csv.DictReader(open(R / "sweep.csv"))); base = list(csv.DictReader(open(R / "baseline.csv")))
    fams = sorted({r["family"] for r in rows})
    B = {(r["family"], r["target"]): (float(r["target_and_ok"]), float(r["k3_accuracy_of_target"])) for r in base}
    S = {(r["family"], r["target"], int(r["layer"]), float(r["alpha"])): float(r["success"]) for r in rows}
    def cell_vals(L, a):
        dots = {d: [(B[(f, d)][0], S[(f, d, L, a)], B[(f, d)][1]) for f in fams] for d in ("nat", "alt")}
        return {d: tuple(np.mean([t[i] for t in dots[d]]) for i in range(3)) for d in dots}, dots
    L, a = sel["layer"], sel["alpha"]; vals, dots = cell_vals(L, a)
    fig, ax = plt.subplots(figsize=(8, 5)); bars(ax, vals, f"Read-feature steering at every evidence token, layer {L}, α = {a:g}\nmean over {len(fams)} families (dots = families), held-out k = 3 prompts", dots); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(R / "headline_bars.png", dpi=150); plt.close(fig)
    top = sel["top5"]; fig, axes = plt.subplots(1, len(top), figsize=(4.2 * len(top), 4.6), sharey=True)
    for ax, t in zip(axes, top):
        v, dd = cell_vals(t["layer"], t["alpha"]); bars(ax, v, f"layer {t['layer']}, α = {t['alpha']:g} (mean {t['mean_success']:.2f})", dd, ylabel=ax is axes[0])
    axes[0].legend(fontsize=7, loc="upper left"); fig.suptitle("The five best cells of the sweep, same bars", fontsize=11); fig.tight_layout(); fig.savefig(R / "headline_top5.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True); x = np.arange(len(fams)); w = .27
    for ax, d in zip(axes, ("nat", "alt")):
        for i, key in enumerate(("unsteered", "steered", "k3")):
            ys = [(B[(f, d)][0], S[(f, d, L, a)], B[(f, d)][1])[i] for f in fams]; ax.bar(x + (i - 1) * w, ys, w, color=COL[key], label={"unsteered": "unsteered (wrong-style demos)", "steered": f"steered, L{L} α{a:g}", "k3": "real 3-shot, right-style demos"}[key])
        ax.set_ylim(0, 1.05); ax.set_title(f"steering towards the {'natural' if d == 'nat' else 'alternative'} convention", fontsize=10); ax.grid(axis="y", alpha=.3); ax.set_ylabel("success")
    axes[1].set_xticks(x); axes[1].set_xticklabels(fams, rotation=30, ha="right"); axes[0].legend(fontsize=8)
    fig.suptitle("Per family at the selected setting", fontsize=11); fig.tight_layout(); fig.savefig(R / "headline_by_family.png", dpi=130); plt.close(fig)
    print("->", R)


if __name__ == "__main__":
    main()
