#!/usr/bin/env python
"""Headline: mean steering accuracy per method, one panel per steering direction.

Average success rate over the 13 style properties - a rollout counts ONLY if it is judged
coherent AND adopts the target convention (incoherent and unscorable both count as failures),
with 95% bootstrap CIs over properties (properties are the sampling unit). No baselines -
this figure answers one question: which vector construction steers best.

Methods share one order (their across-direction mean) so the panels can be compared.
SANDBOX: ordering shows effectiveness, it does not promote any cell to a default.

Output: results/style_properties/steering/headline_methods.png (+ .csv)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_PROPERTIES_DIR
from src.sandbox.ext_styleprops.grid import CELLS, DIRECTIONS

OUT = STYLE_PROPERTIES_DIR / "steering"
RNG = np.random.RandomState(0)
TECH_COL = {"meandiff_paired": "#1f4e79", "meandiff_unpaired": "#2e75b6", "meanact": "#c55a11"}


def pretty(name):
    tech, k, s = name.split("__")
    tech = {"meandiff_paired": "mean-diff (paired)", "meandiff_unpaired": "mean-diff (unpaired)",
            "meanact": "mean-activation"}[tech]
    return f"{tech}   ·   {'k≥2' if k == 'k2' else 'all k'}   ·   {'success-filtered' if s == 'succyes' else 'no filter'}"


def boot_ci(vals, n=10000):
    vals = np.asarray([v for v in vals if not np.isnan(v)])
    if len(vals) < 2:
        return np.nan, np.nan
    means = vals[RNG.randint(0, len(vals), (n, len(vals)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    stats = {}
    for c in CELLS:
        f = OUT / "variants" / c.name / "results.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(open(f)))
        s = {}
        for d in DIRECTIONS:
            vals = [float(r[f"steered_{d}"]) for r in rows if r.get(f"steered_{d}") not in (None, "")]
            lo, hi = boot_ci(vals)
            s[d] = dict(mean=float(np.mean(vals)), lo=lo, hi=hi, n=len(vals))
        stats[c.name] = s
    order = sorted(stats, key=lambda n: -np.mean([stats[n][d]["mean"] for d in DIRECTIONS]))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), sharey=True)
    y = np.arange(len(order))
    for ax, d in zip(axes, ("alt", "nat")):
        means = [stats[n][d]["mean"] for n in order]
        lo = [stats[n][d]["mean"] - stats[n][d]["lo"] for n in order]
        hi = [stats[n][d]["hi"] - stats[n][d]["mean"] for n in order]
        cols = [TECH_COL[n.rsplit("__", 2)[0]] for n in order]
        ax.barh(y, means, 0.68, color=cols, xerr=[lo, hi],
                error_kw=dict(ecolor="#333", capsize=3, lw=1.2))
        for yi, m in zip(y, means):
            ax.text(m + 0.015, yi, f"{m:.2f}", va="center", fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.invert_yaxis()
        ax.set_xlabel("mean success rate (coherent AND correct convention)", fontsize=9)
        ax.set_title(f"steering → {'ALT' if d == 'alt' else 'NAT'} convention", fontsize=12,
                     fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [pretty(n) for n in order], fontsize=9)
    fig.suptitle("Which steering vector works best?  mean over 13 style properties, 95% bootstrap CI\n"
                 "(sandbox — 0-shot text, one vector at the cue token; incoherent OR unscorable rollouts count as failures)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "headline_methods.png", dpi=170, bbox_inches="tight")

    with open(OUT / "headline_methods.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method"] + [f"{d}_{k}" for d in DIRECTIONS for k in ("mean", "ci_lo", "ci_hi", "n_props")])
        for n in order:
            row = [n]
            for d in DIRECTIONS:
                s = stats[n][d]
                row += [round(s["mean"], 4), round(s["lo"], 4), round(s["hi"], 4), s["n"]]
            w.writerow(row)
    print(f"{'method':46s} {'alt mean [95% CI]':28s} nat mean [95% CI]")
    for n in order:
        a, b = stats[n]["alt"], stats[n]["nat"]
        print(f"{n:46s} {a['mean']:.3f} [{a['lo']:.3f},{a['hi']:.3f}]      "
              f"{b['mean']:.3f} [{b['lo']:.3f},{b['hi']:.3f}]")
    print(f"-> {OUT}/headline_methods.png")


if __name__ == "__main__":
    main()
