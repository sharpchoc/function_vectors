#!/usr/bin/env python
"""Per-property outcome breakdown for one steering cell: where every rollout lands.

Each rollout falls in exactly one of 6 cells (coherence x classifier), only one of which
counts as a success:

  coherent   + correct convention   -> SUCCESS
  coherent   + wrong convention     -> failure
  coherent   + unscorable           -> failure (model never produced the feature)
  incoherent + correct convention   -> failure (right convention, broken text)
  incoherent + wrong convention     -> failure
  incoherent + unscorable           -> failure

Stacked to 100% per property, one panel per steering direction.
Output: results/style_properties/steering/variants/<cell>/outcome_breakdown.png (+ .csv)
"""
import argparse
import csv
import json
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
from src.utils.paths import ARTIFACTS_ROOT, STYLE_PROPERTIES_DIR
from src.sandbox.ext_styleprops.properties import PROPS
from src.sandbox.ext_styleprops.grid import BY_NAME, DIRECTIONS
from src.sandbox.ext_styleprops.plot_grid import ALIAS, run_tag

GRID = ARTIFACTS_ROOT / "style_properties" / "steering" / "grid"
OUT = STYLE_PROPERTIES_DIR / "steering" / "variants"
CATS = [("coh_correct", "coherent + CORRECT convention  (success)", "#2a9d3f", None),
        ("coh_wrong", "coherent + wrong convention", "#e63946", None),
        ("coh_unscorable", "coherent + unscorable (feature never produced)", "#c9c9c9", None),
        ("inc_correct", "incoherent + correct convention", "#2a9d3f", "//"),
        ("inc_wrong", "incoherent + wrong convention", "#e63946", "//"),
        ("inc_unscorable", "incoherent + unscorable", "#8d8d8d", "//")]


def breakdown(prop, cond, tgt):
    tails = cond["tails"]
    coh = cond.get("coherent") or [None] * len(tails)
    labs = [PROPS[prop].classify(t) for t in tails]
    c = {k: 0 for k, *_ in CATS}
    for l, v in zip(labs, coh):
        pre = "inc" if v is False else "coh"
        key = "unscorable" if l is None else ("correct" if l == tgt else "wrong")
        c[f"{pre}_{key}"] += 1
    n = max(len(tails), 1)
    return {k: v / n for k, v in c.items()}, len(tails)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default="meandiff_unpaired__kall__succyes")
    args = ap.parse_args()
    cell = BY_NAME[args.cell]
    data, ns = {}, {}
    for d in DIRECTIONS:
        for f in sorted((GRID / "headline" / run_tag(cell.name, d)).glob("*.json")):
            j = json.load(open(f)); prop = j["property"]
            key = [k for k in j["conditions"] if k.startswith("steered_")][0]
            data[(prop, d)], ns[(prop, d)] = breakdown(prop, j["conditions"][key], d)
    props = sorted({p for p, _ in data})
    order = sorted(props, key=lambda p: -data[(p, "alt")]["coh_correct"])

    fig, axes = plt.subplots(2, 1, figsize=(max(11, 0.95 * len(order) + 3), 9), sharex=True)
    x = np.arange(len(order))
    for ax, d in zip(axes, DIRECTIONS):
        bottom = np.zeros(len(order))
        for key, lab, col, hatch in CATS:
            vals = np.array([data[(p, d)][key] for p in order])
            ax.bar(x, vals, 0.72, bottom=bottom, color=col, hatch=hatch,
                   edgecolor="white" if hatch else "none", linewidth=0.6, label=lab)
            bottom += vals
        for xi, p in enumerate(order):
            ax.text(xi, 1.015, f"{data[(p, d)]['coh_correct']:.2f}", ha="center", fontsize=7.5,
                    fontweight="bold", color="#1a5c27")
        ax.set_ylim(0, 1.10)
        ax.set_ylabel(f"share of rollouts, steering → {d.upper()}", fontsize=9)
        ax.set_title(f"steering direction: → {d}    (number above each bar = success rate)",
                     fontsize=9, loc="left")
    axes[0].legend(fontsize=7.5, ncol=2, loc="lower left", framealpha=0.95)
    axes[-1].set_xticks(x, order, rotation=30, ha="right", fontsize=9)
    fig.suptitle(f"Where every rollout lands — SANDBOX variant {cell.name}\n{cell.formula}\n"
                 "0-shot text · sentence rollouts · only 'coherent + correct' counts as success",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p_out = OUT / cell.name
    fig.savefig(p_out / "outcome_breakdown.png", dpi=160, bbox_inches="tight")
    with open(p_out / "outcome_breakdown.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["property", "direction", "n"] + [k for k, *_ in CATS])
        for p in order:
            for d in DIRECTIONS:
                w.writerow([p, d, ns[(p, d)]] + [round(data[(p, d)][k], 4) for k, *_ in CATS])
    print(f"{'property':15s} " + "  ".join(f"{k[:12]:>12s}" for k, *_ in CATS))
    for p in order:
        print(f"{p:15s} " + "  ".join(f"{data[(p,'alt')][k]:12.2f}" for k, *_ in CATS))
    print(f"\n-> {p_out}/outcome_breakdown.png")


if __name__ == "__main__":
    main()
