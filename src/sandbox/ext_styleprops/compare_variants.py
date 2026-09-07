#!/usr/bin/env python
"""Cross-variant comparison for the steering sandbox.

Alphabetical in both axes. Deliberately UNRANKED: no sorting by performance, no
best-cell highlight, no "winner" - nothing here is canonical (user decision 2026-09-06).

Outputs results/style_properties/steering/comparison_table.{csv,png}
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


def main():
    data = {}
    for c in CELLS:
        f = OUT / "variants" / c.name / "results.csv"
        if f.exists():
            data[c.name] = {r["property"]: r for r in csv.DictReader(open(f))}
    cells = sorted(data)
    props = sorted({p for rows in data.values() for p in rows})

    with open(OUT / "comparison_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["property"]
        for d in DIRECTIONS:
            head += [f"unsteered_{d}", f"reference_{d}"]
        head += [f"{c}::steered_{d}" for c in cells for d in DIRECTIONS]
        w.writerow(head)
        for p in props:
            any_row = next(data[c][p] for c in cells if p in data[c])
            row = [p]
            for d in DIRECTIONS:
                row += [any_row.get(f"unsteered_{d}", ""), any_row.get(f"reference_{d}", "")]
            row += [data[c].get(p, {}).get(f"steered_{d}", "") for c in cells for d in DIRECTIONS]
            w.writerow(row)

    fig, axes = plt.subplots(2, 1, figsize=(max(12, 1.15 * len(props) + 3), 9), sharex=True)
    x = np.arange(len(props))
    cols = plt.cm.tab20(np.linspace(0, 1, max(len(cells), 2)))
    for ax, d in zip(axes, DIRECTIONS):
        w_ = 0.8 / (len(cells) + 2)
        f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else np.nan
        base = [f(next(data[c][p] for c in cells if p in data[c]), f"unsteered_{d}") for p in props]
        ref = [f(next(data[c][p] for c in cells if p in data[c]), f"reference_{d}") for p in props]
        ax.bar(x - 0.4 + w_ / 2, base, w_, color="#bdbdbd", label="unsteered (0-shot)")
        for i, c in enumerate(cells, start=1):
            ax.bar(x - 0.4 + w_ * i + w_ / 2, [f(data[c].get(p, {}), f"steered_{d}") for p in props],
                   w_, color=cols[i - 1], label=c)
        ax.bar(x - 0.4 + w_ * (len(cells) + 1) + w_ / 2, ref, w_, color="#7f9c8b",
               label="reference: k>=4 in context")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"adopt {d.upper()} convention\n(unscorable = no)", fontsize=9)
        ax.set_title(f"steering direction: -> {d}", fontsize=9, loc="left")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=6.5, ncol=4, loc="upper right", framealpha=0.9)
    axes[-1].set_xticks(x, props, rotation=30, ha="right", fontsize=8)
    fig.suptitle("STEERING SANDBOX - variant comparison (alphabetical, unranked; no cell is canonical)\n"
                 "0-shot text - sentence rollouts - gibberish dropped by LLM judge - unscorable counts as not adopted",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "comparison_table.png", dpi=150)
    print(f"{len(cells)} cells x {len(DIRECTIONS)} directions, {len(props)} properties -> {OUT}/comparison_table.png")


if __name__ == "__main__":
    main()
