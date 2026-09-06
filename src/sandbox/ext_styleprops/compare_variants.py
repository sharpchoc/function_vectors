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
from src.sandbox.ext_styleprops.variants import POPULATED, EMPTY, PROTOCOL

OUT = STYLE_PROPERTIES_DIR / "steering"


def main():
    data, caveat_flags = {}, {}
    for v in POPULATED:
        f = OUT / "variants" / v.name / "results.csv"
        if not f.exists():
            continue
        rows = {r["property"]: r for r in csv.DictReader(open(f))}
        data[v.name] = rows
        flags = []
        if not v.layers_searched:
            flags.append("no layer sweep")
        if not v.has_cf:
            flags.append("no cf control")
        if not v.has_reverse:
            flags.append("no reverse")
        if not all(r["judged"] == "True" for r in rows.values()):
            flags.append("unjudged")
        caveat_flags[v.name] = flags
    variants = sorted(data)
    props = sorted({p for rows in data.values() for p in rows})

    with open(OUT / "comparison_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["property", "unsteered_strict"]
                   + [f"{v}::{m}" for v in variants for m in ("strict", "unscorable", "cf_strict")])
        for p in props:
            base = next((data[v][p]["baseline_strict"] for v in variants if p in data[v]), "")
            row = [p, base]
            for v in variants:
                r = data[v].get(p, {})
                row += [r.get("strict", ""), r.get("unscorable", ""), r.get("cf_strict", "")]
            w.writerow(row)

    # figure: grouped bars, alphabetical, no highlight
    fig, ax = plt.subplots(figsize=(max(11, 1.05 * len(props) + 3), 5.6))
    x = np.arange(len(props))
    w_ = 0.8 / (len(variants) + 1)
    greys = ["#9e9e9e"]
    cols = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
    ax.bar(x - 0.4 + w_ / 2, [float(next((data[v][p]["baseline_strict"] for v in variants if p in data[v]), 0)) for p in props],
           w_, color=greys[0], label="unsteered")
    for i, v in enumerate(variants, start=1):
        vals = [float(data[v][p]["strict"]) if p in data[v] and data[v][p]["strict"] != "" else np.nan
                for p in props]
        lab = v + (f"  [{', '.join(caveat_flags[v])}]" if caveat_flags[v] else "")
        ax.bar(x - 0.4 + w_ * i + w_ / 2, vals, w_, color=cols[(i - 1) % len(cols)], label=lab)
    ax.set_xticks(x, props, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("adopt target convention\n(unscorable = no)", fontsize=9)
    ax.set_title("STEERING SANDBOX — variant comparison (alphabetical, unranked)\n"
                 "no cell is canonical; bracketed flags mark cells that searched less or "
                 "lack controls, so peaks are not directly comparable", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "comparison_table.png", dpi=150, bbox_inches="tight")

    print(f"{len(variants)} populated cells, {len(EMPTY)} not run; {len(props)} properties")
    for v in variants:
        print(f"  {v:32s} flags: {caveat_flags[v] or ['-']}")
    print(f"-> {OUT}/comparison_table.png")


if __name__ == "__main__":
    main()
