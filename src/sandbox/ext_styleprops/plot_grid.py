#!/usr/bin/env python
"""The two figures per grid cell, plus per-cell CSVs (user spec 2026-09-06).

figure 1  summary.png   per property: unsteered / steered / k>=4 reference / cf control,
                        for BOTH steering directions; unscorable % annotated
figure 2  by_layer.png  per property panel: adherence vs injection layer, one line per
                        direction (judged, each layer at its own best dose), unsteered floor

Metric = variant_metrics.stats: strict = P(target convention | rollout coherent), an
unscorable rollout counting as NOT adopting. Rollouts judged gibberish are dropped.

>>> SANDBOX: no cell is canonical. Figures are titled accordingly. <<<
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
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, STYLE_PROPERTIES_DIR
from src.sandbox.ext_styleprops.variant_metrics import stats
from src.sandbox.ext_styleprops.grid import (CELLS, BY_NAME, DIRECTIONS, SCREEN_LAYERS,
                                             REFERENCE_K)

GRID = ARTIFACTS_ROOT / "style_properties" / "steering" / "grid"
OUT = STYLE_PROPERTIES_DIR / "steering" / "variants"
# paired == unpaired when there is no success filter (proved: cos=1.0000), so those cells
# share one run; map the paired name onto the executed unpaired run.
ALIAS = {"meandiff_paired__kall__succno": "meandiff_unpaired__kall__succno",
         "meandiff_paired__k2__succno": "meandiff_unpaired__k2__succno"}
DCOL = {"alt": "#e63946", "nat": "#457b9d"}


def run_tag(cell_name, direction):
    return f"{ALIAS.get(cell_name, cell_name)}__{direction}"


def load_headline(cell_name, direction, prop):
    f = GRID / "headline" / run_tag(cell_name, direction) / f"{prop}.json"
    return json.load(open(f)) if f.exists() else None


def load_refs(direction, prop):
    f = GRID / "refs" / direction / f"{prop}.json"
    return json.load(open(f)) if f.exists() else None


def collect(cell_name, props):
    rows = {}
    for prop in props:
        r = {"property": prop}
        for direction in DIRECTIONS:
            refs = load_refs(direction, prop)
            hd = load_headline(cell_name, direction, prop)
            if refs:
                u = stats(prop, refs["conditions"]["unsteered_zeroshot"], tgt=direction)
                r[f"unsteered_{direction}"] = round(u["overall"], 3)
                rk = refs["conditions"].get(f"reference_k{REFERENCE_K}")
                if rk:
                    s = stats(prop, rk, tgt=direction)
                    r[f"reference_{direction}"] = round(s["overall"], 3)
                    r[f"reference_{direction}_unscorable"] = round(s["unscorable"], 3)
            if hd:
                C = hd["conditions"]
                sk = [k for k in C if k.startswith("steered_")]
                if sk:
                    s = stats(prop, C[sk[0]], tgt=direction)
                    r[f"steered_{direction}"] = round(s["overall"], 3)
                    r[f"steered_{direction}_coherentonly"] = round(s["strict"], 3)
                    r[f"steered_{direction}_conditional"] = round(s["conditional"], 3)
                    r[f"steered_{direction}_unscorable"] = round(s["unscorable"], 3)
                    r[f"steered_{direction}_incoherent"] = round(s["incoherent"], 3)
                    r[f"steered_{direction}_n"] = s["n"]
                    r[f"L_{direction}"] = hd["pick"]["L"]
                    r[f"alpha_{direction}"] = hd["pick"]["alpha"]
                if "cfprop" in C:
                    r[f"cf_{direction}"] = round(stats(prop, C["cfprop"], tgt=direction)["overall"], 3)
        rows[prop] = r
    return rows


def fig_summary(cell, rows, path):
    props = sorted(rows)
    x = np.arange(len(props))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, 0.95 * len(props) + 3), 8.4), sharex=True)
    for ax, direction in zip(axes, DIRECTIONS):
        w = 0.2
        series = [("unsteered", "#bdbdbd", -1.5), ("steered", DCOL[direction], -0.5),
                  ("reference", "#7f9c8b", 0.5), ("cf", "#c9ada7", 1.5)]
        for name, col, off in series:
            vals = [rows[p].get(f"{name}_{direction}", np.nan) for p in props]
            lab = {"unsteered": "unsteered (0-shot text)",
                   "steered": f"steered -> {direction}",
                   "reference": f"reference: k>={REFERENCE_K} in context",
                   "cf": "control: other property's vector"}[name]
            ax.bar(x + off * w, vals, w, color=col, label=lab)
        for xi, p in enumerate(props):
            u = rows[p].get(f"steered_{direction}_unscorable")
            if u is not None:
                ax.text(xi, -0.055, f"{u*100:.0f}", ha="center", fontsize=6.5, color="#a00")
        ax.text(-0.9, -0.055, "unscorable %", ha="right", fontsize=6.5, color="#a00")
        ax.set_ylim(-0.1, 1.05)
        ax.set_ylabel(f"success rate -> {direction.upper()}\n(incoherent or unscorable = failure)", fontsize=9)
        ax.legend(fontsize=7.5, ncol=4, loc="upper right", framealpha=0.9)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"steering direction: -> {direction}", fontsize=9, loc="left")
    axes[-1].set_xticks(x, props, rotation=30, ha="right", fontsize=8)
    fig.suptitle(f"SANDBOX variant  {cell.name}\n{cell.formula}\n"
                 "0-shot text · sentence rollouts · a rollout counts only if it is COHERENT and "
                 "adopts the convention", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_by_layer(cell, props, path):
    n, ncol = len(props), 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 2.9 * nrow), squeeze=False,
                             sharex=True, sharey=True)
    rows_csv = []
    for pi, prop in enumerate(sorted(props)):
        ax = axes[pi // ncol][pi % ncol]
        for direction in DIRECTIONS:
            f = GRID / "bylayer" / run_tag(cell.name, direction) / f"{prop}.json"
            if not f.exists():
                continue
            C = json.load(open(f))["conditions"]
            xs, ys = [], []
            for L in SCREEN_LAYERS:
                keys = [k for k in C if k.startswith(f"steered_L{L}_")]
                if not keys:
                    continue
                s = stats(prop, C[keys[0]], tgt=direction)
                xs.append(L); ys.append(s["overall"])
                rows_csv.append(dict(property=prop, direction=direction, layer=L,
                                     alpha=keys[0].split("_a")[1], overall=round(s["overall"], 3),
                                     coherent_only=round(s["strict"], 3),
                                     unscorable=round(s["unscorable"], 3),
                                     incoherent=round(s["incoherent"], 3), n=s["n"]))
            ax.plot(xs, ys, "o-", ms=3.5, lw=1.5, color=DCOL[direction], label=f"-> {direction}")
            refs = load_refs(direction, prop)
            if refs:
                u = stats(prop, refs["conditions"]["unsteered_zeroshot"], tgt=direction)["overall"]
                ax.axhline(u, color=DCOL[direction], ls=":", lw=1, alpha=0.7)
        ax.set_title(prop, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.03, 1.03); ax.set_xticks(SCREEN_LAYERS); ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if pi % ncol == 0:
            ax.set_ylabel("success rate\n(incoh/unscorable = fail)", fontsize=8)
        if pi + ncol >= n:
            ax.set_xlabel("injection layer", fontsize=8)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", bbox_to_anchor=(0.98, 0.04), fontsize=9,
               title="steering direction (dotted = unsteered floor)")
    fig.suptitle(f"SANDBOX variant  {cell.name} — accuracy by injection layer\n"
                 "each layer at its own best dose · judged sentence rollouts on 0-shot text",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150); plt.close(fig)
    return rows_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="*", default=None)
    args = ap.parse_args()
    props = sorted(json.load(open(REPO_ROOT / "task_splits" / "style_properties_pool.json"))["pass"])
    cells = [BY_NAME[c] for c in args.cells] if args.cells else list(CELLS)
    for cell in cells:
        if not (GRID / "headline" / run_tag(cell.name, "alt")).exists():
            print(f"{cell.name}: no headline run yet, skipped")
            continue
        d = OUT / cell.name
        d.mkdir(parents=True, exist_ok=True)
        rows = collect(cell.name, props)
        fields = sorted({k for r in rows.values() for k in r} - {"property"})
        with open(d / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["property"] + fields, extrasaction="ignore")
            w.writeheader()
            for p in sorted(rows):
                w.writerow(rows[p])
        fig_summary(cell, rows, d / "summary.png")
        bl = fig_by_layer(cell, props, d / "by_layer.png")
        if bl:
            with open(d / "by_layer.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(bl[0])); w.writeheader(); w.writerows(bl)
        alias = f"  (run shared with {ALIAS[cell.name]}: identical vector)" if cell.name in ALIAS else ""
        print(f"wrote {cell.name}{alias}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
