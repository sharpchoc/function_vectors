#!/usr/bin/env python
"""Assign completed steering arms into the sandbox variant grid (no recomputation).

For every populated cell in variants.py: read its source run JSON(s), pick the best
available (layer, dose) point of its main arm by the shared strict metric, and write
  results/style_properties/steering/variants/<cell>/{spec.md, results.csv, adherence.png}
For empty cells: one NOT_RUN.md listing them and what each would require.

--check re-derives and compares against the numbers previously reported for the two runs
(extraction must be lossless, not a re-measurement).

>>> SANDBOX: no cell is canonical; there is no default steering result. <<<
"""
import argparse
import csv
import json
import re
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
from src.sandbox.ext_styleprops.variants import (ALL_VARIANTS, POPULATED, EMPTY, PROTOCOL,
                                                 SHARED_LAYERS, SHARED_ALPHAS)
from src.sandbox.ext_styleprops.variant_metrics import stats

STEER = ARTIFACTS_ROOT / "style_properties" / "steering"
PRE = ARTIFACTS_ROOT / "style_properties" / "prescreen"
OUT = STYLE_PROPERTIES_DIR / "steering"
VAR_OUT = OUT / "variants"

EXPECTED = {   # previously reported strict rates, for the losslessness check
    "meandiff__kall__succno": {"double_space": 1.00, "oxford_comma": 1.00, "quote_punct": 1.00,
                               "sentence_caps": 1.00, "contractions": 0.69, "us_uk": 0.57},
    "meandiff__k4__succyes": {"quote_punct": 1.00, "double_space": 0.98, "sentence_caps": 0.88,
                              "contractions": 0.25},
}


def reference_rate(prop):
    """What the model does reading a genuine alt-convention context (k>=4) - context, not a result."""
    recs = json.load(open(PRE / f"{prop}.json"))["records"]
    ref = [r for r in recs if r["pol"] == "alt" and r["k"] >= 4 and r["label"]]
    return float(np.mean([r["label"] == "alt" for r in ref])) if ref else np.nan


def collect(variant, props):
    """rows[prop] = best main-arm point + controls, over all sources of this cell."""
    rows = {}
    for src in variant.sources:
        d_dir = STEER / src.run
        for f in sorted(d_dir.glob("*.json")):
            prop = f.stem
            if prop not in props:
                continue
            d = json.load(open(f)); C = d["conditions"]
            sweep_vec = (d.get("best_from_sweep") or {}).get("vector", "")
            rx = re.compile(src.main_re)
            cands = {k: stats(prop, v) for k, v in C.items()
                     if rx.match(k) and v.get("tails")}
            if not cands:
                continue
            best = max(cands, key=lambda k: (cands[k]["judged"], cands[k]["strict"]))
            s = cands[best]
            prev = rows.get(prop)
            if prev and (prev["judged"], prev["strict"]) >= (s["judged"], s["strict"]):
                continue
            base = stats(prop, C["baseline_nat2alt"])
            row = dict(property=prop, source_run=src.run, arm=best,
                       n_arm_points=len(cands),
                       baseline_strict=round(base["strict"], 3),
                       strict=round(s["strict"], 3), conditional=round(s["conditional"], 3),
                       unscorable=round(s["unscorable"], 3), incoherent=round(s["incoherent"], 3),
                       n=s["n"], n_coherent=s["n_coherent"], judged=s["judged"],
                       reference_k4=round(reference_rate(prop), 3))
            cf_ok = src.cf and (src.cf_requires_vector in ("ANY", sweep_vec))
            if cf_ok and src.cf in C and C[src.cf].get("tails"):
                cf = stats(prop, C[src.cf])
                row.update(cf_strict=round(cf["strict"], 3), cf_unscorable=round(cf["unscorable"], 3))
            rev_keys = [k for k in C if src.reverse_re and re.match(src.reverse_re, k)
                        and C[k].get("tails")] if src.reverse_re else []
            if rev_keys:
                rv = stats(prop, C[rev_keys[0]], tgt="nat")
                rb = stats(prop, C["baseline_alt2nat"], tgt="nat")
                row.update(reverse_strict=round(rv["strict"], 3),
                           reverse_baseline=round(rb["strict"], 3))
            row["judged"] = bool(s["judged"])
            rows[prop] = row
    return rows


def write_spec(variant, rows, path):
    ctl = []
    ctl.append("counterfactual-property control: " + ("PRESENT" if variant.has_cf else "NOT RUN"))
    ctl.append("reverse direction (alt->nat): " + ("PRESENT" if variant.has_reverse else "NOT RUN"))
    judged = all(r["judged"] for r in rows.values()) if rows else False
    lines = [
        f"# Variant: `{variant.name}`",
        "",
        "> **SANDBOX.** One cell of the steering variant grid. Not canonical, not a headline",
        "> result. Promotion to repo standard requires an explicit user decision.",
        "",
        "| field | value |",
        "|---|---|",
        f"| technique | `{variant.technique}` |",
        f"| k filter | `{variant.k_filter}` |",
        f"| success filter | `{variant.success_filter}` |",
        f"| vector formula | {variant.formula} |",
        f"| vector artifact | `artifacts/style_properties/{variant.vector.replace(':', '` key `')}` |",
        f"| injection site | cue token (first cue of each document) |",
        f"| layers searched | {', '.join(map(str, variant.layers_searched)) or 'none (layer borrowed - see caveats)'} |",
        f"| doses searched | {', '.join(f'{a:g}' for a in variant.alphas_searched)} |",
        f"| coherence judged | {'yes' if judged else 'NO - strict rate here is unfiltered'} |",
        f"| properties | {len(rows)} |",
        "",
        "## Protocol (shared across all variants)",
        "",
        f"{PROTOCOL}.",
        "",
        f"Shared sweep grid for reference: layers {SHARED_LAYERS}, doses "
        f"{tuple(f'{a:g}' for a in SHARED_ALPHAS)}.",
        "",
        "## Arms",
        "",
        *[f"- {c}" for c in ctl],
        "",
        "## Caveats",
        "",
        *[f"- {c}" for c in (variant.caveats or ("none recorded",))],
        "",
        "## Provenance",
        "",
        *[f"- `artifacts/style_properties/steering/{s.run}/<prop>.json` -> arms matching "
          f"`{s.main_re}`"
          + (f", cf `{s.cf}`" if s.cf else "")
          + (f", reverse `{s.reverse_re}`" if s.reverse_re else "")
          for s in variant.sources],
        "",
        "Metrics recomputed from the stored rollouts by `variant_metrics.stats`; see",
        "`results.csv` for per-property numbers.",
        "",
    ]
    path.write_text("\n".join(lines))


def figure(variant, rows, path):
    props = sorted(rows, key=lambda p: p)          # alphabetical: no implied ranking
    x = np.arange(len(props)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(props) + 3), 4.6))
    ax.bar(x - w, [rows[p]["baseline_strict"] for p in props], w, color="#bdbdbd", label="unsteered")
    ax.bar(x, [rows[p]["strict"] for p in props], w, color="#e63946", label="steered (this variant)")
    ax.bar(x + w, [rows[p]["reference_k4"] for p in props], w, color="#457b9d",
           label="reference: reading genuine alt context (k>=4)")
    for xi, p in zip(x, props):
        ax.text(xi, -0.06, f"{rows[p]['unscorable']*100:.0f}", ha="center", fontsize=6.5, color="#a00")
    ax.text(-0.85, -0.06, "unscorable %", ha="right", fontsize=6.5, color="#a00")
    ax.set_xticks(x, props, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("adopt target convention\n(unscorable = no)", fontsize=9)
    miss = [] if variant.has_cf else ["no cf control"]
    if not variant.has_reverse:
        miss.append("no reverse arm")
    if not variant.layers_searched:
        miss.append("no layer sweep")
    ax.set_title(f"SANDBOX variant: {variant.name}"
                 + (f"   [{'; '.join(miss)}]" if miss else ""), fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify extraction is lossless, write nothing")
    args = ap.parse_args()
    props = set(json.load(open(REPO_ROOT / "task_splits" / "style_properties_pool.json"))["pass"])
    if not args.check:
        VAR_OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    for v in POPULATED:
        rows = collect(v, props)
        if args.check:
            exp = EXPECTED.get(v.name, {})
            for p, want in exp.items():
                got = rows[p]["strict"]
                good = abs(got - want) <= 0.015
                ok &= good
                print(f"{'OK  ' if good else 'FAIL'} {v.name:26s} {p:15s} expected {want:.2f} got {got:.2f}")
            continue
        vdir = VAR_OUT / v.name
        vdir.mkdir(parents=True, exist_ok=True)
        fields = ["property", "source_run", "arm", "n_arm_points", "judged", "baseline_strict", "strict",
                  "conditional", "unscorable", "incoherent", "n", "n_coherent",
                  "cf_strict", "cf_unscorable", "reverse_strict", "reverse_baseline",
                  "reference_k4"]
        with open(vdir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for p in sorted(rows):
                w.writerow(rows[p])
        write_spec(v, rows, vdir / "spec.md")
        figure(v, rows, vdir / "adherence.png")
        print(f"wrote {v.name}: {len(rows)} properties, cf={v.has_cf}, reverse={v.has_reverse}")
    if args.check:
        print("\nlossless" if ok else "\nMISMATCH")
        sys.exit(0 if ok else 1)
    # empty cells
    lines = ["# Steering variant cells NOT RUN", "",
             "> **SANDBOX.** The grid is {mean difference, mean activation, sparse head",
             "> selection} x {k filter} x {success filter}. These cells have no data.", "",
             "| cell | technique | k filter | success filter | would require |", "|---|---|---|---|---|"]
    for v in EMPTY:
        lines.append(f"| `{v.name}` | {v.technique} | {v.k_filter} | {v.success_filter} | {v.needs} |")
    lines += ["", f"Populated cells: {', '.join(v.name for v in POPULATED)}.", ""]
    (VAR_OUT / "NOT_RUN.md").write_text("\n".join(lines))
    print(f"wrote NOT_RUN.md ({len(EMPTY)} cells)")


if __name__ == "__main__":
    main()
