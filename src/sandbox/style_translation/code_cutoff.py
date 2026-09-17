#!/usr/bin/env python
"""Step 3c — the k = 4 cutoff for the code families: which families survive? Rule (DECISIONS 2026-09-16, re-affirmed 2026-09-17):
accuracy at k = 4 >= CUT on BOTH poles (accuracy = convention shown in context AND judge OK, from analyze.py's summary.csv).
Outputs -> <results>/: cutoff_k4.csv (family, language, k0/k4 accuracy per pole, keep), code_pool.json, cutoff_k4.png (two bars
per family: nat / alt accuracy at k = 4, sorted by the weaker pole, cutoff line, dropped families hatched, colour = language)."""
import argparse
import csv
import datetime
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
from src.sandbox.style_translation.code_families import CODE_FAMILY

LANG_COLOR = {"Python": "#3572A5", "JavaScript": "#c9a227", "Rust": "#b7410e", "PHP": "#6f5499", "SQL": "#2e8b57", "C": "#555555", "CSS": "#563d7c", "Bash": "#4e8f2f", "R": "#198ce7"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--cut", type=float, default=0.30); ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    MP = model_paths(args.model); R = MP["results"] / args.tag if args.tag else MP["results"]
    rows = list(csv.DictReader(open(R / "summary.csv")))
    acc = {(r["family"], r["style"], int(r["k"])): float(r["accuracy"]) for r in rows}
    n_at = {(r["family"], r["style"], int(r["k"])): int(r["n"]) for r in rows}
    fams = sorted({r["family"] for r in rows if r["family"] in CODE_FAMILY})
    out = []
    for f in fams:
        d = dict(family=f, language=CODE_FAMILY[f].tgt_lang, n=n_at.get((f, "nat", 4), 0),
                 k0_nat=acc.get((f, "nat", 0), float("nan")), k0_alt=acc.get((f, "alt", 0), float("nan")),
                 k4_nat=acc.get((f, "nat", 4), float("nan")), k4_alt=acc.get((f, "alt", 4), float("nan")))
        d["min_k4"] = min(d["k4_nat"], d["k4_alt"]); d["keep"] = bool(d["min_k4"] >= args.cut); out.append(d)
    out.sort(key=lambda d: -d["min_k4"])
    with open(R / "cutoff_k4.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader()
        for d in out:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})
    pool = [d["family"] for d in out if d["keep"]]; dropped = [d["family"] for d in out if not d["keep"]]
    json.dump({"rule": f"accuracy at k = 4 >= {args.cut:.2f} on both poles (summary.csv of {args.model})", "cut": args.cut, "date": str(datetime.date.today()),
               "model": args.model, "pool": pool, "dropped": dropped}, open(R / "code_pool.json", "w"), indent=1)

    # figure: two bars per family
    x = np.arange(len(out)); w = 0.4
    fig, ax = plt.subplots(figsize=(max(14, 0.42 * len(out)), 5.2))
    for j, (pole, lab, hatch) in enumerate((("nat", "natural convention in context", None), ("alt", "alternative convention in context", "//"))):
        vals = [d[f"k4_{pole}"] for d in out]; cols = [LANG_COLOR.get(d["language"], "#888") for d in out]
        bars = ax.bar(x + (j - 0.5) * w, vals, w, color=cols, alpha=1.0 if pole == "nat" else 0.55, edgecolor="black", linewidth=0.5, label=lab)
        for b_, d in zip(bars, out):
            if not d["keep"]:
                b_.set_hatch("xx"); b_.set_edgecolor("#b00020")
    ax.axhline(args.cut, color="black", ls="--", lw=1.2, label=f"cutoff: both poles ≥ {args.cut:.0%}")
    ax.set_xticks(x); ax.set_xticklabels([d["family"] for d in out], rotation=90, fontsize=8)
    for t, d in zip(ax.get_xticklabels(), out):
        t.set_color("#b00020" if not d["keep"] else "black")
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy at k = 4 (convention followed AND judge OK)")
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor=c, label=l) for l, c in LANG_COLOR.items() if any(d["language"] == l for d in out)]
    handles.append(Patch(facecolor="white", edgecolor="#b00020", hatch="xx", label="dropped (a pole below the cutoff)"))
    ax.legend(handles=handles, fontsize=8, ncol=4, loc="upper right", frameon=False)
    ax.set_title(f"{MP['label']}: k = 4 in-context accuracy per family and pole — {len(pool)} of {len(out)} families survive the cutoff", fontsize=11)
    ax.grid(axis="y", alpha=0.3); fig.tight_layout(); fig.savefig(R / "cutoff_k4.png", dpi=150); plt.close(fig)
    print(f"pool {len(pool)} / {len(out)}; dropped: {dropped}")
    print("->", R / "cutoff_k4.png")


if __name__ == "__main__":
    main()
