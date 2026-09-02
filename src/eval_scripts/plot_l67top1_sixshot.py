#!/usr/bin/env python
"""Summary CSV + headline bars for the 6-shot (L6/7 mean + top-1 dir) steering at L1.

Aggregates artifacts/69_task_run/l67top1_steering/sixshot/ (sixshot_l67top1_steer.py)
and compares against the original full-mean-at-L6 run (raw_mean_steering/sixshot_dummy/):
same prompt bank, same seeding scheme, same alpha grid.

Writes results/69_task_run/bottom_up_read_features/steering_results/l67top1/:
  sixshot_summary.csv   per task: group, baseline, per-alpha acc, best; + reference cols
  sixshot_bars.png      headline bars vs full-mean and real-6-shot references
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR  # noqa: E402

import os
_SUF = os.environ.get("L67_SUFFIX", "")          # "" (bank b) or "_bankA"
_NAME = os.environ.get("L67_NAME", "l67top1_steering" + _SUF)   # artifact dir
_OUTN = os.environ.get("L67_OUT", "l67top1" + _SUF)             # results subdir
AR = ARTIFACTS_ROOT / "69_task_run" / _NAME / os.environ.get("L67_SIXSHOT_SUB", "sixshot")
REF = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "sixshot_dummy"
OUT = TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results" / _OUTN
WA_CSV = TASK69_RUN_DIR / "bottom_up_read_features" / "steering_results" / "l67top1_bankA" / "sixshot_summary.csv"
ALPHAS = (0.5, 1.0, 2.0, 4.0)


def main():
    files = sorted(AR.glob("*.json"))
    assert len(files) == 69, len(files)
    rows = []
    for f in files:
        d = json.load(open(f))
        r = json.load(open(REF / f"{f.stem}.json"))
        acc = {a: d["conditions"][f"a{a}"]["acc"] for a in ALPHAS}
        racc = {a: r["conditions"][f"dummy6_steer_a{a}"]["acc"] for a in ALPHAS}
        rows.append({"task": f.stem, "group": d["group"],
                     "baseline": d["conditions"]["baseline"]["acc"],
                     **{f"a{a}": acc[a] for a in ALPHAS},
                     "best": max(acc.values()),
                     **{f"ref_fullmeanL6_a{a}": racc[a] for a in ALPHAS},
                     "ref_fullmeanL6_best": max(racc.values()),
                     "ref_real6": r["conditions"]["real6_baseline"]["acc"]})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "sixshot_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    is_c = _NAME.startswith("ctop1")
    lay = os.environ.get("L67_SIXSHOT_SUB", "sixshot").replace("sixshot_", "") or "L1"
    lay = lay if lay.startswith("L") else "L1"
    vname = "$u_A$" if is_c else "$w_A$"
    bars = [("unsteered\ndummy", m("baseline"), "0.72"),
            ("full mean\n@L6, best $\\alpha$", m("ref_fullmeanL6_best"), "#0e7c6b")]
    if is_c and WA_CSV.exists():
        wa_best = float(np.mean([float(r["best"]) for r in csv.DictReader(open(WA_CSV))]))
        bars.append(("$w_A$ @L1\nbest $\\alpha$", wa_best, "#c2410c"))
    bars += [(f"{vname} @{lay}\n$\\alpha$=1", m("a1.0"), "#7c3aad"),
             (f"{vname} @{lay}\n$\\alpha$=2", m("a2.0"), "#9d6ad1") if is_c else None,
             (f"{vname} @{lay}\nbest $\\alpha$", m("best"), "#7c3aad"),
             ("real 6-shot\ndemos", m("ref_real6"), "0.35")]
    bars = [b for b in bars if b is not None]
    # --- SIMPLE headline figure (main text): dummy baseline | steered | real demos ---
    if is_c:
        hb = [("dummy 6-shot\n(unsteered)", m("baseline"), "0.72"),
              (f"dummy 6-shot\n+ read feature steering", m("a2.0"), "#7c3aad"),
              ("real 6-shot\ndemonstrations", m("ref_real6"), "0.35")]
        fh, axh = plt.subplots(figsize=(6.0, 4.4), dpi=150)
        fh.patch.set_facecolor("white"); axh.set_facecolor("white")
        xh = np.arange(3)
        for xi, (lab, v, c) in zip(xh, hb):
            axh.bar([xi], [v], width=0.6, color=c, zorder=3)
            axh.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=13, fontweight="bold", color="#181c1e")
        axh.set_xticks(xh, [b[0] for b in hb], fontsize=11)
        axh.set_ylim(0, 0.74)
        axh.set_yticks([0, 0.2, 0.4, 0.6])
        axh.set_ylabel("task accuracy (mean, 69 tasks)", fontsize=11)
        axh.set_title("Steering dummy targets with the read feature", fontsize=12.5,
                      loc="left", pad=10)
        axh.grid(axis="y", color="#e8eae6", lw=0.8, zorder=0)
        for sp in ["top", "right"]: axh.spines[sp].set_visible(False)
        fh.tight_layout()
        fh.savefig(OUT / "headline_bars.png", facecolor="white")
        plt.close(fh)
        print(f"wrote {OUT}/headline_bars.png")

    fig, ax = plt.subplots(figsize=(8.8 if is_c else 7.6, 4.6), dpi=150)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    x = np.arange(len(bars))
    for xi, (lab, v, c) in zip(x, bars):
        ax.bar([xi], [v], width=0.62, color=c, zorder=3)
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color="#181c1e")
    ax.set_xticks(x, [b[0] for b in bars], fontsize=10)
    ax.set_ylim(0, 0.74)
    ax.set_ylabel("task accuracy (mean, 69 tasks)", fontsize=11)
    ax.set_title(("6-shot dummy-target steering: $u_A$ = carrier + $n_A v_1$, injected at " + lay) if is_c else "6-shot dummy-target steering: $w_A$ = L6/7 mean + top-1 dir, injected at L1",
                 fontsize=11.5, loc="left", pad=10)
    ax.grid(axis="y", color="#e8eae6", lw=0.8, zorder=0)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "sixshot_bars.png", facecolor="white")
    print(f"wrote {OUT}/sixshot_bars.png + sixshot_summary.csv")
    print("means:", {b[0].replace(chr(10), ' '): round(b[1], 4) for b in bars})


if __name__ == "__main__":
    main()
