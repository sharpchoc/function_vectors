#!/usr/bin/env python
"""Coding-style ICL curve pooled over the 53-family pool (paper Appendix E): outcome shares by k.

For every k in 0..4 and every (family, convention) cell of the pool (results/code_styles/code_pool.json), the
step-3 summary (results/code_styles/summary.csv, 200 prompts per cell) splits continuations into
  success               target convention and judge-accepted
  judge-rejected        target convention but rejected by the plausibility judge
  wrong convention      the other convention
  no style choice       the continuation shows neither convention (unscorable)
Shares are averaged over the 106 cells (equal weight). Writes results/code_styles/accuracy_by_k_pooled.csv and
paper_materials/uploads/figures/appendix_e_coding_styles/coding_accuracy_by_k.pdf. CPU only.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import CODE_STYLES_DIR, REPO_ROOT  # noqa: E402
from utils.paper_style import apply_paper_style, C  # noqa: E402

FIG = REPO_ROOT / "paper_materials" / "uploads" / "figures" / "appendix_e_coding_styles" / "coding_accuracy_by_k.pdf"


def main():
    pool = set(json.load(open(CODE_STYLES_DIR / "code_pool.json"))["pool"])
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(CODE_STYLES_DIR / "summary.csv")):
        if r["family"] not in pool:
            continue
        k = int(r["k"])
        acc[k]["success"].append(float(r["accuracy"]))
        acc[k]["judge_rejected"].append(float(r["style_ok_but_unfaithful"]))
        acc[k]["wrong_convention"].append(float(r["wrong_style"]))
        acc[k]["no_style_choice"].append(float(r["unscorable"]))
    ks = sorted(acc)
    cols = ["success", "judge_rejected", "wrong_convention", "no_style_choice"]
    rows = [dict(k=k, n_cells=len(acc[k]["success"]), **{c: float(np.mean(acc[k][c])) for c in cols}) for k in ks]
    for r in rows:
        assert abs(sum(r[c] for c in cols) - 1) < 1e-6, r
    with open(CODE_STYLES_DIR / "accuracy_by_k_pooled.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        print(open(CODE_STYLES_DIR / "accuracy_by_k_pooled.csv").read())

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    labels = {"success": "success", "judge_rejected": "target convention, judge-rejected",
              "wrong_convention": "wrong convention", "no_style_choice": "no style choice"}
    colors = {"success": "#1F7A8C", "judge_rejected": "#A9D3DB", "wrong_convention": "#E08E2B",
              "no_style_choice": C.grey}
    bottom = np.zeros(len(ks))
    for c in cols:
        vals = np.array([r[c] for r in rows]) * 100
        bars = ax.bar(ks, vals, bottom=bottom, width=0.72, color=colors[c], label=labels[c], edgecolor="white", lw=0.5)
        if c in ("success", "no_style_choice"):
            for b, v, b0 in zip(bars, vals, bottom):
                ax.text(b.get_x() + b.get_width() / 2, b0 + v / 2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=6.5, color="white" if c == "success" else "black")
        bottom += vals
    ax.set_xticks(ks)
    ax.set_xlabel("visible choices $k$")
    ax.set_ylabel("share of prompts (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
