#!/usr/bin/env python
"""SANDBOX: shot-count robustness of the competence predictor (3-panel scatter).

Panels: clean-6 competence vs best zero-shot steered acc, clean-10 vs same, clean-6 vs
clean-10. Reads clean{6,10}_competence.json (artifacts) and y_zs_best from
steerability_predictors_full.csv (results). Failing tasks (<0.40) highlighted.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT

OUT_DIR = RESULTS_ROOT / "sandbox" / "isolation_upper_bound"
AR = ARTIFACTS_ROOT / "sandbox" / "isolation_upper_bound"


def main():
    rows = list(csv.DictReader(open(OUT_DIR / "steerability_predictors_full.csv")))
    c6 = json.load(open(AR / "clean6_competence.json"))
    c10 = json.load(open(AR / "clean10_competence.json"))
    tasks = [r["task"] for r in rows]
    y = np.array([float(r["y_zs_best"]) for r in rows])
    fail = np.array([int(r["fail"]) for r in rows], dtype=bool)
    x6 = np.array([c6[t] for t in tasks])
    x10 = np.array([c10[t] for t in tasks])

    panels = [(x6, y, "clean 6-shot acc", "best zero-shot steered acc"),
              (x10, y, "clean 10-shot acc", "best zero-shot steered acc"),
              (x6, x10, "clean 6-shot acc", "clean 10-shot acc")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=140)
    for ax, (a, b, xl, yl) in zip(axes, panels):
        rho, p = spearmanr(a, b)
        ax.scatter(a[~fail], b[~fail], c="tab:blue", s=38, label="steerable")
        ax.scatter(a[fail], b[fail], c="tab:red", s=48, marker="s", label="fail (<0.40)")
        for ai, bi, t in zip(a, b, tasks):
            if fail[tasks.index(t)]:
                ax.annotate(t, (ai, bi), fontsize=6.5, alpha=0.8,
                            xytext=(2, 3), textcoords="offset points")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"Spearman rho={rho:.3f} (p={p:.1e})", fontsize=10)
        ax.grid(alpha=0.25)
        print(f"{xl} vs {yl}: rho={rho:.3f}")
    axes[0].legend(fontsize=8)
    fig.suptitle("Shot-count robustness of the competence predictor (29 tasks, SANDBOX "
                 "isolation upper bound)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "clean6_vs_clean10_scatter.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
