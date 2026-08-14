#!/usr/bin/env python
"""Steerability-hypotheses analysis: which task properties predict zero-shot steering failure?

Merges the predictor table with clean-10-shot competence, computes Spearman correlations of
each predictor with y = best zero-shot steered accuracy (over all isolation products), and a
scatter grid with the failing tasks (<0.40) highlighted. n=29 - correlations are descriptive.
"""
import csv
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
from src.utils.paths import RESULTS_ROOT

OUT_DIR = RESULTS_ROOT / "sandbox" / "isolation_upper_bound"
rows = list(csv.DictReader(open(OUT_DIR / "steerability_predictors_full.csv")))

PREDICTORS = [
    ("clean10", "H1: clean 10-shot accuracy"),
    ("shuf10_baseline", "H2: unsteered shuffled-10 acc"),
    ("npc90_unc", "H3: per-prompt FV npc90 (sparse23)"),
    ("sr_centered", "H3b: centered stable rank"),
    ("cie_max", "H4: max task-specific CIE (zs)"),
    ("cie_top10_mean", "H4b: top-10 mean CIE (zs)"),
    ("label_ntok_mean", "H5: mean label tokens"),
    ("n_unique_out", "H6: # unique outputs"),
    ("out_entropy", "H6b: output entropy (bits)"),
    ("fv_norm_median", "extra: median per-prompt FV norm"),
]
y = np.array([float(r["y_zs_best"]) for r in rows])
fail = np.array([int(r["fail"]) for r in rows], dtype=bool)
tasks = [r["task"] for r in rows]

print(f"{'predictor':38s} {'rho':>6s} {'p':>7s}   fail-vs-pass medians")
stats = []
for key, label in PREDICTORS:
    x = np.array([float(r[key]) for r in rows])
    rho, p = spearmanr(x, y)
    med_f, med_p = np.median(x[fail]), np.median(x[~fail])
    stats.append((key, label, rho, p, med_f, med_p))
    print(f"{label:38s} {rho:6.2f} {p:7.3f}   {med_f:.3g} vs {med_p:.3g}")

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for ax, (key, label, rho, p, _, _) in zip(axes.flat, stats):
    x = np.array([float(r[key]) for r in rows])
    ax.scatter(x[~fail], y[~fail], c="tab:blue", s=38, label="steerable")
    ax.scatter(x[fail], y[fail], c="tab:red", s=48, marker="s", label="fail (<0.40)")
    for xi, yi, t in zip(x, y, tasks):
        if yi < 0.45 or t == "commonsense_qa":
            ax.annotate(t, (xi, yi), fontsize=6.5, alpha=0.8,
                        xytext=(2, 3), textcoords="offset points")
    ax.axhline(0.40, color="k", ls=":", lw=0.8)
    ax.set_title(f"{label}\nSpearman rho={rho:.2f} (p={p:.3f})", fontsize=9.5)
    ax.set_ylabel("best zero-shot steered acc")
    ax.grid(alpha=0.25)
axes.flat[0].legend(fontsize=8)
fig.suptitle("What predicts zero-shot steerability? 29 tasks, task-specific isolation upper bound "
             "(red = failing tasks; n=29, descriptive)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "steerability_predictors_scatter.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT_DIR/'steerability_predictors_scatter.png'}")

