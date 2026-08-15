#!/usr/bin/env python
"""SANDBOX **TRIAL** figure: staged (111-head, competence-grouped accretion) vs pooled
(39-head) FV steering per train task. Two panels (zero-shot, mixed-task 10-shot):
paired bars sorted by pooled accuracy; baselines as black dashes.
Usage: plot_staged_trial.py <summary_csv> <out_png>
"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
fig, axes = plt.subplots(2, 1, figsize=(21, 11), dpi=140)
for ax, s, title in ((axes[0], "test_zeroshot", "zero-shot"),
                     (axes[1], "test_mixedtask10", "mixed-task mixed-label 10-shot")):
    rs = sorted(rows, key=lambda r: float(r[f"{s}_pooled_best"]))
    x = np.arange(len(rs))
    ax.bar(x - 0.2, [float(r[f"{s}_pooled_best"]) for r in rs], width=0.4,
           color="tab:blue", label="pooled 39-head FV (phase 1)")
    ax.bar(x + 0.2, [float(r[f"{s}_staged_best"]) for r in rs], width=0.4,
           color="tab:orange", label="TRIAL staged 111-head FV")
    ax.plot(x, [float(r[f"{s}_base"]) for r in rs], "k_", markersize=8,
            markeredgewidth=1.4, linestyle="none", label="no steering")
    ax.set_xticks(x)
    ax.set_xticklabels([r["task"] for r in rs], rotation=60, ha="right", fontsize=6.8)
    ax.set_ylabel("best-layer full-label acc")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(f"test setting: {title} — 72 train tasks, alpha=1", fontsize=11)
fig.suptitle("TRIAL: staged sparse selection by competence groups (48+23+16+5+19 = 111 heads) "
             "vs pooled 39-head baseline (SANDBOX ext_steerability)", fontsize=12)
fig.tight_layout()
fig.savefig(dst, bbox_inches="tight")
print(f"wrote {dst}")
