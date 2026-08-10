#!/usr/bin/env python
"""SANDBOX: render the §3.2 layer-ablation summary as a PNG (bar chart + winning-quad strip).

Reads layer_ablation_results.json / layer_quad_best.json and writes ONE summary PNG to
results/sandbox/sparse_head_selection/ (grid/summary PNGs only, per repo figure prefs).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT

ART = ARTIFACTS_ROOT / "sandbox" / "sparse_head_selection"
OUT = RESULTS_ROOT / "sandbox" / "sparse_head_selection"

ACCENT = "#0E7368"
ACCENT_SOFT = "#0E736820"
MUTED = "#5A6572"
FAINT = "#8B95A0"

results = json.load(open(ART / "layer_ablation_results.json"))
quad = json.load(open(ART / "layer_quad_best.json"))

singles = {k: v for k, v in results.items() if k.startswith("single_")}
pairs = {k: v for k, v in results.items() if k.startswith("pair_")}
best_single = max(singles.values(), key=lambda v: v["acc"])
best_pair = max(pairs.values(), key=lambda v: v["acc"])
full = results["all_task_specific"]
floor = results["no_intervention"]["acc"]
ablated = results["all_mean_ablated"]["acc"]
bq = quad["exhaustive_best_quad"]

bars = [
    (f"k = 1\n(L{best_single['layers_kept'][0]})", best_single["acc"]),
    ("k = 2\n(L" + "+L".join(map(str, best_pair["layers_kept"])) + ")", best_pair["acc"]),
    ("k = 4\n(L" + "+L".join(map(str, bq["layers"])) + ")", bq["acc"]),
    ("all 25\nlayers", full["acc"]),
]

fig, (ax, ax2) = plt.subplots(
    1, 2, figsize=(11.5, 4.4), dpi=200, gridspec_kw={"width_ratios": [1.35, 1]}
)
fig.patch.set_facecolor("white")

# --- left: accuracy vs layer budget ---
xs = range(len(bars))
vals = [b[1] for b in bars]
ax.bar(xs, vals, width=0.62, color=ACCENT, zorder=3)
ax.axhline(full["acc"], color=FAINT, ls=(0, (5, 4)), lw=1.2, zorder=2)
ax.axhline(floor, color=FAINT, ls=(0, (5, 4)), lw=1.2, zorder=2)
ax.text(-0.42, 0.545, f"— — ceiling: all 73 task-specific ({full['acc']:.3f})",
        fontsize=8, color=MUTED, va="top")
ax.text(-0.42, 0.518, f"— — floor: no intervention {floor:.3f} · all mean-ablated {ablated:.3f}",
        fontsize=8, color=MUTED, va="top")
for x, v in zip(xs, vals):
    ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold", color="#1E242B")
    ax.text(x, v + 0.042, f"{v / full['acc'] * 100:.0f}%", ha="center", fontsize=8, color=FAINT)
ax.set_xticks(list(xs))
ax.set_xticklabels([b[0] for b in bars], fontsize=9)
ax.set_ylim(0, 0.56)
ax.set_ylabel("zero-shot steering accuracy\n(teacher-forced full label, FV injected @ L9)", fontsize=9)
ax.set_title("Task-specific layers kept vs. accuracy — no small set suffices", fontsize=10.5, pad=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#DDE2E7", lw=0.7, zorder=0)
ax.tick_params(labelsize=9)

# --- right: winning quad layer strip ---
kept = set(bq["layers"])
head_counts = {}
for l, h, c in quad["winner"]["heads_kept_task_specific"]:
    head_counts[l] = head_counts.get(l, 0) + 1
for layer in range(28):
    color = ACCENT if layer in kept else ACCENT_SOFT
    ax2.bar(layer, 1, width=0.88, color=color, zorder=3)
    if layer in kept:
        ax2.text(layer, 0.5, str(head_counts[layer]), ha="center", va="center",
                 fontsize=9, fontweight="bold", color="white")
ax2.set_xlim(-0.6, 27.6)
ax2.set_ylim(0, 1.6)
ax2.set_yticks([])
ax2.set_xticks(range(0, 28, 4))
ax2.set_xticklabels([f"L{l}" for l in range(0, 28, 4)], fontsize=8)
ax2.set_xlabel("GPT-J layer", fontsize=9)
ax2.set_title(f"Winning quad L{'+L'.join(map(str, bq['layers']))} — 17 task-specific heads\n"
              "(counts shown; other 56 significant heads mean-ablated)", fontsize=10.5, pad=10)
ax2.spines[["top", "right", "left"]].set_visible(False)
ax2.text(13.5, 1.28, "incl. canonical trio (9,14) (12,10) (15,5) at c ≈ 1",
         ha="center", fontsize=8, color=MUTED)

fig.suptitle("SANDBOX §3.2 layer-restricted mean-ablations — 73 sparse-optimized FV heads, GPT-J",
             fontsize=11, y=1.02)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / "layer_ablation_summary.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
