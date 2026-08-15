#!/usr/bin/env python
"""Bar charts over ALL 90 tasks (72 train + 18 HELD-OUT) for pooled 39-head vs TRIAL staged
111-head FV vs baseline. Held-out tasks: hatched bars + red starred x-labels.
Panels selectable; sorted ascending by staged accuracy within the chosen setting.
Usage: plot_all90_bars.py <split_json> <artifacts_root> <out_png> <setting1> [<setting2> ...]
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

split_p, ar_root, dst = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
settings = sys.argv[4:]
split = json.load(open(split_p))
train, heldout = split["train_tasks"], split["heldout_tasks"]
tasks = train + heldout
TITLES = {"test_zeroshot": "zero-shot",
          "test_sametask_shuffled10": "same-task shuffled-label 10-shot",
          "test_mixedtask10": "mixed-task mixed-label 10-shot"}

data = {}
for t in tasks:
    d = {}
    for name, key in (("eval_headset.json", "pooled"), ("eval_staged_trial.json", "staged")):
        j = json.load(open(ar_root / t / name))
        for s in TITLES:
            d[f"{s}_{key}"] = max(j["settings"][s]["acc_by_layer"])
            d[f"{s}_base"] = j["settings"][s]["baseline"]
    data[t] = d

fig, axes = plt.subplots(len(settings), 1, figsize=(23, 5.8 * len(settings)), dpi=140,
                         squeeze=False)
for ax, s in zip(axes[:, 0], settings):
    order = sorted(tasks, key=lambda t: data[t][f"{s}_staged"])
    x = np.arange(len(order))
    ho = np.array([t in heldout for t in order])
    for j, (key, color, label) in enumerate((("pooled", "tab:blue", "pooled 39-head FV"),
                                             ("staged", "tab:orange", "TRIAL staged 111-head FV"))):
        vals = [data[t][f"{s}_{key}"] for t in order]
        off = (j - 1) * 0.27
        ax.bar(x[~ho] + off, np.array(vals)[~ho], width=0.27, color=color, label=label)
        ax.bar(x[ho] + off, np.array(vals)[ho], width=0.27, color=color, hatch="///",
               edgecolor="white", linewidth=0.4,
               label=label + " (HELD-OUT)" if j == 0 else None)
    bvals = np.array([data[t][f"{s}_base"] for t in order])
    ax.bar(x[~ho] + 0.27, bvals[~ho], width=0.27, color="0.45", label="no-steering baseline")
    ax.bar(x[ho] + 0.27, bvals[ho], width=0.27, color="0.45", hatch="///",
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    labels = [("* " + t) if t in heldout else t for t in order]
    ax.set_xticklabels(labels, rotation=62, ha="right", fontsize=6.4)
    for tick, is_ho in zip(ax.get_xticklabels(), ho):
        if is_ho:
            tick.set_color("tab:red"); tick.set_fontweight("bold")
    ax.set_ylabel("best-layer full-label acc")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.set_title(f"test setting: {TITLES[s]} — all 90 tasks (red */hatched = 18 HELD-OUT "
                 f"tasks, never seen in any fitting), ascending by staged FV", fontsize=11)
fig.suptitle("Pooled vs TRIAL-staged FV steering, train + HELD-OUT tasks "
             "(SANDBOX ext_steerability phase 2)", fontsize=12)
fig.tight_layout()
fig.savefig(dst, bbox_inches="tight")
print(f"wrote {dst}")
