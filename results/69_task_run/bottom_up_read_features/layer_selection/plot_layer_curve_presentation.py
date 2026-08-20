#!/usr/bin/env python3.12
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/workspace/function_vectors/results/69_task_run/bottom_up_read_features/layer_selection/layer_summary.csv"
OUT = "/root/.claude/jobs/6a46ec85/tmp/layer_curve_presentation.png"

L, acc = [], []
for r in csv.DictReader(open(SRC)):
    if r["alpha"] == "best":
        L.append(int(r["layer"])); acc.append(float(r["mean_acc_all"]))

TEAL, INK, MUTED = "#0e7c6b", "#181c1e", "#5d6771"

fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=150)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

ax.plot(L, acc, color=TEAL, lw=2.2, marker="o", ms=5, mfc=TEAL, mec="white", mew=0.9, zorder=3)

ax.set_xlim(-0.6, 27.6); ax.set_ylim(0, 0.145)
ax.set_xticks(range(0, 28, 3))
ax.set_yticks([0, 0.05, 0.10])
ax.set_xlabel("injection layer", color=INK, fontsize=11)
ax.set_ylabel("steered accuracy (mean, 69 tasks)", color=INK, fontsize=11)
ax.set_title("Dummy Label Token Steering (1 Shot)",
             fontsize=13, color=INK, loc="left", pad=10)
ax.grid(axis="y", color="#e8eae6", lw=0.8, zorder=0)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color("#c9ccc7")
ax.tick_params(colors=MUTED, labelsize=10)

fig.tight_layout()
fig.savefig(OUT, facecolor="white")
print("saved", OUT)
