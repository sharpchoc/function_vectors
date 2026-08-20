#!/usr/bin/env python3.12
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TMP = "/root/.claude/jobs/6a46ec85/tmp"
rows = list(csv.DictReader(open(f"{TMP}/tasklevel_ridge_all_layers.csv")))
L = [int(r["layer"]) for r in rows]
held = [float(r["heldout_r2_trainmean"]) for r in rows]
loo = [float(r["loo_train_r2"]) for r in rows]

BLUE, GRAY, INK, MUTED = "#2a78d6", "#8a8a85", "#1a1a19", "#6b6b66"

fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

ax.plot(L, held, color=BLUE, lw=2, marker="o", ms=4.5, mfc=BLUE, mec="white", mew=0.8,
        label="held-out R² (14 test tasks, baseline = train-mean FV)", zorder=3)
ax.plot(L, loo, color=GRAY, lw=2, ls="--", marker="o", ms=3.5, mfc=GRAY, mec="white", mew=0.7,
        label="train LOO R² (55 tasks, leave-one-task-out)", zorder=2)

pk = held.index(max(held))
ax.annotate(f"peak {held[pk]:.3f} (L{L[pk]}–L{L[pk]+1})", (L[pk], held[pk]),
            xytext=(L[pk] + 1.2, held[pk] + 0.05), color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate(f"L6 (steering peak): {held[6]:.3f}", (6, held[6]),
            xytext=(3.2, 0.30), color=MUTED, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate(f"L0: {held[0]:.3f}", (0, held[0]), xytext=(0.3, 0.30), color=MUTED, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate(f"L27: {held[27]:.3f}", (27, held[27]), xytext=(24.6, 0.52), color=MUTED, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

ax.set_xlim(-0.6, 27.6); ax.set_ylim(0, 0.78)
ax.set_xticks(range(0, 28, 2))
ax.set_xlabel("layer of the task-mean label-token activation (feature)", color=INK)
ax.set_ylabel("R² against task FVs", color=INK)
ax.set_title("Task-level ridge: mean label-token activation → task FV, by layer\n"
             "55 train tasks as samples; λ by leave-one-task-out CV; scored on 14 held-out tasks",
             fontsize=11, color=INK, loc="left", pad=12)
ax.grid(axis="y", color="#e6e6e2", lw=0.8, zorder=0)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
for s in ["left", "bottom"]: ax.spines[s].set_color("#c9c9c4")
ax.tick_params(colors=MUTED, labelsize=9)
ax.legend(loc="lower right", frameon=False, fontsize=9)

fig.tight_layout()
out = f"{TMP}/tasklevel_ridge_r2_by_layer.png"
fig.savefig(out, facecolor="white")
print("saved", out)
