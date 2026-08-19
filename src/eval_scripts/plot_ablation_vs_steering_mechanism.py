#!/usr/bin/env python
"""Stream W: explanatory schematic — why steering at the demo label works while ablating it
looks null (redundant routes summed at the query token; necessity vs sufficiency).
Static figure, no data inputs. Output: oneshot_preimage_ablation/train_varicl_top40/figures/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
from utils.paths import FV_FORMATION_DIR

BLUE, AQUA, GRAY, RED = "#2a78d6", "#1baf7a", "#8a8a85", "#e34948"
INK, MUT = "#1f1f1e", "#6e6d66"
TINT = {BLUE: "#eaf1fb", AQUA: "#e6f6ef", GRAY: "#f0f0ee", RED: "#fdecec"}

fig, axes = plt.subplots(1, 3, figsize=(19, 6.9))
fig.patch.set_facecolor("white")

SRC = [  # (y-center, color, title, subtitle)
    (8.3, BLUE, "target1  (label ' prose')", "task signal in preimage coords"),
    (5.0, AQUA, "raw demo tokens", "'poetry → prose' stays readable"),
    (1.7, GRAY, "query word (' increased')", "weak 0-shot prior"),
]

def box(ax, x, y, w, h, ec, fc, title, sub, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y - h/2), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w/2, y + 0.32, title, ha="center", va="center", fontsize=10.5,
            color=tc, fontweight="bold")
    ax.text(x + w/2, y - 0.42, sub, ha="center", va="center", fontsize=8.6, color=MUT)

def arrow(ax, p, q, color, lw, ls="-", alpha=1.0, zorder=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14 + 2*lw,
                                 color=color, lw=lw, linestyle=ls, alpha=alpha,
                                 zorder=zorder, shrinkA=2, shrinkB=4))

def panel(ax, title, subtitle, blue_lw, blue_ls, out_txt, out_color, note, extra=None):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(5.0, 10.0, title, ha="center", fontsize=13, color=INK, fontweight="bold")
    ax.text(5.0, 9.35, subtitle, ha="center", fontsize=9.2, color=MUT)
    sx, sw = 0.15, 3.5
    for y, c, t, s in SRC:
        box(ax, sx, y, sw, 1.75, c, TINT[c], t, s)
    cx, cy = 6.05, 5.0
    ax.add_patch(Circle((cx, cy), 0.52, fc="white", ec=INK, lw=1.6, zorder=4))
    ax.text(cx, cy, "Σ", ha="center", va="center", fontsize=15, color=INK, zorder=5,
            fontweight="bold")
    ax.text(cx, 3.95, "attention →\nfinal cue ' A:'", ha="center", va="top",
            fontsize=8.2, color=MUT)
    # routes
    arrow(ax, (sx + sw + 0.12, 8.3), (cx - 0.42, cy + 0.34), BLUE, blue_lw, blue_ls)
    arrow(ax, (sx + sw + 0.12, 5.0), (cx - 0.58, cy), AQUA, 2.4)
    arrow(ax, (sx + sw + 0.12, 1.7), (cx - 0.42, cy - 0.34), GRAY, 1.3)
    # output
    arrow(ax, (cx + 0.56, cy), (7.15, cy), INK, 1.8)
    box(ax, 7.25, cy, 2.6, 1.9, out_color, TINT.get(out_color, "#f7f7f5"),
        out_txt[0], out_txt[1], tc=out_color if out_color == RED else INK)
    ax.text(5.0, 0.25, note, ha="center", va="bottom", fontsize=9.0, color=INK,
            style="italic")
    if extra:
        extra(ax, sx, sw, cx, cy)

# --- panel 1: clean ---
panel(axes[0], "CLEAN RUN", "parallel routes; contributions sum at the final cue",
      4.2, "-", ("correct answer", "log p high"), INK,
      "the blue route is used — and also backed up")

# --- panel 2: ablation ---
def ab_extra(ax, sx, sw, cx, cy):
    mx, my = (sx + sw + 0.12 + cx - 0.42) / 2, (8.3 + cy + 0.34) / 2
    ax.text(mx, my + 0.42, "⊘", ha="center", va="center", fontsize=17, color=RED)
    ax.text(7.0, 8.15, "component clamped to 0\nat every layer ≥ L\n(deficit ≤ its natural size)",
            ha="center", fontsize=8.4, color=RED)
panel(axes[1], "ABLATE the direction at target1",
      "remove the blue component; raw tokens & prior untouched",
      1.2, (0, (4, 3)), ("still correct", "Δ log p small"), INK,
      "null ≠ not causal: backups deliver the same vote",
      extra=ab_extra)

# --- panel 3: steering ---
def st_extra(ax, sx, sw, cx, cy):
    arrow(ax, (sx + sw + 0.12, 8.75), (cx - 0.28, cy + 0.5), RED, 5.6, zorder=6)
    ax.text(7.15, 8.0, "injected  α·FV(task B),  α ≫ natural size\nnothing in Σ cancels an addition",
            ha="center", fontsize=8.4, color=RED, fontweight="bold")
panel(axes[2], "STEER at target1",
      "add a large task-B vector at the same site; blue route still intact",
      2.2, "-", ("task FLIPS → B", "injection dominates Σ"), RED,
      "works because it tests SUFFICIENCY: unbounded, readout-aligned signal",
      extra=st_extra)

fig.suptitle("Why steering at the demo label works while ablating it looks null — redundant routes summed at the query token",
             fontsize=14.5, color=INK, fontweight="bold", y=1.02)
fig.text(0.5, -0.045,
         "Ablation tests NECESSITY given intact alternatives (bounded deletion — backups fill the gap).   "
         "Steering tests SUFFICIENCY (unbounded injection — no cancellation mechanism).\n"
         "Under redundant summation a route can be genuinely causal without being necessary — "
         "so a null ablation bounds necessity, it does not rule out causal use.",
         ha="center", fontsize=10.5, color=INK)
fig.tight_layout(rect=[0, 0.0, 1, 0.98])
out = (FV_FORMATION_DIR / "ablation/preimages/oneshot/main/train_varicl_top40/figures"
       / "ablation_vs_steering_mechanism.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(out)
