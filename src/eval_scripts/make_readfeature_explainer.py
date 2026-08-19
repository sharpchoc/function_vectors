#!/usr/bin/env python
"""Poster explainer: decomposing read features into shared carrier + task-unique part.

Three panels:
  A (conceptual): the 69 per-task L6 read features all point nearly the same way,
    clustered around the all-task mean direction (the shared carrier).
  B (conceptual): each feature splits into its projection on the carrier plus the
    orthogonal task-unique part:  m_A = (m_A . mhat) mhat + r_A.
  C (data): cross-task pairwise cosine similarity of the unit features before vs
    after removing the carrier (raw ~.74 -> task-unique ~0), computed from the L6
    slot-averaged task features (label_avg10_L5-15_acts).

Panels A/B stay purely conceptual (no measured numbers — repo convention);
panel C carries the measured distributions.

Output: results/.../ablation/explainer_visuals/readfeature_decomposition.png (+ npz).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR  # noqa: E402

ACTS = ARTIFACTS_ROOT / "69_task_run" / "label_avg10_L5-15_acts"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"
OUT = TASK69_RUN_DIR / "bottom_up_read_features" / "ablation" / "explainer_visuals"
LAYER = 6

BLUE, ORANGE, GRAY = "#2f7fe0", "#e07b2f", "#6b7280"   # CVD-checked pair + neutral


def pairwise(M):
    M = M / M.norm(dim=1, keepdim=True)
    G = (M @ M.T).numpy()
    return G[np.triu_indices(len(M), k=1)]


def arrow(ax, x0, y0, x1, y1, color, lw=2.2, ls="-", alpha=1.0, z=3, head=14):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=head, color=color, lw=lw, linestyle=ls,
                                 alpha=alpha, zorder=z, shrinkA=0, shrinkB=0))


def main():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    feats, layers = {}, None
    for t in tasks:
        d = torch.load(ACTS / f"{t}.pt", map_location="cpu", weights_only=False)
        layers = d["layers"]
        feats[t] = d["acts"].double().mean(dim=0)[layers.index(LAYER)]
    X = torch.stack([feats[t] for t in tasks])            # (69, 4096)
    mhat = X.mean(dim=0)
    mhat = mhat / mhat.norm()
    R = X - (X @ mhat)[:, None] * mhat                    # task-unique parts

    raw = pairwise(X)
    uniq = pairwise(R)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "pairwise_cos_L6.npz", tasks=np.array(tasks), raw=raw, task_unique=uniq)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=250,
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax in axes[:2]:
        ax.set_xlim(-0.05, 1.12)
        ax.set_ylim(-0.05, 1.08)
        ax.set_aspect("equal")
        ax.axis("off")

    # ---------- panel A: all tasks point nearly the same way ----------
    ax = axes[0]
    base = np.deg2rad(52)
    rng = np.random.RandomState(3)
    for i, off in enumerate(np.linspace(-16, 16, 8)):
        th = base + np.deg2rad(off)
        L = 0.92 + rng.uniform(-0.05, 0.05)
        arrow(ax, 0, 0, L * np.cos(th), L * np.sin(th), BLUE, lw=1.8, alpha=0.65, head=11)
    arrow(ax, 0, 0, 1.02 * np.cos(base), 1.02 * np.sin(base), GRAY, lw=4.0, head=18)
    ax.annotate("all-task mean $\\bar{m}$\n(shared carrier)",
                (1.02 * np.cos(base), 1.02 * np.sin(base)), xytext=(0.30, 1.00),
                fontsize=13, color="0.25", ha="left", va="top")
    ax.annotate("read features $m_A$\none per task", (0.86, 0.42), fontsize=13,
                color=BLUE, ha="left")
    ax.set_title("one read feature per task —\nthey all point nearly the same way",
                 fontsize=14.5, pad=10)

    # ---------- panel B: decomposition ----------
    ax = axes[1]
    th_m, th_a = np.deg2rad(38), np.deg2rad(66)
    Lm = 1.0
    mA = np.array([Lm * np.cos(th_a), Lm * np.sin(th_a)])
    mdir = np.array([np.cos(th_m), np.sin(th_m)])
    proj = float(mA @ mdir) * mdir
    r = mA - proj
    arrow(ax, 0, 0, *(1.06 * mdir), GRAY, lw=1.6, ls=(0, (4, 3)), alpha=0.8, head=10)
    arrow(ax, 0, 0, *mA, BLUE, lw=3.2, head=16)
    arrow(ax, 0, 0, *proj, GRAY, lw=3.2, head=16)
    arrow(ax, *proj, *(proj + r), ORANGE, lw=3.2, head=16)
    # right-angle marker at the foot of the perpendicular
    s = 0.055
    p1 = proj - s * mdir
    p2 = p1 + s * (r / np.linalg.norm(r))
    p3 = proj + s * (r / np.linalg.norm(r))
    ax.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], color="0.45", lw=1.2, zorder=2)
    ax.annotate("$m_A$", mA * 0.55 + np.array([-0.11, 0.05]), fontsize=16, color=BLUE)
    ax.annotate("shared carrier\n$(m_A\\!\\cdot\\!\\hat{m})\\,\\hat{m}$",
                proj * 0.52 + np.array([0.05, -0.13]), fontsize=13, color="0.25")
    ax.annotate("task-unique part\n$r_A \\perp \\hat{m}$",
                proj + r * 0.45 + np.array([0.04, 0.0]), fontsize=13, color=ORANGE)
    ax.set_title("split each feature into carrier +\northogonal task-unique part",
                 fontsize=14.5, pad=10)

    # ---------- panel C: pairwise cos before/after ----------
    ax = axes[2]
    bins = np.linspace(-0.6, 1.0, 65)
    ax.hist(raw, bins=bins, color=BLUE, alpha=0.75, label="raw read features")
    ax.hist(uniq, bins=bins, color=ORANGE, alpha=0.75, label="task-unique parts")
    for v, c in ((raw.mean(), BLUE), (uniq.mean(), ORANGE)):
        ax.axvline(v, color=c, lw=1.8, ls=(0, (5, 3)))
    ax.annotate(f"mean {raw.mean():.2f}", (raw.mean(), ax.get_ylim()[1]),
                xytext=(raw.mean() - 0.03, ax.get_ylim()[1] * 0.97), ha="right",
                fontsize=13, color=BLUE)
    ax.annotate(f"mean {uniq.mean():.2f}", (uniq.mean(), ax.get_ylim()[1]),
                xytext=(uniq.mean() + 0.03, ax.get_ylim()[1] * 0.97), ha="left",
                fontsize=13, color=ORANGE)
    ax.set_xlabel("pairwise cosine similarity between tasks", fontsize=13)
    ax.set_ylabel("task pairs", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=12, loc="upper left", frameon=False)
    ax.grid(axis="y", color="0.92")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_title("carrier removed: tasks become\nnear-orthogonal (identity code)",
                 fontsize=14.5, pad=10)

    fig.suptitle("From one read feature to two components: shared carrier + task-unique code "
                 "(L6 label-token features, 69 tasks)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "readfeature_decomposition.png", bbox_inches="tight")
    print(f"raw mean {raw.mean():.3f} sd {raw.std():.3f} | unique mean {uniq.mean():.3f} "
          f"sd {uniq.std():.3f}")
    print(f"wrote {OUT / 'readfeature_decomposition.png'}")


if __name__ == "__main__":
    main()
