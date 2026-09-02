#!/usr/bin/env python
"""Explainer: read feature = shared carrier c + one task-unique direction v1 (L5-7).

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

import os
_BANKA = os.environ.get("BANKA") == "1"
ACTS = (ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA" / "actsfmt"
        if _BANKA else ARTIFACTS_ROOT / "69_task_run" / "label_avg10_L5-15_acts")
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
    """Three panels following the paper-draft definition (user wording 2026-09-02):
      A (conceptual): the per-task L5-7 mean read features all point nearly the same way;
         their cross-task mean is the shared carrier c.
      B (data):       pairwise cosine between tasks: raw mbar_A (~0.73) vs carrier-removed.
      C (conceptual): per task, project the carrier out of each of m_A(5), m_A(6), m_A(7);
         stack the three residuals into a 3 x d matrix and take its top SVD direction v1.
    Uses bank (a) (label_resid_means) rows 5, 6, 7."""
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    RM = ARTIFACTS_ROOT / "69_task_run" / "label_resid_means"
    L3 = [5, 6, 7]
    per_layer = {}
    for t in tasks:
        rm = torch.load(RM / f"{t}.pt", map_location="cpu", weights_only=False)["resid_means"]
        per_layer[t] = rm[L3].double()                          # (3, 4096)
    mbar = torch.stack([per_layer[t].mean(dim=0) for t in tasks])   # (69, 4096) L5-7 mean
    c = mbar.mean(dim=0)
    chat = c / c.norm()
    R_bar = mbar - (mbar @ chat)[:, None] * chat               # carrier-removed L5-7 means

    raw = pairwise(mbar)
    uniq = pairwise(R_bar)

    # per-task SVD of the three carrier-removed layer vectors (per-layer carrier direction,
    # as in build_bankA_taskunique_bases.py)
    Xl = torch.stack([per_layer[t] for t in tasks])            # (69, 3, 4096)
    cdirs = Xl.mean(dim=0); cdirs = cdirs / cdirs.norm(dim=1, keepdim=True)
    Rl = Xl - (Xl * cdirs).sum(-1, keepdim=True) * cdirs
    energy = []
    for ti in range(len(tasks)):
        U = Rl[ti] / Rl[ti].norm(dim=1, keepdim=True)
        sv = torch.linalg.svdvals(U)
        energy.append(float(sv[0] ** 2 / (sv ** 2).sum()))
    energy = np.array(energy)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "pairwise_cos_L5to7.npz", tasks=np.array(tasks), raw=raw,
             task_unique=uniq, top1_energy=energy)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=250,
                             gridspec_kw={"width_ratios": [1, 1.2, 1.1]})
    for ax in (axes[0], axes[2]):
        ax.set_xlim(-0.05, 1.12)
        ax.set_ylim(-0.05, 1.08)
        ax.set_aspect("equal")
        ax.axis("off")

    # ---------- A: all tasks point nearly the same way ----------
    ax = axes[0]
    base = np.deg2rad(52)
    rng = np.random.RandomState(3)
    for off in np.linspace(-16, 16, 8):
        th = base + np.deg2rad(off)
        L = 0.92 + rng.uniform(-0.05, 0.05)
        arrow(ax, 0, 0, L * np.cos(th), L * np.sin(th), BLUE, lw=1.8, alpha=0.65, head=11)
    arrow(ax, 0, 0, 1.02 * np.cos(base), 1.02 * np.sin(base), GRAY, lw=4.0, head=18)
    ax.annotate("shared carrier $c$\n(cross-task mean)",
                (1.02 * np.cos(base), 1.02 * np.sin(base)), xytext=(0.30, 1.00),
                fontsize=13, color="0.25", ha="left", va="top")
    ax.annotate("$\\bar{m}_A$ = mean of\n$m_A(5), m_A(6), m_A(7)$\none per task", (0.80, 0.36),
                fontsize=12.5, color=BLUE, ha="left")
    ax.set_title("A. read features (L5–7 mean) —\nthey all point nearly the same way",
                 fontsize=14, pad=10)

    # ---------- B: pairwise cos before/after removing the carrier ----------
    ax = axes[1]
    bins = np.linspace(-0.6, 1.0, 65)
    ax.hist(raw, bins=bins, color=BLUE, alpha=0.75, label="raw $\\bar{m}_A$")
    ax.hist(uniq, bins=bins, color=ORANGE, alpha=0.75, label="carrier removed")
    for v, col in ((raw.mean(), BLUE), (uniq.mean(), ORANGE)):
        ax.axvline(v, color=col, lw=1.8, ls=(0, (5, 3)))
    top = ax.get_ylim()[1]
    ax.annotate(f"mean {raw.mean():.2f}", (raw.mean(), top), xytext=(raw.mean() - 0.03, top * 0.97),
                ha="right", fontsize=12.5, color=BLUE)
    ax.annotate(f"mean {uniq.mean():.2f}", (uniq.mean(), top), xytext=(uniq.mean() + 0.03, top * 0.97),
                ha="left", fontsize=12.5, color=ORANGE)
    ax.set_xlabel("pairwise cosine similarity between tasks", fontsize=12.5)
    ax.set_ylabel("task pairs", fontsize=12.5)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11.5, loc="upper left", frameon=False)
    ax.grid(axis="y", color="0.92")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.set_title("B. tasks share a carrier; removing it\nleaves near-orthogonal remainders",
                 fontsize=14, pad=10)

    # ---------- C: per-layer projection + stack + SVD (conceptual) ----------
    ax = axes[2]
    th_c = np.deg2rad(30)
    cdir = np.array([np.cos(th_c), np.sin(th_c)])
    perp = np.array([-cdir[1], cdir[0]])                      # true task-unique direction
    arrow(ax, 0, 0, *(1.05 * cdir), GRAY, lw=1.6, ls=(0, (4, 3)), alpha=0.8, head=10)
    ax.annotate("$c$", 1.05 * cdir + np.array([0.02, -0.03]), fontsize=14, color="0.35")
    cols = ("#1f5fb4", BLUE, "#8db8f0")
    # three layer means: same carrier-ish component, slightly different residual sizes/tilts
    specs = ((0.50, 0.42, -14), (0.62, 0.52, 0), (0.56, 0.36, 14))   # (along c, along perp, tilt deg)
    for k, ((a, b, tilt), col) in enumerate(zip(specs, cols)):
        rdir = np.array([np.cos(np.deg2rad(90 + 30 + tilt)), np.sin(np.deg2rad(90 + 30 + tilt))])
        m = a * cdir + b * rdir
        proj = float(m @ cdir) * cdir
        arrow(ax, 0, 0, *m, col, lw=2.0, alpha=0.9, head=12)
        arrow(ax, *proj, *m, ORANGE, lw=1.5, alpha=0.55, head=9)   # residual = m - proj
        ax.annotate(f"$m_A({5 + k})$", m * 1.08 + np.array([0.0, 0.01]), fontsize=11.5,
                    color=col, ha="center")
    arrow(ax, 0, 0, *(0.50 * perp), ORANGE, lw=3.4, head=17)
    ax.annotate("$v_1$: top SVD direction\nof the 3 residuals\n(task-unique part)",
                (-0.62, 0.62), fontsize=12.5, color=ORANGE, ha="left", va="top")
    ax.set_xlim(-0.65, 1.15); ax.set_ylim(-0.05, 1.08)
    ax.set_title("C. per layer: project out $c$, stack the 3\nresiduals, take the top SVD direction",
                 fontsize=14, pad=10)

    fig.suptitle("Read feature = shared carrier $c$ + one task-unique direction $v_1$ "
                 "(L5–7 target-token activations, 69 tasks)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "readfeature_decomposition.png", bbox_inches="tight")
    print(f"raw mean {raw.mean():.3f} sd {raw.std():.3f} | carrier-removed mean {uniq.mean():.3f} "
          f"| v1 energy median {np.median(energy):.3f} min {energy.min():.3f}")
    print(f"wrote {OUT / 'readfeature_decomposition.png'}")


if __name__ == "__main__":
    main()
