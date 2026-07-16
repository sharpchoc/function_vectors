#!/usr/bin/env python
"""Stream W: heatmaps for the 1-shot preimage/FV projection-ablation study (CPU).

Reads the per-(task, arm) npz written by ablate_oneshot_preimage_logprob.py and renders, per arm,
a token-row x start-layer heatmap of the task-mean delta log p (ablated - clean) of the first
answer token. One shared symmetric color scale across ALL arms (the *_cf arms are the control, so
cross-arm comparability is the point). Also a per-task supplementary grid (tasks x arms).

Outputs under <root>/figures/:
  heatmap_<arm>.png              task-mean over all 170 prompts/task
  heatmap_all_arms.png           the 6 arms side by side, shared scale + one colorbar
  per_task_grid.png              tasks x arms grid (row-normalized shared scale)
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.paths import FV_FORMATION_DIR

ARMS = ["preimage_matched", "preimage_icl10", "fv",
        "preimage_matched_cf", "preimage_icl10_cf", "fv_cf"]
ARM_TITLES = {
    "preimage_matched": "preimage of position-matched regression",
    "preimage_icl10": "preimage of icl10 regression",
    "fv": "FV direction",
    "preimage_matched_cf": "preimage of position-matched regression — counterfactual task",
    "preimage_icl10_cf": "preimage of icl10 regression — counterfactual task",
    "fv_cf": "FV direction — counterfactual task",
}
ROW_TITLES = {"cue1": "cue1 (demo 'A:')", "target1": "target1 (demo label)",
              "final_cue": "final cue (query 'A:')"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path,
                   default=FV_FORMATION_DIR / "oneshot_preimage_ablation/train_varicl_top40")
    p.add_argument("--rows", nargs="+", default=None,
                   help="Only plot these token rows (e.g. target1). The color scale is computed "
                        "from the kept rows only, so small effects aren't washed out by the "
                        "final-cue magnitudes. Output filenames get a _<rows> suffix.")
    p.add_argument("--annotate", action="store_true",
                   help="Write the value into each cell of the per-arm heatmaps.")
    p.add_argument("--skip_per_arm", action="store_true",
                   help="Only render heatmap_all_arms and per_task_grid (skip the per-arm figures).")
    return p.parse_args()


def load_arm(task_dir, arm):
    """-> (row_names, [n_rows, 28] prompt-mean delta) or None."""
    f = task_dir / f"{arm}_delta_logp.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    delta = z["delta_logp"]                       # [rows, 28, n]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(delta, axis=2)
    return [str(r) for r in z["row_names"]], mean


def render(ax, grid, row_names, vmax, title, annotate=False, show_xlabel=True,
           show_ylabels=True):
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels([ROW_TITLES.get(r, r) for r in row_names] if show_ylabels else [],
                       fontsize=10)
    ax.set_xticks(range(0, grid.shape[1], 4))
    ax.tick_params(labelsize=9)
    if show_xlabel:
        ax.set_xlabel("start edit layer L (ablate h.b for all b ≥ L)", fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    if annotate:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.5,
                            color="black")
    return im


def main():
    args = parse_args()
    fig_dir = args.root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.rows:
        suffix += "_" + "_".join(args.rows)

    task_dirs = sorted(d for d in args.root.iterdir()
                       if d.is_dir() and d.name != "figures")
    tasks = [d.name for d in task_dirs]

    # per (arm) -> task -> (row_names, [rows, 28])
    per_arm = {arm: {} for arm in ARMS}
    for d in task_dirs:
        for arm in ARMS:
            got = load_arm(d, arm)
            if got is None:
                continue
            if args.rows:
                row_names, grid = got
                idx = [i for i, r in enumerate(row_names) if r in set(args.rows)]
                if not idx:
                    continue
                got = ([row_names[i] for i in idx], grid[idx])
            per_arm[arm][d.name] = got

    # task-mean grids per arm
    arm_grids = {}
    for arm in ARMS:
        if not per_arm[arm]:
            continue
        row_names = next(iter(per_arm[arm].values()))[0]
        stack = np.stack([g for _, g in per_arm[arm].values()])   # [tasks, rows, 28]
        arm_grids[arm] = (row_names, np.nanmean(stack, axis=0))
    if not arm_grids:
        raise SystemExit(f"no npz found under {args.root}")

    vmax = max(np.nanmax(np.abs(g)) for _, g in arm_grids.values())
    print(f"tasks={tasks}")
    print(f"shared scale vmax={vmax:.3f}")

    # --- per-arm figures ---
    for arm, (row_names, grid) in ({} if args.skip_per_arm else arm_grids).items():
        fig, ax = plt.subplots(figsize=(10, 0.8 * len(row_names) + 2.2),
                               constrained_layout=True)
        im = render(ax, grid, row_names, vmax,
                    f"{ARM_TITLES[arm]} — mean Δ log p(correct) over {len(per_arm[arm])} tasks, 1-shot prompts",
                    annotate=args.annotate)
        fig.colorbar(im, ax=ax, label="log p(ablated) − log p(clean)")
        out = fig_dir / f"heatmap_{arm}{suffix}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")

    # --- combined 2x3 figure (labels only on the left column / bottom row) ---
    fig, axes = plt.subplots(2, 3, figsize=(19, 7.5), constrained_layout=True)
    for k, arm in enumerate(ARMS):
        r, c = divmod(k, 3)
        ax = axes[r][c]
        if arm not in arm_grids:
            ax.axis("off")
            continue
        row_names, grid = arm_grids[arm]
        im = render(ax, grid, row_names, vmax, ARM_TITLES[arm],
                    show_xlabel=(r == 1), show_ylabels=(c == 0))
    fig.suptitle("1-shot projection-ablation (all ablations applied in 1-shot prompts): "
                 f"mean Δ log p(correct answer), {len(tasks)} tasks", fontsize=14)
    fig.colorbar(im, ax=axes, label="log p(ablated) − log p(clean)", shrink=0.85)
    out = fig_dir / f"heatmap_all_arms{suffix}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")

    # --- per-task supplementary grid ---
    fig, axes = plt.subplots(len(tasks), len(ARMS),
                             figsize=(3.6 * len(ARMS), 1.7 * len(tasks) + 1.0),
                             squeeze=False, constrained_layout=True)
    for ti, task in enumerate(tasks):
        for ai, arm in enumerate(ARMS):
            ax = axes[ti][ai]
            if task not in per_arm[arm]:
                ax.axis("off")
                continue
            row_names, grid = per_arm[arm][task]
            last = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                             interpolation="nearest")
            ax.set_xticks(range(0, grid.shape[1], 8) if ti == len(tasks) - 1 else [])
            ax.tick_params(labelsize=7)
            ax.set_yticks(range(len(row_names)))
            ax.set_yticklabels(row_names if ai == 0 else [], fontsize=8)
            if ti == 0:
                ax.set_title(ARM_TITLES[arm], fontsize=9)
            if ai == 0:
                ax.set_ylabel(task, fontsize=9, rotation=90)
            if ti == len(tasks) - 1:
                ax.set_xlabel("start layer L", fontsize=8)
    fig.suptitle(f"per-task Δ log p heatmaps, 1-shot prompts (shared scale ±{vmax:.2f})", fontsize=13)
    fig.colorbar(last, ax=axes, label="log p(ablated) − log p(clean)", shrink=0.6)
    out = fig_dir / f"per_task_grid{suffix}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
