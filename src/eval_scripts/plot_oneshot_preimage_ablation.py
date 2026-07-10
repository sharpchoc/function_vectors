#!/usr/bin/env python
"""Stream W: heatmaps for the 1-shot preimage/FV projection-ablation study (CPU).

Reads the per-(task, arm) npz written by ablate_oneshot_preimage_logprob.py and renders, per arm,
a token-row x start-layer heatmap of the task-mean delta log p (ablated - clean) of the first
answer token. One shared symmetric color scale across ALL arms (the *_cf arms are the control, so
cross-arm comparability is the point). Also a per-task supplementary grid (tasks x arms).

Outputs under <root>/figures/:
  heatmap_<arm>.png              task-mean, --metric all170 (default) or test40 suffix
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
    "preimage_matched": "preimage (matched cells)",
    "preimage_icl10": "preimage (icl10 cells)",
    "fv": "FV direction",
    "preimage_matched_cf": "preimage matched — counterfactual task",
    "preimage_icl10_cf": "preimage icl10 — counterfactual task",
    "fv_cf": "FV direction — counterfactual task",
}
ROW_TITLES = {"cue1": "cue1 (demo 'A:')", "target1": "target1 (demo label)",
              "final_cue": "final cue (query 'A:')",
              "final_cue_ctx": "final cue [pre_label_icl2]",
              "final_cue_icl10": "final cue [last_prompt_icl10]"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path,
                   default=FV_FORMATION_DIR / "oneshot_preimage_ablation/train_varicl_max4_top40")
    p.add_argument("--metric", choices=["all170", "test40"], default="all170")
    p.add_argument("--annotate", action="store_true",
                   help="Write the value into each cell of the per-arm heatmaps.")
    return p.parse_args()


def load_arm(task_dir, arm, metric):
    """-> (row_names, [n_rows, 28] prompt-mean delta) or None."""
    f = task_dir / f"{arm}_delta_logp.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    delta = z["delta_logp"]                       # [rows, 28, n]
    if metric == "test40":
        delta = delta[:, :, z["split"] == "test"]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(delta, axis=2)
    return [str(r) for r in z["row_names"]], mean


def render(ax, grid, row_names, vmax, title, annotate=False):
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels([ROW_TITLES.get(r, r) for r in row_names], fontsize=8)
    ax.set_xticks(range(0, grid.shape[1], 4))
    ax.set_xlabel("start edit layer L (ablate h.b for all b >= L)", fontsize=8)
    ax.set_title(title, fontsize=9)
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
    suffix = "" if args.metric == "all170" else f"_{args.metric}"

    task_dirs = sorted(d for d in args.root.iterdir()
                       if d.is_dir() and d.name != "figures")
    tasks = [d.name for d in task_dirs]

    # per (arm) -> task -> (row_names, [rows, 28])
    per_arm = {arm: {} for arm in ARMS}
    for d in task_dirs:
        for arm in ARMS:
            got = load_arm(d, arm, args.metric)
            if got is not None:
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
    print(f"shared scale vmax={vmax:.3f}  (metric={args.metric})")

    # --- per-arm figures ---
    for arm, (row_names, grid) in arm_grids.items():
        fig, ax = plt.subplots(figsize=(9, 0.6 * len(row_names) + 1.6))
        im = render(ax, grid, row_names, vmax,
                    f"{ARM_TITLES[arm]} — mean Δ log p(correct) over {len(per_arm[arm])} tasks",
                    annotate=args.annotate)
        fig.colorbar(im, ax=ax, label="Δ log p (ablated − clean)")
        fig.tight_layout()
        out = fig_dir / f"heatmap_{arm}{suffix}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")

    # --- combined 2x3 figure ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 6.5))
    for k, arm in enumerate(ARMS):
        ax = axes[k // 3][k % 3]
        if arm not in arm_grids:
            ax.axis("off")
            continue
        row_names, grid = arm_grids[arm]
        im = render(ax, grid, row_names, vmax, ARM_TITLES[arm])
    fig.suptitle(f"1-shot projection-ablation: mean Δ log p(correct answer), {len(tasks)} tasks "
                 f"({args.metric})", fontsize=12)
    fig.colorbar(im, ax=axes, label="Δ log p (ablated − clean)", shrink=0.8)
    out = fig_dir / f"heatmap_all_arms{suffix}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")

    # --- per-task supplementary grid ---
    fig, axes = plt.subplots(len(tasks), len(ARMS),
                             figsize=(4.0 * len(ARMS), 1.9 * len(tasks)), squeeze=False)
    for ti, task in enumerate(tasks):
        for ai, arm in enumerate(ARMS):
            ax = axes[ti][ai]
            if task not in per_arm[arm]:
                ax.axis("off")
                continue
            row_names, grid = per_arm[arm][task]
            ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks(range(len(row_names)))
            ax.set_yticklabels(row_names, fontsize=5)
            if ti == 0:
                ax.set_title(ARM_TITLES[arm], fontsize=7)
            if ai == 0:
                ax.set_ylabel(task, fontsize=7)
    fig.suptitle(f"per-task Δ log p heatmaps ({args.metric}, shared scale ±{vmax:.2f})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = fig_dir / f"per_task_grid{suffix}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
