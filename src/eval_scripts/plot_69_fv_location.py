#!/usr/bin/env python
"""Aggregate + plot the FV-location capture (capture_69_fv_location.py).

Loads the per-task (28, 32) prompt-averaged projection matrices for all 69 tasks and
averages across tasks (every task contributes 150 prompts, so the flat task mean equals
the prompt-weighted mean). One heatmap per metric — cosine and raw dot with the unit task
FV — layer (0..27) vs structural token position (demo1..10 x {input, cue, label} + query
{input, cue}).

Outputs (RESULTS/69_task_run/FV_location/):
  fv_location_heatmap.png   2-panel figure (cos / dot), all-69-task average
  fv_location.npz           per-task stacks + task list + columns (regenerate any view)
  summary_cos.csv, summary_dot.csv   the plotted all-task matrices (rows=layer)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, TASK69_RUN_DIR  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "fv_location")
    p.add_argument("--out_dir", type=Path, default=TASK69_RUN_DIR / "FV_location")
    return p.parse_args()


def col_label(c):
    d, kind = c.split("_", 1)
    d = d.replace("demo", "d").replace("query", "q")
    return f"{d} {'A:' if kind == 'cue' else ('in' if kind == 'input' else 'lab')}"


def main():
    args = parse_args()
    files = sorted(args.in_root.glob("*.npz"))
    assert len(files) == 69, f"expected 69 task files, found {len(files)}"
    tasks, groups, cos_stack, dot_stack = [], [], [], []
    columns = None
    for f in files:
        z = np.load(f, allow_pickle=False)
        tasks.append(f.stem)
        groups.append(str(z["group"]))
        cos_stack.append(z["cos_mean"])
        dot_stack.append(z["dot_mean"])
        cols = [str(c) for c in z["columns"]]
        assert columns is None or cols == columns
        columns = cols
    cos_stack, dot_stack = np.stack(cos_stack), np.stack(dot_stack)
    n_layers = cos_stack.shape[1]
    labels = [col_label(c) for c in columns]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "fv_location.npz",
             cos=cos_stack, dot=dot_stack, tasks=np.array(tasks),
             groups=np.array(groups), columns=np.array(columns))

    fig, axes = plt.subplots(2, 1, figsize=(13, 11))
    for ax, mat, name in ((axes[0], cos_stack.mean(0), "cosine  cos(z_l, v_A)"),
                          (axes[1], dot_stack.mean(0), "raw dot  z_l . v_A / ||v_A||")):
        vmax = np.abs(mat).max()
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(f"FV direction in the residual stream — {name}  "
                     f"(mean over 69 tasks x 150 ten-shot prompts)", fontsize=11)
        ax.set_ylabel("layer (block output)")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        for x in np.arange(2.5, 29.5, 3):  # demo boundaries
            ax.axvline(x, color="k", lw=0.4, alpha=0.35)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    axes[1].set_xlabel("token position (in = input tokens, A: = cue, lab = label tokens; "
                       "spans averaged)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "fv_location_heatmap.png", dpi=150)
    plt.close(fig)

    for name, mat in (("cos", cos_stack.mean(0)), ("dot", dot_stack.mean(0))):
        with open(args.out_dir / f"summary_{name}.csv", "w") as f:
            f.write("layer," + ",".join(columns) + "\n")
            for l in range(n_layers):
                f.write(f"{l}," + ",".join(f"{v:.5f}" for v in mat[l]) + "\n")
    print(f"wrote {args.out_dir} (69 tasks: {groups.count('train')} train / "
          f"{groups.count('heldout')} heldout)")


if __name__ == "__main__":
    main()
