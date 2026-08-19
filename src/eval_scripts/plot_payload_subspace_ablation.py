#!/usr/bin/env python
"""Heatmaps for the payload-subspace 1-shot ablation (CPU).

Thin wrapper around plot_oneshot_preimage_ablation.render/load_arm with the payload arm set:
2x2 grid — columns = op (zero | mean-clamp), rows = subspace (same task | counterfactual) —
one shared symmetric RdBu_r scale, per-panel min annotation. Reads the npz written by
ablate_oneshot_payload_subspace_logprob.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.eval_scripts.plot_oneshot_preimage_ablation import load_arm, render
from utils.paths import FV_FORMATION_DIR

ARMS = ["payload_zero", "payload_mean", "payload_cf_zero", "payload_cf_mean"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=FV_FORMATION_DIR / "ablation/attention_head_mechanisms/train_tasks")
    p.add_argument("--task", type=str, default="present-past")
    p.add_argument("--avg_tasks", nargs="+", default=None,
                   help="Task-AVERAGED mode: nanmean the per-task grids across these tasks "
                        "per arm and render one 2x2 payload figure (no fv figure).")
    p.add_argument("--suffix", type=str, default="")
    return p.parse_args()


def plot_avg(root, tasks, suffix=""):
    """Task-averaged 2x2 payload figure (Stream W heatmap_all_arms pattern)."""
    grids, row_names = {}, None
    for arm in ARMS:
        per_task = []
        for task in tasks:
            got = load_arm(root / task, arm)
            if got is None:
                print(f"[avg] missing {task}/{arm} (skipped)")
                continue
            row_names, g = got
            per_task.append(g)
        if per_task:
            grids[arm] = np.nanmean(np.stack(per_task), axis=0)
    if not grids:
        raise SystemExit(f"no arm npz found under {root} for {tasks}")
    vmax = max(np.nanmax(np.abs(g)) for g in grids.values())
    titles = {
        "payload_zero": "own payload subspace — project to 0",
        "payload_mean": "own payload subspace — clamp to 20-task mean",
        "payload_cf_zero": "shuffled task's subspace (cf) — project to 0",
        "payload_cf_mean": "shuffled task's subspace (cf) — clamp to 20-task mean",
    }
    # zero-only runs (e.g. propagated mode) get a 2x1 layout instead of blank panels
    if any(a.endswith("_mean") for a in grids):
        order = [["payload_zero", "payload_mean"], ["payload_cf_zero", "payload_cf_mean"]]
    else:
        order = [["payload_zero"], ["payload_cf_zero"]]
    ncols = len(order[0])
    fig, axes = plt.subplots(2, ncols, figsize=(6.75 * ncols + 0.5, 7.8), squeeze=False,
                             constrained_layout=True)
    im = None
    for r in range(2):
        for c in range(ncols):
            arm = order[r][c]
            ax = axes[r][c]
            if arm not in grids:
                ax.axis("off")
                continue
            g = grids[arm]
            im = render(ax, g, row_names, vmax, titles[arm],
                        show_xlabel=(r == 1), show_ylabels=(c == 0))
            ax.text(0.99, 0.02, f"min {np.nanmin(g):.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
    fig.colorbar(im, ax=[a for row in axes for a in row], shrink=0.85,
                 label="log p(ablated) − log p(clean)")
    fig.suptitle("1-shot ablation of the attention_head_payload_subspace —\n"
                 f"mean over {len(tasks)} test tasks, Δ log p(correct answer)"
                 if ncols == 1 else
                 f"1-shot ablation of the attention_head_payload_subspace — "
                 f"mean over {len(tasks)} test tasks, Δ log p(correct answer)",
                 fontsize=10 if ncols == 1 else 13)
    out = root / "figures" / f"heatmap_payload_arms_test{len(tasks)}avg{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def plot_fv_arms(root, tasks, suffix=""):
    """2x2: columns = task prompts, rows = {same-task FV, cf FV}; each panel = 3 site rows."""
    panels = {}
    for task in tasks:
        for arm in ("fv_zero", "fv_cf_zero"):
            got = load_arm(root / task, arm)
            if got is None:
                print(f"[fv figure] missing {task}/{arm} (skipped)")
                continue
            z = np.load(root / task / f"{arm}_delta_logp.npz", allow_pickle=False)
            other = str(z["cf_task"]) if arm == "fv_cf_zero" else task
            title = f"{task} prompts — {other} FV" + (" (counterfactual)" if arm == "fv_cf_zero" else "")
            panels[(task, arm)] = (got[0], got[1], title)
    if not panels:
        print("[fv figure] no fv arm npz found; skipping")
        return
    vmax = max(np.nanmax(np.abs(g)) for _, g, _ in panels.values())
    fig, axes = plt.subplots(2, len(tasks), figsize=(6.8 * len(tasks), 7.8),
                             squeeze=False, constrained_layout=True)
    im = None
    for c, task in enumerate(tasks):
        for r, arm in enumerate(("fv_zero", "fv_cf_zero")):
            ax = axes[r][c]
            if (task, arm) not in panels:
                ax.axis("off")
                continue
            row_names, g, title = panels[(task, arm)]
            im = render(ax, g, row_names, vmax, title, show_xlabel=(r == 1),
                        show_ylabels=(c == 0))
            ax.text(0.99, 0.02, f"min {np.nanmin(g):.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
    fig.colorbar(im, ax=[a for row in axes for a in row], shrink=0.85,
                 label="log p(ablated) − log p(clean)")
    fig.suptitle("1-shot ablation of the canonical FV direction (project to 0, "
                 "site token, blocks b ≥ L)", fontsize=13)
    out = root / "figures" / f"heatmap_fv_arms{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def main():
    args = parse_args()
    if args.avg_tasks is not None:
        plot_avg(args.root, list(args.avg_tasks), args.suffix)
        return
    task_dir = args.root / args.task
    import json
    cfg = json.loads((args.root / f"run_config_{args.task}.json").read_text())
    cf_task = cfg.get("cf_task", "cf")
    k = cfg.get("subspaces", {}).get(args.task, {}).get("k", "?")
    titles = {
        "payload_zero": f"{args.task} payload subspace — project to 0",
        "payload_mean": f"{args.task} payload subspace — clamp to 20-task mean",
        "payload_cf_zero": f"{cf_task} subspace (counterfactual) — project to 0",
        "payload_cf_mean": f"{cf_task} subspace (counterfactual) — clamp to 20-task mean",
    }

    grids, row_names = {}, None
    for arm in ARMS:
        got = load_arm(task_dir, arm)
        if got is None:
            print(f"missing arm npz: {arm} (skipped)")
            continue
        row_names, grids[arm] = got
    if not grids:
        raise SystemExit(f"no arm npz found under {task_dir}")
    vmax = max(np.nanmax(np.abs(g)) for g in grids.values())

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.8), squeeze=False,
                             constrained_layout=True)
    order = [["payload_zero", "payload_mean"], ["payload_cf_zero", "payload_cf_mean"]]
    im = None
    for r in range(2):
        for c in range(2):
            arm = order[r][c]
            ax = axes[r][c]
            if arm not in grids:
                ax.axis("off")
                continue
            g = grids[arm]
            im = render(ax, g, row_names, vmax, titles[arm],
                        show_xlabel=(r == 1), show_ylabels=(c == 0))
            mn = np.nanmin(g)
            ax.text(0.99, 0.02, f"min {mn:.2f}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
    fig.colorbar(im, ax=[a for row in axes for a in row], shrink=0.85,
                 label="log p(ablated) − log p(clean)")
    fig.suptitle(f"1-shot ablation of the {k}D attention_head_payload_subspace — "
                 f"{args.task} prompts, mean Δ log p(correct answer)", fontsize=13)

    fig_dir = args.root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"heatmap_payload_arms_{args.task}{args.suffix}.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    plot_fv_arms(args.root, [args.task, cf_task])


if __name__ == "__main__":
    main()
