#!/usr/bin/env python
"""Task-space confusion matrices for the saved full-dim ridge weight matrices.

For each saved (icl, role, layer) cell: push every task's MEAN activation through the map and
compare the predicted FV against every task's true FV by cosine similarity, both centered on the
train-FV mean ybar (the ridge intercept), so the plot shows what the learned map contributes
beyond predicting the mean FV. Rows = task whose activation goes in; columns = candidate task FV;
a bright diagonal = the map recovers task identity. Train tasks come first (alphabetical), then
test tasks (red bold labels, separated by divider lines); a black dot marks each row's argmax.

Requires the weight banks saved by plot_fulldim_ridge_weight_heatmaps.py.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR
from eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
    QUERY_ICL_INDEX,
    load_function_vector,
    load_json,
    load_task_role_pooled,
    role_load_icl_index,
)
from eval_scripts.plot_fulldim_ridge_weight_heatmaps import DEFAULT_CELLS, ROLE_SHORT, cell_key

TEST_COLOR = "#d62728"


def parse_args():
    p = argparse.ArgumentParser(description="Task-space confusion matrices for ridge weight banks.")
    p.add_argument("--cells", nargs="+", default=None,
                   help="Cells as icl:role:layer. Default: the 6 study cells.")
    p.add_argument("--weights_dir", type=Path, default=ARTIFACTS_ROOT / "fulldim_ridge_weight_matrices")
    p.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_selected")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--output_dir", type=Path, default=FV_FORMATION_DIR / "activation_to_fv_decoding/fulldim_ridge/weight_heatmaps")
    return p.parse_args()


def cosine(a, b):
    return float(torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b) + 1e-12))


def main():
    args = parse_args()
    cells = DEFAULT_CELLS if args.cells is None else [
        (int(i), r, int(l)) for i, r, l in (c.split(":") for c in args.cells)]

    manifest = load_json(args.task_manifest)
    train_tasks = sorted(manifest["train_tasks"])
    test_tasks = sorted(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)
    tasks = train_tasks + test_tasks
    n_train = len(train_tasks)
    fvs = {t: load_function_vector(args.fv_root, t) for t in tasks}

    matrices = {}
    for icl, role, layer in cells:
        key = cell_key(icl, role, layer)
        bank = torch.load(args.weights_dir / f"{key}.pt", map_location="cpu", weights_only=False)
        w, ybar = bank["weight"], bank["ybar"]
        fmean, fstd, xbar = bank["feature_mean"], bank["feature_std"], bank["xbar"]
        root = args.query_activations_root if icl == QUERY_ICL_INDEX else Path(
            args.icl_activations_root_template.format(icl=icl))
        load_icl = role_load_icl_index(role, icl)

        cos = np.zeros((len(tasks), len(tasks)))
        for i, task in enumerate(tasks):
            acts = load_task_role_pooled(root, task, args.splits, role, load_icl)[:, layer, :].to(torch.float32)
            x_std = (acts.mean(dim=0) - fmean) / fstd
            pred_centered = (x_std - xbar) @ w          # the map's contribution beyond ybar
            for j, cand in enumerate(tasks):
                cos[i, j] = cosine(pred_centered, fvs[cand] - ybar)
        matrices[key] = cos
        diag_test = np.mean([cos[i, i] for i in range(n_train, len(tasks))])
        print(f"{key}: mean diag cos train={np.mean(np.diag(cos)[:n_train]):.3f} test={diag_test:.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(20, 13.5), sharex=True, sharey=True)
    for ax, (icl, role, layer) in zip(axes.flat, cells):
        key = cell_key(icl, role, layer)
        cos = matrices[key]
        im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", interpolation="nearest")
        ax.scatter(np.argmax(cos, axis=1), np.arange(len(tasks)), s=8, c="black", marker=".", zorder=3)
        ax.axhline(n_train - 0.5, color="black", linewidth=1.2)
        ax.axvline(n_train - 0.5, color="black", linewidth=1.2)
        ax.set_title(f"icl{icl:02d}/{ROLE_SHORT[role]}  L{layer}", fontsize=11)
        ax.set_xticks(range(len(tasks)))
        ax.set_yticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=90, fontsize=5.5)
        ax.set_yticklabels(tasks, fontsize=5.5)
        for k, lbl in enumerate(ax.get_xticklabels()):
            if k >= n_train:
                lbl.set_color(TEST_COLOR); lbl.set_fontweight("bold")
        for k, lbl in enumerate(ax.get_yticklabels()):
            if k >= n_train:
                lbl.set_color(TEST_COLOR); lbl.set_fontweight("bold")
        ax.tick_params(length=0)
    for ax in axes[-1]:
        ax.set_xlabel("candidate task FV", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("task whose mean activation is mapped", fontsize=9)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("cos(predicted FV - ybar, true FV - ybar)")
    fig.suptitle("GPT-J full-dim ridge: task-space confusion of the learned maps\n"
                 f"rows/cols: {n_train} train tasks then {len(test_tasks)} TEST tasks "
                 "(red bold, beyond the divider); black dot = row argmax", fontsize=13)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "task_confusion_6cells.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(args.weights_dir / "task_confusion_matrices.npz",
             tasks=np.array(tasks), n_train=n_train, **matrices)
    stats = {k: {"mean_diag_cos_train": float(np.mean(np.diag(m)[:n_train])),
                 "mean_diag_cos_test": float(np.mean(np.diag(m)[n_train:])),
                 "top1_accuracy_test": float(np.mean(np.argmax(m[n_train:], axis=1)
                                                     == np.arange(n_train, len(tasks))))}
             for k, m in matrices.items()}
    with open(args.output_dir / "task_confusion_summary.json", "w") as f:
        json.dump({"tasks": tasks, "n_train": n_train, "cells": stats}, f, indent=2)
    print(f"Wrote {out} + task_confusion_summary.json (+ matrices npz in {args.weights_dir})")


if __name__ == "__main__":
    main()
