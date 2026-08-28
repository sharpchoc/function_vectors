#!/usr/bin/env python
"""Effective rank of raw residual activations by (layer, token position), pooled and per task.

For each requested token position (ICL example x role) and layer, stack the 170-prompt
activations of all manifest tasks into one matrix (and per task into [170, 4096] matrices),
MEAN-CENTER (subtract the matrix's mean activation vector), and compute from the singular
values:

  * stable rank            ||A||_F^2 / sigma_1^2
  * rank90                 smallest k with sum_{i<=k} sigma_i^2 / sum sigma_i^2 >= threshold
                           (= PCA explained-variance rank)
  * participation ratio    (sum sigma_i^2)^2 / sum sigma_i^4

Outputs (grid-only PNG policy): pooled 1x3 line figure (one line per position), three 29-panel
per-task grids, both metric CSVs, and the full singular-value spectra (npz) for replotting.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR
from eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    QUERY_ICL_INDEX,
    load_json,
    load_task_role_pooled,
    role_load_icl_index,
    write_json,
)

ROLE_SHORT = {"pre_label_token": "pre", "last_label_token": "label", "first_label_token": "first"}
ROLE_STYLE = {"pre_label_token": "-", "last_label_token": "--", "first_label_token": ":"}
ICL_COLORS = {1: "tab:blue", 2: "tab:orange", 9: "tab:green", 10: "tab:red"}
METRICS = [("rank90", "rank90"), ("stable_rank", "stable rank"),
           ("participation_ratio", "participation ratio")]


def parse_args():
    p = argparse.ArgumentParser(description="Stable rank / rank90 / PR of activations by (layer, position).")
    p.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--tasks", nargs="+", default=None, help="Override tasks (smoke tests).")
    p.add_argument("--icl_indices", nargs="+", type=int, default=[1, 2, 9, 10])
    p.add_argument("--token_roles", nargs="+", default=["pre_label_token", "last_label_token"],
                   choices=sorted(ROLE_SHORT))
    p.add_argument("--layers", nargs="+", type=int, default=list(range(6, 21)),
                   help="Layer indices on the 29-length axis (0 = embeddings).")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--energy_threshold", type=float, default=0.90)
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--output_dir", type=Path, default=FV_FORMATION_DIR / "activation_geometry/activation_rank_by_position")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sv_metrics(mat, threshold):
    """Center columns, return (singular values, stable_rank, rank90, participation_ratio)."""
    centered = mat - mat.mean(dim=0)
    sv = torch.linalg.svdvals(centered)
    energy = (sv ** 2).double()
    total = energy.sum()
    stable_rank = float(total / energy[0])
    cum = torch.cumsum(energy, dim=0) / total
    rank90 = int(torch.searchsorted(cum, threshold).item()) + 1
    pr = float(total ** 2 / (energy ** 2).sum())
    return sv.numpy().astype(np.float32), stable_rank, rank90, pr


def pos_name(icl, role):
    return f"icl{icl:02d}/{ROLE_SHORT[role]}"


def main():
    args = parse_args()
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.output_dir / "run_config.json"
    if cfg_path.exists() and not args.overwrite:
        raise FileExistsError(f"{cfg_path} exists; pass --overwrite to replace.")

    manifest = load_json(args.task_manifest)
    if args.tasks is not None:
        tasks = sorted(args.tasks)
    else:
        tasks = sorted(manifest["train_tasks"]) + sorted(manifest["test_tasks"])
    layers = sorted(args.layers)
    icl_indices = sorted(args.icl_indices)
    positions = [(icl, role) for icl in icl_indices for role in args.token_roles]

    # ---- Load: acts[(icl, role, task)] = [170, n_layers_kept, 4096] fp32 ------------------------
    acts = {}
    for icl in icl_indices:
        root = args.query_activations_root if icl == QUERY_ICL_INDEX else Path(
            args.icl_activations_root_template.format(icl=icl))
        t1 = time.time()
        for role in args.token_roles:
            load_icl = role_load_icl_index(role, icl)
            for task in tasks:
                a = load_task_role_pooled(root, task, args.splits, role, load_icl)
                acts[(icl, role, task)] = a[:, layers, :].float()
        print(f"[load] icl{icl:02d}: {len(tasks)} tasks x {len(args.token_roles)} roles "
              f"({time.time() - t1:.0f}s)", flush=True)

    pooled_rows, per_task_rows = [], []
    spectra = {}
    for icl, role in positions:
        pname = pos_name(icl, role)
        for li, layer in enumerate(layers):
            pooled = torch.cat([acts[(icl, role, t)][:, li, :] for t in tasks], dim=0)
            sv, sr, r90, pr = sv_metrics(pooled, args.energy_threshold)
            spectra[f"pooled_{pname.replace('/', '_')}_L{layer:02d}"] = sv
            pooled_rows.append({"position": pname, "icl_index": icl, "token_role": role,
                                "layer": layer, "n_rows": int(pooled.shape[0]),
                                "stable_rank": sr, "rank90": r90, "participation_ratio": pr})
            for task in tasks:
                svt, srt, r90t, prt = sv_metrics(acts[(icl, role, task)][:, li, :],
                                                 args.energy_threshold)
                spectra[f"task_{task}_{pname.replace('/', '_')}_L{layer:02d}"] = svt
                per_task_rows.append({"task": task, "position": pname, "icl_index": icl,
                                      "token_role": role, "layer": layer,
                                      "stable_rank": srt, "rank90": r90t,
                                      "participation_ratio": prt})
        print(f"[svd] {pname}: {len(layers)} layers done ({time.time() - t0:.0f}s)", flush=True)

    for name, rows in (("pooled_metrics.csv", pooled_rows), ("per_task_metrics.csv", per_task_rows)):
        with open(args.output_dir / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(args.output_dir / "singular_values.npz", **spectra)

    # ---- Pooled figure: 1x3 panels, one line per position. -------------------------------------
    def line_label(icl, role):
        return f"icl{icl:02d} {ROLE_SHORT[role]}"

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    for ax, (mkey, mlabel) in zip(axes, METRICS):
        for icl, role in positions:
            vals = {r["layer"]: r[mkey] for r in pooled_rows
                    if r["icl_index"] == icl and r["token_role"] == role}
            ax.plot(layers, [vals[l] for l in layers], color=ICL_COLORS[icl],
                    ls=ROLE_STYLE[role], marker="o", markersize=3.2, lw=1.5,
                    label=line_label(icl, role))
        ax.set_xlabel("layer index (0 = embeddings)")
        ax.set_ylabel(mlabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Effective rank of pooled activations ({len(tasks)} tasks x 170 prompts, "
                 f"mean-centered)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output_dir / "pooled_rank_vs_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Per-task grids: one figure per metric. -------------------------------------------------
    ncols = 5
    nrows = int(np.ceil(len(tasks) / ncols))
    for mkey, mlabel in METRICS:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 2.9 * nrows),
                                 squeeze=False, sharex=True, sharey=True)
        for i, task in enumerate(tasks):
            ax = axes[i // ncols][i % ncols]
            for icl, role in positions:
                vals = {r["layer"]: r[mkey] for r in per_task_rows
                        if r["task"] == task and r["icl_index"] == icl and r["token_role"] == role}
                ax.plot(layers, [vals[l] for l in layers], color=ICL_COLORS[icl],
                        ls=ROLE_STYLE[role], lw=1.1)
            ax.set_title(task, fontsize=9)
            ax.grid(alpha=0.25)
        for i in range(len(tasks), nrows * ncols):
            axes[i // ncols][i % ncols].axis("off")
        handles = [plt.Line2D([], [], color=ICL_COLORS[icl], ls=ROLE_STYLE[role],
                              label=line_label(icl, role))
                   for icl, role in positions]
        fig.legend(handles=handles, loc="upper right", ncol=4, fontsize=9, frameon=False)
        fig.suptitle(f"Per-task {mlabel} of activations by layer (170 prompts/task, mean-centered)",
                     fontsize=13)
        fig.supxlabel("layer index (0 = embeddings)")
        fig.supylabel(mlabel)
        fig.tight_layout(rect=(0.01, 0.01, 1, 0.95))
        fig.savefig(args.output_dir / f"per_task_{mkey}_grid.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    write_json(cfg_path, {
        "script": Path(__file__).name,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "tasks": tasks, "positions": [pos_name(i, r) for i, r in positions],
        "layers": layers, "n_prompts_per_task": 170,
        "centering": "column-mean subtracted per matrix before all metrics",
        "metrics": {"stable_rank": "frobenius^2 / sigma1^2",
                    "rank90": f"min k with cumulative energy >= {args.energy_threshold}",
                    "participation_ratio": "(sum s^2)^2 / sum s^4"},
    })
    print(f"done in {time.time() - t0:.0f}s -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
