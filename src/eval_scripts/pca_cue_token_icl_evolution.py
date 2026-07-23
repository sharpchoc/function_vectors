#!/usr/bin/env python
"""PCA visualization of L13 cue-token (pre-label) representations across ICL positions.

For every task (20 train + 7 test, the full-dim ridge study's task set) this loads the layer-13
residual activation at each ICL example's PRE-LABEL token (the cue right before the label) for
all 170 prompts (train+test splits pooled), then projects everything into two top-PC spaces fit
on the TRAIN tasks only:

  * pca_all_positions — PCA fit on the pooled activations across ALL ICL positions
    (20 tasks x 10 positions x 170 prompts).
  * pca_final_cue     — PCA fit on the final cue position only (icl10/pre; 20 tasks x 170).

Per variant it writes, for each ICL position, a 1x3 figure of the PC1-PC2 / PC1-PC3 / PC2-PC3
scatters (per-prompt points colored by task + a larger task-mean marker; train tasks 'o', test
tasks '^'), a positions-by-pairs grid figure, and the full projected coordinates
(projections.npz + CSVs) so 3D or alternate-pair views can be replotted without reloading
activations.

Layer axis convention matches the ridge study: index 0 = token embeddings, so --layer_index 13
is the output of transformer block 12.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR
from eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
    QUERY_ICL_INDEX,
    load_json,
    load_role_activations_all_layers,
    write_json,
)

CUE_ROLE = "pre_label_token"
VARIANTS = ["pca_all_positions", "pca_final_cue"]
PC_PAIRS = [(0, 1), (0, 2), (1, 2)]


def parse_args():
    p = argparse.ArgumentParser(description="PCA of layer-13 cue-token activations across ICL positions.")
    p.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train", "test"],
                   help="Activation splits pooled into the 170 rows per task/position.")
    p.add_argument("--layer_index", type=int, default=13,
                   help="Residual layer index (0 = embeddings, so 13 = output of block 12).")
    p.add_argument("--icl_indices", nargs="+", type=int, default=list(range(1, QUERY_ICL_INDEX + 1)))
    p.add_argument("--n_components", type=int, default=10,
                   help="PCs to fit/save (top 3 are plotted).")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--train_tasks", nargs="+", default=None, help="Override train tasks (smoke tests).")
    p.add_argument("--test_tasks", nargs="+", default=None,
                   help="Override test tasks. Default: the ridge study's 7 (9 minus cc/pc).")
    p.add_argument("--alpha", type=float, default=0.25, help="Per-prompt point alpha.")
    p.add_argument("--point_size", type=float, default=6.0)
    p.add_argument("--mean_point_size", type=float, default=130.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=Path, default=FV_FORMATION_DIR / "pca_cue_token_icl_evolution")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_cue_matrix(activations_root, task, splits, icl_index, layer_index):
    """[n_rows, 4096] float32 cue-token activations at one layer, with per-row split labels."""
    parts = []
    split_labels = []
    for split in splits:
        a = load_role_activations_all_layers(activations_root, task, split, CUE_ROLE, icl_index)
        parts.append(a[:, layer_index, :].float())
        split_labels.extend([split] * a.shape[0])
    return torch.cat(parts, dim=0).numpy().astype(np.float32), split_labels


def build_task_colors(tasks):
    palettes = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    colors = []
    for palette in palettes:
        colors.extend(palette(i) for i in range(palette.N))
    return {task: colors[i % len(colors)] for i, task in enumerate(tasks)}


def pca_artifact(pca, fit_tasks, extra):
    return {
        "components": torch.from_numpy(pca.components_).float(),
        "mean": torch.from_numpy(pca.mean_).float(),
        "explained_variance": torch.from_numpy(pca.explained_variance_).float(),
        "explained_variance_ratio": torch.from_numpy(pca.explained_variance_ratio_).float(),
        "fit_tasks": fit_tasks,
        **extra,
    }


def pca_json_summary(pca, fit_tasks, extra):
    return {
        "n_components": int(pca.n_components_),
        "input_feature_count": int(pca.n_features_in_),
        "fit_tasks": fit_tasks,
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        **extra,
    }


def variance_label(pca, pc):
    return f"PC{pc + 1} ({pca.explained_variance_ratio_[pc] * 100:.1f}% var)"


def draw_panel(ax, coords, tasks, task_group, task_colors, icl_index, pcx, pcy, args):
    for task in tasks:
        pts = coords[(task, icl_index)]
        ax.scatter(pts[:, pcx], pts[:, pcy], s=args.point_size, alpha=args.alpha,
                   color=task_colors[task], linewidths=0, rasterized=True, zorder=1)
    for task in tasks:
        mean = coords[(task, icl_index)].mean(axis=0)
        marker = "o" if task_group[task] == "train" else "^"
        ax.scatter([mean[pcx]], [mean[pcy]], s=args.mean_point_size, marker=marker,
                   color=task_colors[task], edgecolors="black", linewidths=1.0, zorder=3)


def legend_handles(tasks, task_group, task_colors, mean_size):
    handles = [
        Line2D([], [], linestyle="", marker="o" if task_group[t] == "train" else "^",
               markersize=7, markerfacecolor=task_colors[t], markeredgecolor="black",
               markeredgewidth=0.6, label=f"{t} ({task_group[t]})")
        for t in tasks
    ]
    handles.append(Line2D([], [], linestyle="", marker="o", markersize=7, markerfacecolor="white",
                          markeredgecolor="black", label="task mean (train)"))
    handles.append(Line2D([], [], linestyle="", marker="^", markersize=7, markerfacecolor="white",
                          markeredgecolor="black", label="task mean (test)"))
    return handles


def axis_limits(coords, icl_indices, tasks, n_pcs=3, lo_pct=1.0, hi_pct=99.0, pad_frac=0.06):
    stacked = np.concatenate([coords[(t, icl)] for t in tasks for icl in icl_indices], axis=0)
    limits = []
    for pc in range(n_pcs):
        lo, hi = np.percentile(stacked[:, pc], [lo_pct, hi_pct])
        pad = (hi - lo) * pad_frac
        limits.append((lo - pad, hi + pad))
    return limits


def main():
    args = parse_args()
    np.random.seed(args.seed)
    t_start = time.time()

    manifest = load_json(args.task_manifest)
    train_tasks = sorted(args.train_tasks) if args.train_tasks is not None else sorted(manifest["train_tasks"])
    test_tasks = sorted(args.test_tasks) if args.test_tasks is not None else sorted(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)
    overlap = sorted(set(train_tasks).intersection(test_tasks))
    if overlap:
        raise ValueError(f"Tasks cannot be both train and test: {overlap}")
    all_tasks = train_tasks + test_tasks
    task_group = {t: "train" for t in train_tasks}
    task_group.update({t: "test" for t in test_tasks})

    icl_indices = sorted(args.icl_indices)
    final_icl = icl_indices[-1]
    if "pca_final_cue" in args.variants and final_icl != QUERY_ICL_INDEX:
        print(f"WARNING: final cue position is icl{final_icl}, not icl{QUERY_ICL_INDEX}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = args.output_dir / "run_config.json"
    if run_config_path.exists() and not args.overwrite:
        raise FileExistsError(f"{run_config_path} exists; pass --overwrite to replace.")

    # ---- Load: [(task, icl)] -> [n_rows, 4096] at the requested layer. -------------------------
    data = {}
    split_labels = {}
    for icl in icl_indices:
        if icl == QUERY_ICL_INDEX:
            root = args.query_activations_root
        else:
            root = Path(args.icl_activations_root_template.format(icl=icl))
        t0 = time.time()
        for task in all_tasks:
            data[(task, icl)], split_labels[(task, icl)] = load_cue_matrix(
                root, task, args.splits, icl, args.layer_index)
        n_rows = {data[(t, icl)].shape[0] for t in all_tasks}
        print(f"[load] icl{icl:02d}: {len(all_tasks)} tasks, rows/task={sorted(n_rows)} "
              f"({time.time() - t0:.1f}s)", flush=True)

    task_colors = build_task_colors(all_tasks)
    run_config = {
        "script": Path(__file__).name,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "icl_indices": icl_indices,
        "cue_role": CUE_ROLE,
        "layer_index": args.layer_index,
        "row_order_note": "rows per (task, icl) follow --splits order (train shards then test shards); "
                          "row_index is the position in that pooled order",
        "variants": {},
    }

    # ---- Fit + project + save + plot per variant. -----------------------------------------------
    for variant in args.variants:
        vdir = args.output_dir / variant
        fig_dir = vdir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        if variant == "pca_all_positions":
            fit_icls = icl_indices
        else:  # pca_final_cue
            fit_icls = [final_icl]
        x_fit = np.concatenate([data[(t, icl)] for t in train_tasks for icl in fit_icls], axis=0)
        pca = PCA(n_components=args.n_components, random_state=args.seed)
        pca.fit(x_fit)
        evr3 = pca.explained_variance_ratio_[:3]
        print(f"[{variant}] fit on {x_fit.shape} | top-3 var ratio = "
              f"{evr3[0]:.3f}/{evr3[1]:.3f}/{evr3[2]:.3f}", flush=True)

        coords = {key: pca.transform(mat).astype(np.float32) for key, mat in data.items()}

        extra = {"variant": variant, "fit_icl_indices": fit_icls, "cue_role": CUE_ROLE,
                 "layer_index": args.layer_index, "fit_shape": list(x_fit.shape)}
        torch.save(pca_artifact(pca, train_tasks, extra), vdir / "pca_model.pt")
        write_json(vdir / "pca_model.json", pca_json_summary(pca, train_tasks, extra))

        # Full projected coordinates (3D-ready): npz + per-point CSV + task-mean CSV.
        rows_task, rows_group, rows_icl, rows_split, rows_ridx, rows_coords = [], [], [], [], [], []
        for task in all_tasks:
            for icl in icl_indices:
                c = coords[(task, icl)]
                rows_coords.append(c)
                rows_task.extend([task] * c.shape[0])
                rows_group.extend([task_group[task]] * c.shape[0])
                rows_icl.extend([icl] * c.shape[0])
                rows_split.extend(split_labels[(task, icl)])
                rows_ridx.extend(range(c.shape[0]))
        all_coords = np.concatenate(rows_coords, axis=0)
        np.savez_compressed(
            vdir / "projections.npz",
            coords=all_coords,
            task=np.array(rows_task),
            group=np.array(rows_group),
            icl_index=np.array(rows_icl, dtype=np.int64),
            split=np.array(rows_split),
            row_index=np.array(rows_ridx, dtype=np.int64),
        )
        pc_cols = [f"pc{i + 1}" for i in range(3)]
        with open(vdir / "projection_points.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task", "group", "icl_index", "split", "row_index"] + pc_cols)
            for i in range(all_coords.shape[0]):
                writer.writerow([rows_task[i], rows_group[i], rows_icl[i], rows_split[i], rows_ridx[i]]
                                + [f"{v:.6g}" for v in all_coords[i, :3]])
        mean_cols = [f"pc{i + 1}" for i in range(args.n_components)]
        with open(vdir / "task_means.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task", "group", "icl_index"] + mean_cols)
            for task in all_tasks:
                for icl in icl_indices:
                    mean = coords[(task, icl)].mean(axis=0)
                    writer.writerow([task, task_group[task], icl] + [f"{v:.6g}" for v in mean])

        limits = axis_limits(coords, icl_indices, all_tasks)
        handles = legend_handles(all_tasks, task_group, task_colors, args.mean_point_size)

        # Per-position 1x3 pair figures.
        for icl in icl_indices:
            fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
            for ax, (pcx, pcy) in zip(axes, PC_PAIRS):
                draw_panel(ax, coords, all_tasks, task_group, task_colors, icl, pcx, pcy, args)
                ax.set_xlim(*limits[pcx])
                ax.set_ylim(*limits[pcy])
                ax.set_xlabel(variance_label(pca, pcx))
                ax.set_ylabel(variance_label(pca, pcy))
            pos_name = f"icl{icl:02d}/pre"
            fig.suptitle(f"{variant} | L{args.layer_index} cue tokens | position {pos_name}", fontsize=13)
            fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
                       fontsize=7, frameon=False)
            fig.tight_layout(rect=(0, 0, 0.99, 0.95))
            for ext in ("png", "pdf"):
                fig.savefig(fig_dir / f"icl{icl:02d}_pc_pairs.{ext}", dpi=170, bbox_inches="tight")
            plt.close(fig)

        # Grid: rows = ICL positions, cols = PC pairs.
        n_rows_fig = len(icl_indices)
        fig, axes = plt.subplots(n_rows_fig, 3, figsize=(15.0, 3.6 * n_rows_fig), squeeze=False)
        for r, icl in enumerate(icl_indices):
            for c, (pcx, pcy) in enumerate(PC_PAIRS):
                ax = axes[r][c]
                draw_panel(ax, coords, all_tasks, task_group, task_colors, icl, pcx, pcy, args)
                ax.set_xlim(*limits[pcx])
                ax.set_ylim(*limits[pcy])
                if r == n_rows_fig - 1:
                    ax.set_xlabel(variance_label(pca, pcx))
                if c == 0:
                    ax.set_ylabel(f"icl{icl:02d}/pre\n{variance_label(pca, pcy)}")
                else:
                    ax.set_ylabel(variance_label(pca, pcy), fontsize=8)
        fig.suptitle(f"{variant} | L{args.layer_index} cue-token PCA across ICL positions "
                     f"(fit on {len(train_tasks)} train tasks)", fontsize=14, y=1.0)
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0, 0.99, 0.99))
        fig.savefig(fig_dir / "grid_positions_by_pcpairs.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        run_config["variants"][variant] = {
            "fit_icl_indices": fit_icls,
            "fit_shape": list(x_fit.shape),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "output_dir": str(vdir),
        }
        print(f"[{variant}] wrote {vdir}", flush=True)

    write_json(run_config_path, run_config)
    print(f"Done in {time.time() - t_start:.1f}s -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
