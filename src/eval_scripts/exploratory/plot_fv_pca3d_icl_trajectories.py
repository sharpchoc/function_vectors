#!/usr/bin/env python
"""ICL trajectories of ridge-mapped test predictions in the 3D FV-PCA space.

For every ICL example index n = 1..10 and both the pre-label and last-label token roles, take
that position's best-test-MSE layer, fit (or load) the full-dim ridge weight bank exactly as in
plot_fulldim_ridge_weight_heatmaps.py, and push each TEST task's mean activation through the map.
Plotting all predictions in the top-3 PCA space of the 27 true FVs shows how the mapped estimate
of a held-out task's FV moves as demonstrations accumulate over the prompt (icl01 -> icl10).

Missing weight banks are fitted on the fly (refit test MSE is checked against combined_metrics)
and saved to the shared bank dir, so reruns are cheap.
"""
import argparse
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
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
    QUERY_ICL_INDEX,
    load_function_vector,
    load_json,
    load_task_role_pooled,
    role_load_icl_index,
)
from eval_scripts.exploratory.plot_fulldim_ridge_weight_heatmaps import (
    ROLE_SHORT,
    cell_key,
    fit_weight_matrix,
    lookup_metrics,
)

ROLES = ["pre_label_token", "last_label_token"]
ROLE_STYLE = {"pre_label_token": {"linestyle": "-", "marker": "o"},
              "last_label_token": {"linestyle": "--", "marker": "^"}}
TRAIN_COLOR = "#9db6cc"


def parse_args():
    p = argparse.ArgumentParser(description="ICL trajectories of mapped test FVs in FV-PCA space.")
    p.add_argument("--icl_indices", nargs="+", type=int, default=list(range(1, 11)))
    p.add_argument("--metrics_csv", type=Path,
                   default=FV_FORMATION_DIR / "activation_to_fv_decoding/fulldim_ridge/main/combined_metrics_with_r2.csv")
    p.add_argument("--weights_dir", type=Path, default=ARTIFACTS_ROOT / "fulldim_ridge_weight_matrices")
    p.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_selected")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--output_dir", type=Path, default=FV_FORMATION_DIR / "activation_to_fv_decoding/fulldim_ridge/weight_heatmaps")
    p.add_argument("--std_eps", type=float, default=1e-6)
    return p.parse_args()


def best_layers(metrics_csv, icl_indices, roles):
    """Best test-MSE layer + alpha per (icl, role) from the combined metrics."""
    import csv as _csv
    best = {}
    with open(metrics_csv) as f:
        for r in _csv.DictReader(f):
            key = (int(r["icl_example_index"]), r["token_role"])
            if key[0] in icl_indices and key[1] in roles:
                mse = float(r["test_mse"])
                if key not in best or mse < best[key][1]:
                    best[key] = (int(r["layer"]), mse)
    return {k: v[0] for k, v in best.items()}


def main():
    args = parse_args()
    manifest = load_json(args.task_manifest)
    train_tasks = list(manifest["train_tasks"])
    test_tasks = sorted(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)
    all_tasks = train_tasks + test_tasks
    fvs = {t: load_function_vector(args.fv_root, t) for t in all_tasks}

    layers = best_layers(args.metrics_csv, set(args.icl_indices), set(ROLES))
    cells = [(icl, role, layers[(icl, role)]) for icl in args.icl_indices for role in ROLES]
    stored = lookup_metrics(args.metrics_csv, cells)
    args.weights_dir.mkdir(parents=True, exist_ok=True)

    # Mapped test predictions per cell (fit bank on the fly if not saved).
    preds = {}   # (icl, role) -> [7, 4096]
    for icl, role, layer in cells:
        alpha, stored_mse, _ = stored[(icl, role, layer)]
        key = cell_key(icl, role, layer)
        bank_path = args.weights_dir / f"{key}.pt"
        root = args.query_activations_root if icl == QUERY_ICL_INDEX else Path(
            args.icl_activations_root_template.format(icl=icl))
        load_icl = role_load_icl_index(role, icl)
        t0 = time.time()

        if bank_path.exists():
            bank = torch.load(bank_path, map_location="cpu", weights_only=False)
            w, ybar = bank["weight"], bank["ybar"]
            fmean, fstd, xbar = bank["feature_mean"], bank["feature_std"], bank["xbar"]
            xs_test = {t: load_task_role_pooled(root, t, args.splits, role, load_icl)[:, layer, :]
                       .to(torch.float32) for t in test_tasks}
            xs_test = {t: (x - fmean) / fstd for t, x in xs_test.items()}
            print(f"{key}: loaded bank ({time.time()-t0:.0f}s)")
        else:
            xs = {t: load_task_role_pooled(root, t, args.splits, role, load_icl)[:, layer, :]
                  .to(torch.float32) for t in all_tasks}
            pool = torch.cat([xs[t] for t in train_tasks], dim=0)
            fmean, fstd = pool.mean(dim=0), pool.std(dim=0, unbiased=False).clamp_min(args.std_eps)
            xs = {t: (x - fmean) / fstd for t, x in xs.items()}
            w, xbar, ybar, test_mse = fit_weight_matrix(xs, fvs, train_tasks, test_tasks, alpha)
            if abs(test_mse - stored_mse) > 5e-4:
                raise AssertionError(f"{key}: refit test_mse {test_mse:.5f} != stored {stored_mse:.5f}")
            torch.save({"weight": w, "feature_mean": fmean, "feature_std": fstd, "xbar": xbar,
                        "ybar": ybar, "alpha": alpha, "icl": icl, "token_role": role, "layer": layer,
                        "orientation": "pred = ybar + xc @ weight; weight[in_dim, out_dim]"}, bank_path)
            xs_test = {t: xs[t] for t in test_tasks}
            print(f"{key}: fitted bank (test_mse {test_mse:.5f} == stored, {time.time()-t0:.0f}s)")

        preds[(icl, role)] = np.stack([
            (ybar + (xs_test[t].mean(dim=0) - xbar) @ w).numpy() for t in test_tasks])

    # PCA on the 27 true FVs.
    fv = torch.stack([fvs[t] for t in sorted(train_tasks) + test_tasks]).numpy()
    center = fv.mean(axis=0)
    _, s, vt = np.linalg.svd(fv - center, full_matrices=False)
    var_frac = (s ** 2) / np.sum(s ** 2)
    comps = vt[:3]
    fv3 = (fv - center) @ comps.T
    n_train = len(train_tasks)
    pred3 = {k: (p - center) @ comps.T for k, p in preds.items()}   # [7, 3] each

    task_colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(test_tasks)))
    fig = plt.figure(figsize=(19, 9))
    for pi, (elev, azim) in enumerate([(18, -60), (25, 120)]):
        ax = fig.add_subplot(1, 2, pi + 1, projection="3d")
        ax.scatter(*fv3[:n_train].T, c=TRAIN_COLOR, s=22, alpha=0.7, label="train FV (20)")
        for ti, task in enumerate(test_tasks):
            col = task_colors[ti]
            p = fv3[n_train + ti]
            ax.scatter(*p, color=col, s=90, marker="*", edgecolors="black", linewidths=0.5,
                       label=None, zorder=4)
            ax.text(*p, task, fontsize=7, color=col, fontweight="bold")
            for role in ROLES:
                traj = np.stack([pred3[(icl, role)][ti] for icl in args.icl_indices])
                st = ROLE_STYLE[role]
                ax.plot(*traj.T, color=col, linewidth=1.2, alpha=0.8, linestyle=st["linestyle"])
                ax.scatter(*traj.T, color=col, s=8 + 4 * np.arange(len(traj)), marker=st["marker"],
                           alpha=0.8, zorder=3)
                ax.scatter(*traj[-1], color=col, s=55, marker="X", edgecolors="black",
                           linewidths=0.4, zorder=4)
        ax.set_xlabel(f"FV PC1 ({var_frac[0]:.0%})", fontsize=8)
        ax.set_ylabel(f"FV PC2 ({var_frac[1]:.0%})", fontsize=8)
        ax.set_zlabel(f"FV PC3 ({var_frac[2]:.0%})", fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        if pi == 0:
            from matplotlib.lines import Line2D
            handles = [
                Line2D([], [], color=TRAIN_COLOR, marker="o", linestyle="", label="train FV (20)"),
                Line2D([], [], color="gray", marker="*", linestyle="", markersize=10,
                       markeredgecolor="black", label="true test FV"),
                Line2D([], [], color="gray", marker="o", linestyle="-", label="pre-label ':' mapping (icl01->10)"),
                Line2D([], [], color="gray", marker="^", linestyle="--", label="last-label mapping (icl01->10)"),
                Line2D([], [], color="gray", marker="X", linestyle="", markersize=9,
                       markeredgecolor="black", label="icl10 (final) prediction"),
            ]
            ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.suptitle("Ridge-mapped test-task predictions across ICL positions, in the top-3 FV PCs "
                 f"({var_frac[:3].sum():.0%} var)\nper position: its best-test-MSE layer; "
                 "marker size grows with ICL index; color = test task", fontsize=12)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "fv_pca3d_icl_trajectories.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # Distance-to-true-FV vs ICL index, the quantitative companion.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, role in zip(axes, ROLES):
        for ti, task in enumerate(test_tasks):
            true_fv = fv[n_train + ti]
            d = [float(np.linalg.norm(preds[(icl, role)][ti] - true_fv)) for icl in args.icl_indices]
            ax.plot(args.icl_indices, d, color=task_colors[ti], marker="o", markersize=3,
                    linewidth=1.3, label=task)
        ax.set_xlabel("ICL example index")
        ax.set_title(f"{ROLE_SHORT[role]} token (best layer per position)", fontsize=10)
        ax.set_xticks(args.icl_indices)
    axes[0].set_ylabel("|| predicted FV - true FV ||")
    axes[0].legend(fontsize=7)
    fig.suptitle("Prediction error vs number of ICL examples (7 test tasks)", fontsize=12)
    fig.tight_layout()
    out2 = args.output_dir / "fv_pred_error_vs_icl.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)

    np.savez(args.weights_dir / "icl_trajectory_predictions.npz",
             test_tasks=np.array(test_tasks),
             icl_indices=np.array(args.icl_indices),
             **{f"{ROLE_SHORT[r]}_icl{i:02d}": preds[(i, r)] for i, r in preds})
    with open(args.output_dir / "fv_pca3d_icl_trajectories_cells.json", "w") as f:
        json.dump({f"icl{i:02d}/{ROLE_SHORT[r]}": layers[(i, r)] for i, r in sorted(layers)}, f, indent=2)
    print(f"Wrote {out}, {out2} (+ predictions npz, cells json)")


if __name__ == "__main__":
    main()
