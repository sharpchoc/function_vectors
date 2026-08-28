#!/usr/bin/env python
"""Materialize and plot the full-dim ridge weight matrices for selected cells.

The ridge worker (regress_activation_to_fv_fulldim_ridge.py) never forms the [4096, 4096]
weight matrix explicitly -- it predicts through an eigendecomposition of the feature Gram. This
script refits the requested (icl, token_role, layer) cells at their stored best alpha (no CV),
materializes W, saves it to artifacts, and renders signed heatmaps (diverging map, symmetric
scale shared across panels).

Conventions: X is standardized with the single pooled 20-train-task scaler and centered, Y is
centered, so W maps standardized-centered activation -> centered FV and pred = ybar + xc @ W.
Panels show W^T (rows = FV output dim, cols = activation input dim), 4x4 block-mean downsampled;
full-resolution per-cell PNGs are saved alongside. Each refit's test MSE is checked against the
stored combined_metrics value.
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
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
    QUERY_ICL_INDEX,
    load_function_vector,
    load_json,
    load_task_role_pooled,
    role_load_icl_index,
)

ROLE_SHORT = {"pre_label_token": "pre", "first_label_token": "first",
              "last_label_token": "last", "last_prompt_token": "finaltok"}
# Reader-facing names for the spectra legend: the pre-label position is where the model must
# produce the answer (cue); the last label token is the answer itself (target).
ROLE_PLOT = {"pre_label_token": "cue (pre-label)", "first_label_token": "first label token",
             "last_label_token": "target (last label)", "last_prompt_token": "final prompt token"}

# (icl_example_index, token_role, layer): best-test-MSE layers for the final 3 pre-label and
# final 3 last-label token positions (from combined_metrics_with_r2.csv).
DEFAULT_CELLS = [
    (8, "pre_label_token", 11),
    (9, "pre_label_token", 11),
    (10, "pre_label_token", 11),
    (8, "last_label_token", 11),
    (9, "last_label_token", 13),
    (10, "last_label_token", 13),
]


def parse_args():
    p = argparse.ArgumentParser(description="Plot full-dim ridge weight-matrix heatmaps.")
    p.add_argument("--cells", nargs="+", default=None,
                   help="Cells as icl:role:layer (e.g. 10:pre_label_token:11). Default: 6 study cells.")
    p.add_argument("--metrics_csv", type=Path,
                   default=FV_FORMATION_DIR / "activation_to_fv_decoding/fulldim_ridge/main/combined_metrics_with_r2.csv")
    p.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_selected")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--weights_dir", type=Path, default=ARTIFACTS_ROOT / "fulldim_ridge_weight_matrices")
    p.add_argument("--output_dir", type=Path, default=FV_FORMATION_DIR / "activation_to_fv_decoding/fulldim_ridge/weight_heatmaps")
    p.add_argument("--std_eps", type=float, default=1e-6)
    p.add_argument("--downsample", type=int, default=4, help="Block-mean factor for the combined figure.")
    return p.parse_args()


def cell_key(icl, role, layer):
    return f"icl{icl:02d}_{ROLE_SHORT[role]}_L{layer:02d}"


def lookup_metrics(metrics_csv, cells):
    stored = {}
    with open(metrics_csv) as f:
        for r in csv.DictReader(f):
            stored[(int(r["icl_example_index"]), r["token_role"], int(r["layer"]))] = (
                float(r["best_alpha"]), float(r["test_mse"]), float(r["test_r2"]))
    missing = [c for c in cells if c not in stored]
    if missing:
        raise KeyError(f"Cells not in {metrics_csv}: {missing}")
    return {c: stored[c] for c in cells}


def fit_weight_matrix(xs, fvs, train_tasks, test_tasks, alpha):
    """Fit centered ridge on standardized features; return (W, ybar, test_mse)."""
    x_fit = torch.cat([xs[t] for t in train_tasks], dim=0)
    y_fit = torch.cat([fvs[t].unsqueeze(0).expand(xs[t].shape[0], -1) for t in train_tasks], dim=0)
    xbar = x_fit.mean(dim=0)
    ybar = y_fit.mean(dim=0)
    xc = x_fit - xbar
    gram = xc.T @ xc
    eigvals, eigvecs = torch.linalg.eigh(gram)
    c = eigvecs.T @ (xc.T @ (y_fit - ybar))
    w = eigvecs @ (c / (eigvals + alpha).unsqueeze(1))  # [in_dim, out_dim]

    sqerr, n = 0.0, 0
    for task in test_tasks:
        pred = (xs[task] - xbar) @ w + ybar
        diff = pred - fvs[task].unsqueeze(0)
        sqerr += float(torch.sum(diff ** 2))
        n += xs[task].shape[0]
    test_mse = sqerr / (n * w.shape[1])
    return w, xbar, ybar, test_mse


def block_absmax(a, k):
    """Downsample by keeping each k x k block's largest-|.| element (sign preserved).

    Mean-pooling signed, dense, near-zero-mean weights washes every block toward 0 and renders
    the heatmap blank; keeping the extreme value preserves the visual texture instead."""
    n0, n1 = a.shape[0] // k * k, a.shape[1] // k * k
    b = a[:n0, :n1].reshape(n0 // k, k, n1 // k, k).transpose(0, 2, 1, 3).reshape(n0 // k, n1 // k, k * k)
    idx = np.abs(b).argmax(axis=-1)
    return np.take_along_axis(b, idx[..., None], axis=-1)[..., 0]


def main():
    args = parse_args()
    cells = DEFAULT_CELLS if args.cells is None else [
        (int(i), r, int(l)) for i, r, l in (c.split(":") for c in args.cells)]
    stored = lookup_metrics(args.metrics_csv, cells)

    manifest = load_json(args.task_manifest)
    train_tasks = list(manifest["train_tasks"])
    test_tasks = list(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)
    all_tasks = train_tasks + test_tasks
    fvs = {t: load_function_vector(args.fv_root, t) for t in all_tasks}

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels = {}
    summary = {}
    for icl, role, layer in cells:
        alpha, stored_mse, stored_r2 = stored[(icl, role, layer)]
        key = cell_key(icl, role, layer)
        t0 = time.time()
        saved_path = args.weights_dir / f"{key}.pt"
        if saved_path.exists():
            saved = torch.load(saved_path, map_location="cpu", weights_only=False)
            w = saved["weight"]
            panels[key] = w.T.numpy()
            summary[key] = {"alpha": alpha, "test_mse_refit": None, "test_mse_stored": stored_mse,
                            "test_r2_stored": stored_r2, "w_fro": float(torch.linalg.norm(w)),
                            "w_absmax": float(w.abs().max())}
            print(f"{key}: loaded saved weights ({saved_path})")
            continue
        if icl == QUERY_ICL_INDEX:
            root = args.query_activations_root
        else:
            root = Path(args.icl_activations_root_template.format(icl=icl))
        load_icl = role_load_icl_index(role, icl)
        xs = {}
        for task in all_tasks:
            a = load_task_role_pooled(root, task, args.splits, role, load_icl)
            xs[task] = a[:, layer, :].to(torch.float32)
        # Single pooled 20-train-task scaler, exactly as the worker does.
        pool = torch.cat([xs[t] for t in train_tasks], dim=0)
        mean, std = pool.mean(dim=0), pool.std(dim=0, unbiased=False).clamp_min(args.std_eps)
        xs = {t: (x - mean) / std for t, x in xs.items()}

        w, xbar, ybar, test_mse = fit_weight_matrix(xs, fvs, train_tasks, test_tasks, alpha)
        if abs(test_mse - stored_mse) > 5e-4:
            raise AssertionError(f"{key}: refit test_mse {test_mse:.5f} != stored {stored_mse:.5f}")
        torch.save({"weight": w, "feature_mean": mean, "feature_std": std, "xbar": xbar,
                    "ybar": ybar, "alpha": alpha, "icl": icl, "token_role": role, "layer": layer,
                    "orientation": "pred = ybar + xc @ weight; weight[in_dim, out_dim]"},
                   args.weights_dir / f"{key}.pt")
        panels[key] = w.T.numpy()  # rows = FV output dim, cols = activation input dim
        summary[key] = {"alpha": alpha, "test_mse_refit": test_mse, "test_mse_stored": stored_mse,
                        "test_r2_stored": stored_r2, "w_fro": float(torch.linalg.norm(w)),
                        "w_absmax": float(w.abs().max())}
        print(f"{key}: refit ok (test_mse {test_mse:.5f} vs stored {stored_mse:.5f}, "
              f"alpha={alpha:.3g}, |W|_F={summary[key]['w_fro']:.3f}) in {time.time()-t0:.0f}s")

    # Shared symmetric scale across panels (signed weights -> diverging map, neutral at 0).
    vmax = float(np.percentile(np.abs(np.stack(list(panels.values()))), 99.5))

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11), sharex=True, sharey=True)
    for ax, (icl, role, layer) in zip(axes.flat, cells):
        key = cell_key(icl, role, layer)
        img = block_absmax(panels[key], args.downsample)
        im = ax.imshow(img, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest", extent=[0, 4096, 4096, 0])
        s = summary[key]
        ax.set_title(f"icl{icl:02d}/{ROLE_SHORT[role]}  L{layer}  "
                     f"(alpha={s['alpha']:.3g}, test R2={s['test_r2_stored']:.3f})", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("activation input dim")
    for ax in axes[:, 0]:
        ax.set_ylabel("FV output dim")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("weight (standardized activation -> centered FV)")
    fig.suptitle(f"Full-dim ridge weight matrices W^T ({args.downsample}x{args.downsample} block signed max-|.|; "
                 "shared symmetric scale at 99.5th pct |W|)", fontsize=12)
    out_png = args.output_dir / "weight_heatmaps_6cells.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")

    # Full-resolution per-cell images (exact 4096x4096 pixels, same shared scale). These are
    # ~50 MB apiece, so they go to gitignored artifacts next to the saved weights.
    for key, wt in panels.items():
        plt.imsave(args.weights_dir / f"weight_fullres_{key}.png",
                   np.clip(wt, -vmax, vmax), cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    # Singular spectra: exact full SVD of each 4096x4096 W (torch.linalg.svdvals, all 4096
    # values computed; top 40 stored/plotted — the rest sit on the numerical noise floor).
    spectra = {}
    for (icl, role, layer) in cells:
        key = cell_key(icl, role, layer)
        sv = torch.linalg.svdvals(torch.from_numpy(panels[key]).to(torch.float64)).numpy()
        spectra[key] = sv
        summary[key]["singular_values_top40"] = [float(v) for v in sv[:40]]
        summary[key]["max_singular_value_beyond_40"] = float(sv[40:].max())

    # Plot only the highest-ICL cells (one line per role) to avoid overlapping lines, on a
    # broken log y-axis: signal spectrum on top, numerical noise floor below.
    plot_icl = max(icl for icl, _, _ in cells)
    plot_cells = [(icl, role, layer) for (icl, role, layer) in cells if icl == plot_icl]
    n_show = 40
    shown = np.stack([spectra[cell_key(*c)][:n_show] for c in plot_cells])
    gaps = shown[:, :-1] / shown[:, 1:]
    split = int(np.argmax(gaps.max(axis=0))) + 1  # index of first noise-floor value
    signal, noise = shown[:, :split], shown[:, split:]

    fig = plt.figure(figsize=(7.5, 5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    for (icl, role, layer) in plot_cells:
        sv = spectra[cell_key(icl, role, layer)][:n_show]
        for ax in (ax_top, ax_bot):
            ax.plot(range(1, n_show + 1), sv, marker="o", markersize=3, linewidth=1.5,
                    label=f"icl{icl:02d} {ROLE_PLOT[role]} L{layer}")
    for ax in (ax_top, ax_bot):
        ax.set_yscale("log")
    ax_top.set_ylim(signal.min() * 0.6, signal.max() * 1.6)
    ax_bot.set_ylim(noise.min() * 0.6, noise.max() * 1.6)
    # Broken-axis styling: hide the facing spines and draw diagonal break marks.
    ax_top.spines.bottom.set_visible(False)
    ax_bot.spines.top.set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    d = 0.5
    break_kw = dict(marker=[(-1, -d), (1, d)], markersize=10, linestyle="none",
                    color="k", mec="k", mew=1, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **break_kw)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **break_kw)
    ax_bot.set_xlabel("singular value index")
    ax_top.set_ylabel("singular value")
    ax_bot.set_ylabel("noise floor")
    ax_top.set_title(f"Singular spectra of the ridge weight matrices (exact SVD, top {n_show} of 4096)")
    ax_top.legend(fontsize=8)
    fig.savefig(args.output_dir / "weight_singular_spectra.png", dpi=150)
    plt.close(fig)
    print("Wrote weight_singular_spectra.png")

    with open(args.output_dir / "weight_heatmaps_summary.json", "w") as f:
        json.dump({"cells": summary, "shared_vmax": vmax, "downsample": args.downsample}, f, indent=2)
    print(f"Wrote {len(panels)} full-res PNGs + weight_heatmaps_summary.json")


if __name__ == "__main__":
    main()
