#!/usr/bin/env python
"""SANDBOX (not repo standard): per-task dimensionality of pre-image matrices.

For each selected task and each of the 899 (icl_index, token_role, layer) cells of the
truncated-SVD pre-image set, stacks that task's 170 pre-images into a [170, 4096] matrix
and computes (user definitions, 2026-07-28):

  On the MEAN-CENTERED matrix (sigma = its singular values):
    * stable_rank          = sum(sigma^2) / sigma_1^2
    * rank90               = smallest k with cum sigma^2 / sum sigma^2 >= 0.90
                             (spectrum_stats convention)
    * participation_ratio  = (sum sigma^2)^2 / sum sigma^4
    * n_pca50              = smallest k with cum sigma^2 / sum sigma^2 >= 0.50
  On the RAW rows:
    * mean_pairwise_cos    = mean over all 14,365 unordered row pairs of cos(x_i, x_j)

Interpretive caveat carried in the outputs: pre-images at a cell live in span(U_k) of that
cell's truncated inversion, so their rank is capped by the cell's inversion rank90
(joined into the CSV as cell_rank90 from preimages_truncsvd/cells_summary.csv).

Outputs (results/.../preimages_truncsvd/task_dimensionality/): metrics.csv, metrics.npz,
heatmaps_<task>.png (one grid PNG per task, 5 metric panels, positions x layers — repo
heatmap style). Spectra computed via the 170x170 Gram eigendecomposition; --selfcheck
verifies the first cell against a direct full-matrix SVD.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_scripts.exploratory.merge_fulldim_ridge_results import position_key, position_label  # noqa: E402
from utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT  # noqa: E402

DEFAULT_TASKS = ["commonsense_qa", "national_parks", "capitalize", "capitalize_first_letter"]
LABEL_ROLES = ["pre_label_token", "first_label_token", "last_label_token"]
METRICS = ["stable_rank", "rank90", "participation_ratio", "n_pca50", "mean_pairwise_cos"]
METRIC_TITLES = {
    "stable_rank": "stable rank  Σσ²/σ₁²  (centered)",
    "rank90": "rank90: k for 90% of Σσ²  (centered)",
    "participation_ratio": "participation ratio  (Σσ²)²/Σσ⁴  (centered)",
    "n_pca50": "PCA components for >50% variance  (centered)",
    "mean_pairwise_cos": "mean pairwise cos of raw rows",
}


def parse_args():
    p = argparse.ArgumentParser(description="SANDBOX: per-task dimensionality of pre-image matrices.")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--preimages_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_fv_preimages/gptj_train_varicl_top40")
    p.add_argument("--cells_summary_csv", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/preimages_truncsvd/cells_summary.csv")
    p.add_argument("--output_dir", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/preimages_truncsvd/task_dimensionality")
    p.add_argument("--selfcheck", action="store_true",
                   help="On the first cell, verify Gram-path spectral metrics against a direct SVD.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def spectral_metrics(s2):
    """Metrics 1-4 from the sorted-descending squared singular values (>=0) of the centered matrix."""
    total = float(s2.sum())
    if total <= 1e-30:  # fully degenerate cell: all rows identical
        return {"stable_rank": 1.0, "rank90": 1, "participation_ratio": 1.0, "n_pca50": 1}
    cum = np.cumsum(s2) / total
    return {
        "stable_rank": total / float(s2[0]),
        "rank90": int(np.searchsorted(cum, 0.90) + 1),
        "participation_ratio": total ** 2 / float((s2 ** 2).sum()),
        "n_pca50": int(np.searchsorted(cum, 0.50) + 1),
    }


def task_metrics(x):
    """x: [n, 4096] fp32 raw pre-images of one task at one cell."""
    xc = x - x.mean(dim=0)
    s2 = torch.linalg.eigvalsh(xc @ xc.T).clamp_min(0).numpy()[::-1].astype(np.float64)
    out = spectral_metrics(s2)
    xn = x / torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(1e-12)
    c = xn @ xn.T
    n = x.shape[0]
    out["mean_pairwise_cos"] = float((c.sum() - torch.diagonal(c).sum()) / (n * (n - 1)))
    return out


def selfcheck_against_svd(x, gram_out):
    sv = torch.linalg.svdvals((x - x.mean(dim=0)).double())
    ref = spectral_metrics((sv.numpy() ** 2).astype(np.float64))
    for k in ("stable_rank", "participation_ratio"):
        rel = abs(ref[k] - gram_out[k]) / max(abs(ref[k]), 1e-12)
        if rel > 1e-4:
            raise RuntimeError(f"SELFCHECK FAILED: {k} Gram={gram_out[k]} SVD={ref[k]} rel={rel:.2e}")
    for k in ("rank90", "n_pca50"):
        if ref[k] != gram_out[k]:
            raise RuntimeError(f"SELFCHECK FAILED: {k} Gram={gram_out[k]} SVD={ref[k]}")
    print(f"[dim] SELFCHECK PASSED (Gram vs direct SVD): {gram_out}")


def render_task_figure(task, grids, pos_labels, layers, out_path):
    n = len(METRICS)
    fig, axes = plt.subplots(1, n, figsize=(max(6, len(layers) * 0.30) * n * 0.75,
                                            max(5, len(pos_labels) * 0.28)), sharey=True)
    for k, (ax, metric) in enumerate(zip(axes, METRICS)):
        im = ax.imshow(grids[metric], aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=5)
        ax.set_xlabel("layer (0 = embedding)")
        ax.set_title(METRIC_TITLES[metric], fontsize=8)
        if k == 0:
            ax.set_yticks(range(len(pos_labels)))
            ax.set_yticklabels(pos_labels, fontsize=6)
            ax.set_ylabel("token position (icl/role)")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle(f"SANDBOX {task}: dimensionality of per-prompt-FV pre-images "
                 f"(170 prompts per cell; spectral metrics on centered rows; "
                 f"rank capped by each cell's inversion rank90)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "metrics.csv"
    if csv_path.exists() and not args.overwrite:
        raise FileExistsError(f"{csv_path} exists; pass --overwrite to replace.")

    cell_rank90 = {}
    with open(args.cells_summary_csv, newline="") as f:
        for r in csv.DictReader(f):
            cell_rank90[(int(r["icl_example_index"]), r["token_role"], int(r["layer"]))] = int(r["rank90"])

    rows_out = []
    done_selfcheck = not args.selfcheck
    t0 = time.time()
    n_files = 0
    for icl in range(1, 11):
        roles = LABEL_ROLES + (["last_prompt_token"] if icl == 10 else [])
        for role in roles:
            for layer in range(29):
                path = args.preimages_root / f"icl{icl}" / role / f"L{layer:02d}.pt"
                data = torch.load(path, map_location="cpu", weights_only=False)
                pre = data["preimages"].float()
                meta = data["metadata"]
                n_files += 1
                for task in args.tasks:
                    idx = [i for i, m in enumerate(meta) if m["task"] == task]
                    if len(idx) != 170:
                        raise RuntimeError(f"{path}: expected 170 rows for {task}, got {len(idx)}")
                    x = pre[idx]
                    m = task_metrics(x)
                    if not done_selfcheck:
                        selfcheck_against_svd(x, {k: m[k] for k in
                                                  ("stable_rank", "rank90", "participation_ratio", "n_pca50")})
                        done_selfcheck = True
                    rows_out.append({
                        "task": task, "icl_index": icl, "token_role": role, "layer": layer,
                        "n_rows": len(idx), "cell_rank90": cell_rank90[(icl, role, layer)], **m,
                    })
                del data, pre
        print(f"[dim] icl{icl} done ({n_files} files, {time.time()-t0:.0f}s)", flush=True)

    fields = ["task", "icl_index", "token_role", "layer", "n_rows", "cell_rank90"] + METRICS
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    # Grids [n_positions, n_layers] per (task, metric), repo heatmap orientation.
    pos_set = sorted({(r["icl_index"], r["token_role"]) for r in rows_out}, key=lambda ir: position_key(*ir))
    layers = sorted({r["layer"] for r in rows_out})
    pos_index = {p: i for i, p in enumerate(pos_set)}
    pos_labels = [position_label(icl, role) for icl, role in pos_set]
    npz = {"positions": np.array(pos_labels), "layers": np.array(layers),
           "tasks": np.array(args.tasks), "metrics": np.array(METRICS)}
    for task in args.tasks:
        grids = {metric: np.full((len(pos_set), len(layers)), np.nan) for metric in METRICS}
        for r in rows_out:
            if r["task"] != task:
                continue
            i = pos_index[(r["icl_index"], r["token_role"])]
            for metric in METRICS:
                grids[metric][i, r["layer"]] = float(r[metric])
        for metric in METRICS:
            if np.isnan(grids[metric]).any():
                raise RuntimeError(f"Unfilled grid cells for {task}/{metric} — a cell file is missing.")
            npz[f"{task}__{metric}"] = grids[metric]
        render_task_figure(task, grids, pos_labels, layers, args.output_dir / f"heatmaps_{task}.png")
        print(f"[dim] wrote heatmaps_{task}.png")
    np.savez_compressed(args.output_dir / "metrics.npz", **npz)

    assert len(rows_out) == n_files * len(args.tasks)
    print(f"[dim] DONE: {len(rows_out)} rows ({n_files} cells x {len(args.tasks)} tasks) -> {csv_path} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
