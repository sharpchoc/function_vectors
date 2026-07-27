#!/usr/bin/env python
"""SANDBOX: singular-value spectra of the per-prompt ridge maps across token positions.

For a set of (icl_index, token_role) cells — each at its best layer by stored test_r2_fv in the
sandbox per-prompt study — refit the full-dim per-prompt ridge (canonical protocol: per-cell
20-train-task standardizer, centered eigendecomposition ridge, the cell's stored CV-chosen
alpha) and compute the weight matrix's full singular spectrum.

Repro gate per cell: refit test-vs-stored-FV MSE must match the stored shard metrics
(rel tol --gate_rel_tol); mismatch = hard stop, user adjudicates.

Output (grid-only PNG policy): spectra_positions_grid.png + spectra.npz + summary.json under
results/sandbox/perprompt_ridge_pilot/gptj_train_varicl_top40/weight_spectrum_positions/.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT  # noqa: E402
from sandbox.perprompt_fv.regress_activation_to_perprompt_headsum_ridge import (  # noqa: E402
    align_targets,
    load_function_vector,
    load_json,
    load_perprompt_targets,
    load_task_role_pooled,
    role_load_icl_index,
    write_json,
)

DEFAULT_CELLS = ["1:pre_label_token", "2:pre_label_token", "9:pre_label_token",
                 "10:pre_label_token", "10:last_prompt_token",
                 "1:last_label_token", "2:last_label_token", "9:last_label_token",
                 "10:last_label_token"]
ROLE_SHORT = {"pre_label_token": "pre", "last_label_token": "last", "first_label_token": "first",
              "last_prompt_token": "finaltok"}


def parse_args():
    p = argparse.ArgumentParser(description="SANDBOX: per-prompt ridge spectra across token positions.")
    p.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS), help="icl:role entries.")
    p.add_argument("--rank_metric", type=str, default="test_r2_fv",
                   help="Column of the shard metrics used to pick each cell's best layer.")
    p.add_argument("--study_root", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/gptj_train_varicl_top40")
    p.add_argument("--task_manifest", type=Path, default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_varicl_top40")
    p.add_argument("--targets_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_head_acts/gptj_train_varicl_top40")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--test_tasks", nargs="+", default=[
        "landmark-country", "word_length", "capitalize_first_letter", "synonym",
        "lowercase_first_letter", "capitalize", "antonym"])
    p.add_argument("--std_eps", type=float, default=1e-6)
    p.add_argument("--gate_rel_tol", type=float, default=1e-3)
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Default: <study_root>/weight_spectrum_positions")
    return p.parse_args()


def best_cell_row(study_root, icl, role, rank_metric):
    path = study_root / f"perprompt_shard_icl{icl}/metrics.csv"
    rows = [r for r in csv.DictReader(open(path))
            if r["token_role"] == role and r.get("target_mode") == "perprompt"]
    if not rows:
        raise ValueError(f"No perprompt rows for icl{icl}/{role} in {path}")
    return max(rows, key=lambda r: float(r[rank_metric]))


def spectrum_stats(sv):
    sv = np.asarray(sv, dtype=np.float64)
    energy = sv ** 2
    cum = np.cumsum(energy) / energy.sum()
    return {
        "top_sv": float(sv[0]),
        "rank_energy_90": int(np.searchsorted(cum, 0.90) + 1),
        "rank_energy_99": int(np.searchsorted(cum, 0.99) + 1),
        "participation_ratio": float(energy.sum() ** 2 / (energy ** 2).sum()),
        "sv21_over_sv1": float(sv[20] / sv[0]),
        "sv100_over_sv1": float(sv[99] / sv[0]),
    }


def main():
    args = parse_args()
    out_dir = args.output_dir if args.output_dir is not None else args.study_root / "weight_spectrum_positions"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    manifest = load_json(args.task_manifest)
    train_tasks = sorted(manifest["train_tasks"])
    test_tasks = sorted(args.test_tasks)
    all_tasks = train_tasks + test_tasks
    fvs = {t: load_function_vector(args.fv_root, t) for t in all_tasks}
    target_maps = {t: load_perprompt_targets(args.targets_root, t, args.splits) for t in all_tasks}
    print(f"loaded FVs + per-prompt targets for {len(all_tasks)} tasks ({time.time()-t0:.0f}s)", flush=True)

    cells = []
    for spec in args.cells:
        icl_s, role = spec.split(":")
        row = best_cell_row(args.study_root, int(icl_s), role, args.rank_metric)
        cells.append({"icl": int(icl_s), "role": role, "layer": int(row["layer"]),
                      "alpha": float(row["best_alpha"]), "stored_mse_fv": float(row["test_mse_fv"]),
                      "stored_r2_fv": float(row["test_r2_fv"]), "stored_r2_pp": float(row["test_r2_pp"])})

    spectra, summary = {}, {"rank_metric": args.rank_metric, "cells": []}
    for cell in cells:
        icl, role, layer = cell["icl"], cell["role"], cell["layer"]
        root = args.query_activations_root if icl == 10 else Path(
            args.icl_activations_root_template.format(icl=icl))
        load_icl = role_load_icl_index(role, icl)
        xs, ys = [], []
        xs_test = {}
        for task in all_tasks:
            acts, keys = load_task_role_pooled(root, task, args.splits, role, load_icl)
            x = acts[:, layer, :].float()
            y = align_targets(target_maps[task], keys, task)
            if task in train_tasks:
                xs.append(x)
                ys.append(y)
            else:
                xs_test[task] = x
        x_fit = torch.cat(xs, dim=0)
        y_fit = torch.cat(ys, dim=0)

        mu = x_fit.mean(dim=0)
        sd = x_fit.std(dim=0).clamp_min(args.std_eps)
        xz = (x_fit - mu) / sd
        xbar = xz.mean(dim=0)
        xc = xz - xbar
        eigvals, eigvecs = torch.linalg.eigh(xc.T @ xc)
        ybar = y_fit.mean(dim=0)
        c = eigvecs.T @ (xc.T @ (y_fit - ybar))
        w = eigvecs @ (c / (eigvals + cell["alpha"]).unsqueeze(1))

        # Repro gate: test-vs-stored-FV MSE must match the stored shard metrics.
        sqerr, n = 0.0, 0
        for task in test_tasks:
            xzt = ((xs_test[task] - mu) / sd) - xbar
            pred = xzt @ w + ybar
            sqerr += float(torch.sum((pred - fvs[task].unsqueeze(0)) ** 2))
            n += xzt.shape[0]
        mse_fv = sqerr / (n * w.shape[1])
        rel = abs(mse_fv - cell["stored_mse_fv"]) / cell["stored_mse_fv"]
        key = f"icl{icl:02d}_{ROLE_SHORT[role]}_L{layer:02d}"
        print(f"[{key}] refit mse_fv={mse_fv:.6f} stored={cell['stored_mse_fv']:.6f} rel={rel:.2e}", flush=True)
        if rel > args.gate_rel_tol:
            raise RuntimeError(f"REPRO GATE FAILED for {key}: refit {mse_fv} vs stored "
                               f"{cell['stored_mse_fv']} (rel {rel:.2e}). STOP -- user adjudicates.")

        sv = torch.linalg.svdvals(w.double()).numpy()
        spectra[key] = sv
        summary["cells"].append({**cell, "key": key, "refit_mse_fv": mse_fv,
                                 "gate_rel_diff": rel, **spectrum_stats(sv)})
        print(f"[{key}] rank90={summary['cells'][-1]['rank_energy_90']} "
              f"PR={summary['cells'][-1]['participation_ratio']:.0f} ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(out_dir / "spectra.npz", **spectra)
    write_json(out_dir / "summary.json", summary)

    # Grid figure: one panel per cell, sigma_i/sigma_1 log-log.
    n_cells = len(summary["cells"])
    ncols = 3
    nrows = int(np.ceil(n_cells / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows), squeeze=False)
    for i, cell in enumerate(summary["cells"]):
        ax = axes[i // ncols][i % ncols]
        sv = spectra[cell["key"]]
        ax.plot(np.arange(1, len(sv) + 1), sv / sv[0], lw=1.3, color="#d62728")
        ax.axvline(20, color="gray", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 2)
        ax.grid(alpha=0.25)
        pos = f"icl{cell['icl']:02d}/{ROLE_SHORT[cell['role']]}"
        ax.set_title(f"{pos}  L{cell['layer']}  (R²_fv={cell['stored_r2_fv']:.2f}, α={cell['alpha']:.3g})",
                     fontsize=10)
        ax.text(0.03, 0.06, f"rank90={cell['rank_energy_90']}  PR={cell['participation_ratio']:.0f}\n"
                            f"σ21/σ1={cell['sv21_over_sv1']:.2f}",
                transform=ax.transAxes, fontsize=8, va="bottom")
        if i % ncols == 0:
            ax.set_ylabel("σ_i / σ_1")
        if i // ncols == nrows - 1:
            ax.set_xlabel("singular value index")
    for i in range(n_cells, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    fig.suptitle("SANDBOX: per-prompt ridge map spectra by token position "
                 "(best layer per cell, own CV alpha)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "spectra_positions_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"done in {time.time()-t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
