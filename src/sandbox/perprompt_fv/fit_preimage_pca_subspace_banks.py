#!/usr/bin/env python
"""SANDBOX (not repo standard): per-task PCA subspace banks of per-prompt-FV pre-images.

For each Stream W ablation cell and each of the 7 ridge held-out tasks, this builds the
ablation subspaces for the top-k-PCA pre-image subspace ablation study (user spec 2026-07-28):

  x        = the task's 170 pre-images at (cell, capture layer)   [170, 4096] raw space
  m_t      = row mean of x (the task-mean pre-image)
  PC1..PC4 = top right singular vectors of the MEAN-CENTERED x (fp64 SVD)
  Q_k      = QR-orthonormalization of [m_t, PC1..PCk]             [4096, k+1], k in {0,2,3,4}
             (k=0 bridge arm: the unit task-mean direction alone; QR is sequential, so the
             k subspaces are nested and all contain the task-mean direction)
  g        = grand mean over ALL 4590 pre-images (27 tasks) at that (cell, layer)
  tvec_k   = Q_k^T g   (mean-replacement clamp target in subspace coordinates)

Banks are keyed by edit layer b = capture layer - 1 (b in 0..27; hook on transformer.h.{b}
output; the L00 embedding entry is never used), matching DECISIONS 2026-07-10.

Gates (hard stop -> user adjudicates; never self-adjudicate):
  * ORTHONORMALITY: ||Q^T Q - I||_inf < 1e-5 and mean-in-span ||m - Q Q^T m||/||m|| < 1e-5
    for every (cell, layer, task, k).
  * SVD SELFCHECK: on the first cell, spectral metrics from the fp32 Gram path (the
    dimensionality study's task_metrics) must match a direct fp64 SVD (its own selfcheck).
  * CONSISTENCY GATE vs the stored dimensionality study: for the overlap tasks
    (capitalize, capitalize_first_letter), n_pca50/rank90 recomputed with the IDENTICAL
    numeric path must EXACTLY match task_dimensionality/metrics.csv on every overlapping
    (icl, role, layer) cell; stable_rank/participation_ratio at rel <= 1e-8.

Output: artifacts/sandbox/perprompt_fv_preimages/gptj_train_varicl_top40_pca_banks/
          <cell>/<task>_pca_subspace_bank.pt
        + gate report JSON in the same root.

CPU-only by design (LAPACK; avoids the CUDA gesvdj inaccuracy, DECISIONS 2026-07-27).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_scripts.regress_activation_to_fv_fulldim_ridge import (  # noqa: E402
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
)
from sandbox.perprompt_fv.analyze_preimage_task_dimensionality import (  # noqa: E402
    selfcheck_against_svd,
    task_metrics,
)
from utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT  # noqa: E402

DEFAULT_TEST_TASKS = list(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)
# Stream W arm cells: <cell name> -> (icl_index, token_role) in the pre-image tree.
ABLATION_CELLS = {
    "pre_label_token_icl1": (1, "pre_label_token"),
    "last_label_token_icl1": (1, "last_label_token"),
    "pre_label_token_icl2": (2, "pre_label_token"),
    "pre_label_token_icl10": (10, "pre_label_token"),
    "last_label_token_icl10": (10, "last_label_token"),
    "last_prompt_token_icl10": (10, "last_prompt_token"),
}
KS = [0, 2, 3, 4]
N_EDIT_LAYERS = 28
OVERLAP_GATE_TASKS = ["capitalize", "capitalize_first_letter"]


def parse_args():
    p = argparse.ArgumentParser(description="SANDBOX: PCA subspace banks of per-prompt-FV pre-images.")
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TEST_TASKS)
    p.add_argument("--cells", nargs="+", default=list(ABLATION_CELLS), choices=list(ABLATION_CELLS))
    p.add_argument("--preimages_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_fv_preimages/gptj_train_varicl_top40")
    p.add_argument("--output_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_fv_preimages/gptj_train_varicl_top40_pca_banks")
    p.add_argument("--dim_metrics_csv", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/preimages_truncsvd/task_dimensionality/metrics.csv")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_dim_metrics(csv_path):
    """(task, icl, role, capture_layer) -> stored dimensionality metrics (for the consistency gate)."""
    ref = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["task"], int(r["icl_index"]), r["token_role"], int(r["layer"]))
            ref[key] = {"n_pca50": int(r["n_pca50"]), "rank90": int(r["rank90"]),
                        "stable_rank": float(r["stable_rank"]),
                        "participation_ratio": float(r["participation_ratio"])}
    return ref


def consistency_gate(x, task, icl, role, cap_layer, ref):
    """Recompute the dimensionality metrics with the study's own numeric path; require agreement."""
    key = (task, icl, role, cap_layer)
    if key not in ref:
        return 0
    m = task_metrics(x)
    stored = ref[key]
    for k in ("n_pca50", "rank90"):
        if m[k] != stored[k]:
            raise RuntimeError(f"CONSISTENCY GATE FAILED at {key}: {k} recomputed={m[k]} "
                               f"stored={stored[k]} — STOP, user adjudicates.")
    for k in ("stable_rank", "participation_ratio"):
        rel = abs(m[k] - stored[k]) / max(abs(stored[k]), 1e-12)
        if rel > 1e-8:
            raise RuntimeError(f"CONSISTENCY GATE FAILED at {key}: {k} recomputed={m[k]!r} "
                               f"stored={stored[k]!r} rel={rel:.2e} — STOP, user adjudicates.")
    return 1


def subspaces_for_task(x, g):
    """x: [170, 4096] fp32 task pre-images; g: [4096] fp32 grand mean over all 27 tasks.

    Returns per-k dict with Q [4096, k+1] fp32, tvec [k+1] fp32, plus diagnostics."""
    x64 = x.double()
    m = x64.mean(dim=0)
    m_norm = float(torch.linalg.norm(m))
    if m_norm < 1e-8:
        raise RuntimeError("task-mean pre-image is ~0; subspace undefined.")
    xc = x64 - m
    # fp64 thin SVD on CPU (LAPACK); right singular vectors = centered PCs.
    _, S, Vh = torch.linalg.svd(xc, full_matrices=False)
    pcs = Vh[: max(KS)]                                  # [4, 4096]
    out = {}
    g64 = g.double()
    for k in KS:
        A = torch.cat([m.unsqueeze(0), pcs[:k]], dim=0).T    # [4096, k+1]
        Q, R = torch.linalg.qr(A, mode="reduced")
        if not torch.all(torch.diagonal(R).abs() > 1e-10 * m_norm):
            raise RuntimeError(f"rank-deficient QR (k={k}): diag(R)={torch.diagonal(R).tolist()}")
        # Orthonormality + mean-in-span gates.
        eye_err = float((Q.T @ Q - torch.eye(k + 1, dtype=torch.float64)).abs().max())
        proj_m = Q @ (Q.T @ m)
        span_err = float(torch.linalg.norm(m - proj_m) / m_norm)
        if eye_err > 1e-5 or span_err > 1e-5:
            raise RuntimeError(f"ORTHONORMALITY GATE FAILED (k={k}): "
                               f"||QtQ-I||={eye_err:.2e} mean-span rel={span_err:.2e}")
        out[k] = {"Q": Q.float(), "tvec": (Q.T @ g64).float(),
                  "eye_err": eye_err, "span_err": span_err}
    out["diag"] = {"m_norm": m_norm, "g_norm": float(torch.linalg.norm(g64)),
                   "sv_top8": S[:8].float().tolist(),
                   "cos_m_g": float((m @ g64) / (m_norm * torch.linalg.norm(g64)))}
    return out


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    ref = load_dim_metrics(args.dim_metrics_csv)
    torch.set_grad_enabled(False)

    t0 = time.time()
    gate_hits = 0
    did_selfcheck = False
    report = {"cells": {}, "tasks": args.tasks, "ks": KS,
              "preimages_root": str(args.preimages_root)}
    for cell in args.cells:
        icl, role = ABLATION_CELLS[cell]
        cell_dir = args.output_root / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        out_paths = {t: cell_dir / f"{t}_pca_subspace_bank.pt" for t in args.tasks}
        if not args.overwrite and all(p.exists() for p in out_paths.values()):
            print(f"[banks] {cell}: all {len(args.tasks)} banks exist; skipping.", flush=True)
            continue

        banks = {t: {} for t in args.tasks}
        for cap_layer in range(1, N_EDIT_LAYERS + 1):            # edit layer b = cap_layer - 1
            path = args.preimages_root / f"icl{icl}" / role / f"L{cap_layer:02d}.pt"
            data = torch.load(path, map_location="cpu", weights_only=False)
            pre = data["preimages"].float()
            meta = data["metadata"]
            assert pre.shape == (4590, 4096), f"{path}: unexpected shape {tuple(pre.shape)}"
            g = pre.mean(dim=0)                                   # grand mean over all 27 tasks
            for task in args.tasks:
                idx = [i for i, m in enumerate(meta) if m["task"] == task]
                if len(idx) != 170:
                    raise RuntimeError(f"{path}: expected 170 rows for {task}, got {len(idx)}")
                x = pre[idx]
                if not did_selfcheck:
                    m = task_metrics(x)
                    selfcheck_against_svd(x, {k: m[k] for k in
                                              ("stable_rank", "rank90", "participation_ratio", "n_pca50")})
                    did_selfcheck = True
                gate_hits += consistency_gate(x, task, icl, role, cap_layer, ref)
                banks[task][cap_layer - 1] = subspaces_for_task(x, g)
            del data, pre
        for task in args.tasks:
            assert set(banks[task]) == set(range(N_EDIT_LAYERS))
            torch.save({"sandbox": True, "cell": cell, "icl_index": icl, "token_role": role,
                        "task": task, "ks": KS,
                        "subspaces_by_edit_layer": banks[task],
                        "config": {"preimages_root": str(args.preimages_root),
                                   "basis": "QR([task-mean, top-k centered PCs]), fp64 SVD/QR",
                                   "tvec": "Q^T (grand mean over all 4590 pre-images)"}},
                       out_paths[task])
        report["cells"][cell] = {"tasks_written": args.tasks}
        print(f"[banks] {cell}: wrote {len(args.tasks)} banks ({time.time()-t0:.0f}s)", flush=True)

    if gate_hits:
        print(f"[banks] CONSISTENCY GATE PASSED on {gate_hits} overlapping (task, cell, layer) "
              f"combos vs {args.dim_metrics_csv}")
    else:
        print("[banks] WARNING: consistency gate matched 0 stored rows (nothing to compare?)")
    report["consistency_gate_rows"] = gate_hits
    with open(args.output_root / "gate_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[banks] DONE in {time.time()-t0:.0f}s -> {args.output_root}")


if __name__ == "__main__":
    main()
