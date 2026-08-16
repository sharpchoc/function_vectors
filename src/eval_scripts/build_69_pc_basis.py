#!/usr/bin/env python
"""Uncentered PC basis of the 69-task-run TRAIN per-prompt FV stack (CPU).

Stacks the 55 train tasks' per-prompt FVs (8250 x 4096, from perprompt_fvs/), runs an
UNCENTERED float64 SVD on CPU, and saves the top-512 right singular vectors as the PC
dictionary for sparse PC selection. Gate: orthonormality of the saved basis.
Writes ARTIFACTS_ROOT/69_task_run/pc_sparse/pc_basis_uncentered.pt.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT  # noqa: E402

PP_ROOT = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
OUT_DIR = ARTIFACTS_ROOT / "69_task_run" / "pc_sparse"
TOP_K = 512


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    train = split["train_tasks"]
    stack = np.concatenate(
        [torch.load(PP_ROOT / f"{t}.pt", map_location="cpu", weights_only=False)["fv"]
         .float().numpy() for t in train], axis=0).astype(np.float64)
    assert stack.shape == (150 * len(train), 4096), stack.shape
    # UNCENTERED SVD (user-specified); CPU float64 (never CUDA gesvdj for spectra)
    u, s, vt = np.linalg.svd(stack, full_matrices=False)
    pcs = vt[:TOP_K]  # (512, 4096)
    gram_err = np.abs(pcs @ pcs.T - np.eye(TOP_K)).max()
    assert gram_err < 1e-8, f"orthonormality gate failed: {gram_err:.3e}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"pcs": torch.from_numpy(pcs).float(),
                "singular_values": torch.from_numpy(s).float(),
                "top_k": TOP_K, "centered": False,
                "train_tasks": train, "n_rows": stack.shape[0],
                "source": "artifacts/69_task_run/perprompt_fvs (fv rows, uncentered)"},
               OUT_DIR / "pc_basis_uncentered.pt")
    ev = s ** 2
    cum = np.cumsum(ev) / ev.sum()
    print(f"stack {stack.shape}; top-{TOP_K} PCs saved; orthonormality err {gram_err:.2e}")
    print(f"uncentered spectrum: s1={s[0]:.1f} s2={s[1]:.1f} s512={s[TOP_K-1]:.1f}; "
          f"energy in top-512: {cum[TOP_K-1]:.4f}; 90%@{int(np.searchsorted(cum, .9) + 1)} "
          f"95%@{int(np.searchsorted(cum, .95) + 1)}")


if __name__ == "__main__":
    main()
