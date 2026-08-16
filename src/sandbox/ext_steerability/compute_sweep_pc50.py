#!/usr/bin/env python
"""Top-50 UNCENTERED PC directions per read-direction sweep bracket (CPU, float64).

Stack = pooled unit-norm per-prompt read directions of the 55 TRAIN tasks (8250 x 4096)
from artifacts/69_task_run/read_dir_sweep/<bracket>/; uncentered SVD; keep the top-50 right
singular vectors. Used as the ablation subspaces for the PC50 label-token ablation eval.

Output: artifacts/69_task_run/read_dir_sweep/pc50_uncentered.pt
  {bracket: {"V": (50, 4096) fp32 orthonormal rows, "s": (50,) singular values,
             "energy_frac_top50": float}}
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

# local bootstrap for in-repo runs; a PYTHONPATH-supplied repo also works (staged copies)
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT

SWEEP_ROOT = ARTIFACTS_ROOT / "69_task_run" / "read_dir_sweep"
BRACKETS = ("cosine_M", "dot_M", "cosine_perhead", "dot_perhead")


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    out = {}
    for b in BRACKETS:
        rows = []
        for t in split["train_tasks"]:
            d = torch.load(SWEEP_ROOT / b / f"{t}.pt", map_location="cpu", weights_only=False)
            rows.append(d["r"].numpy())
        x = np.concatenate(rows, 0).astype(np.float64)   # UNCENTERED
        _, s, vh = np.linalg.svd(x, full_matrices=False)
        V = vh[:50]
        g = V @ V.T - np.eye(50)
        assert np.abs(g).max() < 1e-10, f"{b}: V rows not orthonormal"
        frac = float((s[:50] ** 2).sum() / (s ** 2).sum())
        out[b] = {"V": torch.from_numpy(V).float(), "s": torch.from_numpy(s[:50]).float(),
                  "energy_frac_top50": frac}
        print(f"{b}: top-50 uncentered energy frac {frac:.4f} (s1={s[0]:.2f})", flush=True)
    torch.save({"brackets": out, "config": {
        "source": "read_dir_sweep unit rows, 55 train tasks, uncentered SVD, top-50"}},
        SWEEP_ROOT / "pc50_uncentered.pt")
    print(f"wrote {SWEEP_ROOT / 'pc50_uncentered.pt'}")


if __name__ == "__main__":
    main()
