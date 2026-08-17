#!/usr/bin/env python
"""The 139 pooled-90% CENTERED PC directions of the dot_perhead unit read dirs (CPU fp64).

Same SVD as the dimensionality analysis (55 train tasks, centered): keep the top-k basis at
cum sigma^2 >= 0.90 (k = 139 per the dot_perhead__unit cell). These are the candidate
ablation directions for the sparse ablation-direction optimization.

Output: artifacts/69_task_run/read_dir_sweep/dot_perhead_unit_pc139.pt
  {"V": (k, 4096) fp32 orthonormal rows, "s": (k,), "k": k, "center": (4096,) fp32}
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT

SWEEP_ROOT = ARTIFACTS_ROOT / "69_task_run" / "read_dir_sweep"


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    rows = []
    for t in split["train_tasks"]:
        d = torch.load(SWEEP_ROOT / "dot_perhead" / f"{t}.pt", map_location="cpu", weights_only=False)
        rows.append(d["r"].numpy())
    x = np.concatenate(rows, 0).astype(np.float64)
    center = x.mean(axis=0)
    xc = x - center
    _, s, vh = np.linalg.svd(xc, full_matrices=False)
    e = np.cumsum(s ** 2) / (s ** 2).sum()
    k = int(np.searchsorted(e, 0.90) + 1)
    V = vh[:k]
    assert np.abs(V @ V.T - np.eye(k)).max() < 1e-10
    torch.save({"V": torch.from_numpy(V).float(), "s": torch.from_numpy(s[:k]).float(),
                "k": k, "center": torch.from_numpy(center).float(),
                "config": {"source": "dot_perhead unit rows, 55 train tasks, centered SVD, "
                                     "cum sigma^2 >= 0.90"}},
               SWEEP_ROOT / "dot_perhead_unit_pc139.pt")
    print(f"k={k} (expect 139); wrote {SWEEP_ROOT / 'dot_perhead_unit_pc139.pt'}")


if __name__ == "__main__":
    main()
