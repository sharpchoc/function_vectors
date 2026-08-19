#!/usr/bin/env python
"""Build the 41-PC (95% variance) basis of the 69 per-task L6 label-token means.

Same PCA as results/69_task_run/raw_mean_steering/dimensionality (centered, float64), but
keeping the right-singular VECTORS this time; k = smallest with cum sigma^2 >= 0.95 (41).
Used by the narrow-patching experiment: z <- (I - P) z + P m_A at the '_' slot, P = V^T V.
(The PCA center cancels exactly in remove-and-replace, so only V is needed; the center is
stored anyway for the record.)

Output: artifacts/69_task_run/raw_mean_steering/pc41_basis.pt
  {"V": (k, 4096) fp32 orthonormal rows, "center": (4096,), "s": (k,), "k": k,
   "cum_at_k": float, "layer": 6, "tasks": [...69...]}
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

RM_ROOT = ARTIFACTS_ROOT / "69_task_run" / "label_resid_means"
OUT = ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "pc41_basis.pt"
LAYER = 6


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    X = np.stack([torch.load(RM_ROOT / f"{t}.pt", map_location="cpu",
                             weights_only=False)["resid_means"][LAYER].numpy()
                  for t in tasks]).astype(np.float64)
    center = X.mean(axis=0)
    _, s, vh = np.linalg.svd(X - center, full_matrices=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    k = int(np.searchsorted(cum, 0.95) + 1)
    V = vh[:k]
    assert np.abs(V @ V.T - np.eye(k)).max() < 1e-10
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"V": torch.from_numpy(V).float(), "center": torch.from_numpy(center).float(),
                "s": torch.from_numpy(s[:k]).float(), "k": k, "cum_at_k": float(cum[k - 1]),
                "layer": LAYER, "tasks": tasks,
                "note": "basis over ALL 69 task means (incl. heldout) per user reference to "
                        "the dimensionality analysis; centered PCA, 95% cut"}, OUT)
    print(f"k={k} (cum var {cum[k-1]:.4f}); wrote {OUT}")


if __name__ == "__main__":
    main()
