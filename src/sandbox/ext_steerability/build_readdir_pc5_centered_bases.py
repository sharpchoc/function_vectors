#!/usr/bin/env python
"""Top-5 CENTERED PCs of each task's per-prompt read-feature activations.

Counterpart of build_readdir_pc5_bases.py (USER REQUEST 2026-08-19): subtract the task's
own mean of the 1500 label-token vectors before the SVD, so the basis spans the top-5
within-task VARIANCE directions and the shared/task mean direction itself is left out of
the ablated subspace ("the shared mean direction can remain, but task unique directions
are removed").

Input per task: artifacts/69_task_run/label_all10_L6_acts/<task>.pt — acts (150, 10, 4096),
block-6 output at the LAST token of each demo label.

Output: artifacts/69_task_run/bottom_up_ablation/pc5_centered_bases.pt
  {tasks: {task: {"V": (5, 4096) fp32 orthonormal rows, "s": (5,) singular values,
                  "mean_frac_in_V": float  # ||P_V m|| / ||m||, m = unit task mean —
                  }},                      # how much of the mean the subspace removes anyway
   "rank": 5, "source": "label_all10_L6_acts", "centered": True}

SVD on CPU float64 (CUDA default gesvdj is ~1e-3-inaccurate; see DECISIONS).
"""
import json
import sys
from pathlib import Path

import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT  # noqa: E402

RANK = 5
ACTS = ARTIFACTS_ROOT / "69_task_run" / "label_all10_L6_acts"
MEANS = ARTIFACTS_ROOT / "69_task_run" / "label_resid_means"
OUT = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "pc5_centered_bases.pt"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"


def main():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    out = {}
    fracs = []
    for task in tasks:
        acts = torch.load(ACTS / f"{task}.pt", map_location="cpu",
                          weights_only=False)["acts"]          # (150, 10, 4096) fp16
        X = acts.reshape(-1, acts.shape[-1]).double()          # (1500, 4096)
        X = X - X.mean(dim=0, keepdim=True)                    # CENTERED
        _, s, Vh = torch.linalg.svd(X, full_matrices=False)
        V = Vh[:RANK]                                          # (5, 4096) orthonormal
        m = torch.load(MEANS / f"{task}.pt", map_location="cpu",
                       weights_only=False)["resid_means"][6].double()
        m = m / m.norm()
        frac = float((V @ m).norm())                           # ||P_V m||, m unit
        out[task] = {"V": V.float(), "s": s[:RANK].float(), "mean_frac_in_V": frac}
        fracs.append(frac)
        g = V @ V.T - torch.eye(RANK, dtype=torch.float64)
        assert float(g.abs().max()) < 1e-8, f"{task}: basis not orthonormal"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tasks": out, "rank": RANK, "source": "label_all10_L6_acts",
                "centered": True,
                "note": "centered PCA over all 1500 label-token vectors"}, OUT)
    fracs = torch.tensor(fracs)
    print(f"wrote {OUT}  ({len(out)} tasks)")
    print(f"||P_V unit-mean||: min={fracs.min():.4f} median={fracs.median():.4f} "
          f"max={fracs.max():.4f}")
    worst = sorted(zip(tasks, fracs.tolist()), key=lambda x: -x[1])[:5]
    print("highest mean leakage:", ", ".join(f"{t}={c:.4f}" for t, c in worst))


if __name__ == "__main__":
    main()
