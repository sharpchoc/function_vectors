#!/usr/bin/env python
"""Top-5 uncentered PCs of each task's per-prompt read-feature activations.

Input per task: artifacts/69_task_run/label_all10_L6_acts/<task>.pt — acts (150, 10, 4096),
block-6 output at the LAST token of each demo label. USER DECISION 2026-08-20: PCA over all
1500 individual label-token vectors (prompts x slots), UNCENTERED, so PC1 ~ the raw mean
direction already used by the single-direction ablation (cos(PC1, unit resid_means[6]) is
computed per task and must be high; the run script asserts >= 0.98 before spending GPU).

Output: artifacts/69_task_run/bottom_up_ablation/pc5_bases.pt
  {tasks: {task: {"V": (5, 4096) fp32 orthonormal rows, "s": (5,) singular values,
                  "cos_pc1_mean": float}}, "rank": 5, "source": "label_all10_L6_acts"}

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
OUT = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "pc5_bases.pt"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"


def main():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    out = {}
    coss = []
    for task in tasks:
        acts = torch.load(ACTS / f"{task}.pt", map_location="cpu",
                          weights_only=False)["acts"]          # (150, 10, 4096) fp16
        X = acts.reshape(-1, acts.shape[-1]).double()          # (1500, 4096), UNCENTERED
        _, s, Vh = torch.linalg.svd(X, full_matrices=False)
        V = Vh[:RANK]                                          # (5, 4096) orthonormal
        m = torch.load(MEANS / f"{task}.pt", map_location="cpu",
                       weights_only=False)["resid_means"][6].double()
        cos = float(torch.abs(V[0] @ m) / m.norm())            # sign-free
        # fix signs so PC1 points along the mean (cosmetic; projector is sign-invariant)
        if float(V[0] @ m) < 0:
            V = -V
        out[task] = {"V": V.float(), "s": s[:RANK].float(), "cos_pc1_mean": cos}
        coss.append(cos)
        g = V @ V.T - torch.eye(RANK, dtype=torch.float64)
        assert float(g.abs().max()) < 1e-8, f"{task}: basis not orthonormal"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tasks": out, "rank": RANK, "source": "label_all10_L6_acts",
                "note": "uncentered PCA over all 1500 label-token vectors"}, OUT)
    coss = torch.tensor(coss)
    print(f"wrote {OUT}  ({len(out)} tasks)")
    print(f"cos(PC1, unit mean): min={coss.min():.4f} median={coss.median():.4f} "
          f"max={coss.max():.4f}")
    worst = sorted(zip(tasks, coss.tolist()), key=lambda x: x[1])[:5]
    print("lowest:", ", ".join(f"{t}={c:.4f}" for t, c in worst))


if __name__ == "__main__":
    main()
