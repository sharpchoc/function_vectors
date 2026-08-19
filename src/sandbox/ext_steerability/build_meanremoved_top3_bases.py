#!/usr/bin/env python
"""Top-3 SVD compression of the 11 task-unique (mean-removed) read directions.

USER REQUEST 2026-08-19: the 11 mean-removed layer-wise directions per task
(build_meanremoved11_bases.py) have effective rank ~1.4; compress them by SVD of the
11 UNIT-NORMED residuals (so every layer counts equally) and keep the top 3 right
singular vectors as the ablation basis.

Output: artifacts/69_task_run/bottom_up_ablation/meanremoved_top3_bases.pt
  {tasks: {task: {"V": (3, 4096) fp32 orthonormal rows, "s": (11,) singular values of
                  the unit-normed stack, "energy_top3": float  # sum s_i^2 i<3 / total
                  }},
   "rank": 3, "layers": [5..15], "source": "label_avg10_L5-15_acts"}

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

ACTS = ARTIFACTS_ROOT / "69_task_run" / "label_avg10_L5-15_acts"
OUT = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "meanremoved_top3_bases.pt"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"
RANK = 3


def main():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    feats, layers = {}, None
    for t in tasks:
        d = torch.load(ACTS / f"{t}.pt", map_location="cpu", weights_only=False)
        layers = d["layers"]
        feats[t] = d["acts"].double().mean(dim=0)          # (11, 4096)
    X = torch.stack([feats[t] for t in tasks])             # (69, 11, 4096)
    mdirs = X.mean(dim=0)
    mdirs = mdirs / mdirs.norm(dim=1, keepdim=True)
    R = X - (X * mdirs).sum(-1, keepdim=True) * mdirs      # mean-removed residuals

    out, energies = {}, []
    for ti, t in enumerate(tasks):
        U = R[ti] / R[ti].norm(dim=1, keepdim=True)        # (11, 4096) UNIT-NORMED
        _, s, Vh = torch.linalg.svd(U, full_matrices=False)
        V = Vh[:RANK]
        energy = float((s[:RANK] ** 2).sum() / (s ** 2).sum())
        g = V @ V.T - torch.eye(RANK, dtype=torch.float64)
        assert float(g.abs().max()) < 1e-8, f"{t}: basis not orthonormal"
        out[t] = {"V": V.float(), "s": s.float(), "energy_top3": energy,
                  "max_meandir_cos": float((V @ mdirs.T).abs().max())}
        energies.append(energy)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tasks": out, "rank": RANK, "layers": list(layers),
                "source": "label_avg10_L5-15_acts",
                "note": "top-3 SVD of the 11 unit-normed mean-removed directions"}, OUT)
    energies = torch.tensor(energies)
    leaks = torch.tensor([out[t]["max_meandir_cos"] for t in tasks])
    print(f"wrote {OUT}  ({len(out)} tasks)")
    print(f"top-3 energy fraction: min={energies.min():.4f} median={energies.median():.4f} "
          f"max={energies.max():.4f}")
    print(f"max |cos(basis, any layer mean dir)|: median={leaks.median():.3f} max={leaks.max():.3f}")


if __name__ == "__main__":
    main()
