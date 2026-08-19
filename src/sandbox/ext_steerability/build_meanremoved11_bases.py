#!/usr/bin/env python
"""Per-task 11-direction 'task-unique' read-feature bases (mean-removed, L5-L15).

USER REQUEST 2026-08-19: for each task, take the task-level read feature at each of
layers 5..15 (mean over the 150 slot-averaged label activations in
artifacts/69_task_run/label_avg10_L5-15_acts), remove its projection onto that layer's
cross-task mean direction (69-task mean of the task-level features), and treat the 11
residuals as the ablation subspace ("still treat them as 11 directions for now",
despite effective rank ~1.4).

Output: artifacts/69_task_run/bottom_up_ablation/meanremoved11_bases.pt
  {tasks: {task: {"V": (11, 4096) fp32 orthonormal rows (SVD of the 11 residuals,
                  descending singular value), "s": (11,) singular values,
                  "resid_fracs": (11,) per-layer ||resid||/||feature||,
                  "eff_rank": float participation ratio}},
   "rank": 11, "layers": [5..15], "source": "label_avg10_L5-15_acts"}

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
OUT = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "meanremoved11_bases.pt"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"


def main():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    feats, layers = {}, None
    for t in tasks:
        d = torch.load(ACTS / f"{t}.pt", map_location="cpu", weights_only=False)
        layers = d["layers"]
        feats[t] = d["acts"].double().mean(dim=0)          # (11, 4096) task-level feature
    X = torch.stack([feats[t] for t in tasks])             # (69, 11, 4096)
    mdirs = X.mean(dim=0)
    mdirs = mdirs / mdirs.norm(dim=1, keepdim=True)        # (11, 4096) per-layer mean dir
    R = X - (X * mdirs).sum(-1, keepdim=True) * mdirs      # mean-removed residuals

    out = {}
    effs = []
    for ti, t in enumerate(tasks):
        res = R[ti]                                        # (11, 4096)
        fracs = res.norm(dim=1) / X[ti].norm(dim=1)
        _, s, Vh = torch.linalg.svd(res, full_matrices=False)
        V = Vh                                             # (11, 4096) orthonormal
        e = (s ** 2) / (s ** 2).sum()
        eff = float(1 / (e ** 2).sum())
        g = V @ V.T - torch.eye(11, dtype=torch.float64)
        assert float(g.abs().max()) < 1e-8, f"{t}: basis not orthonormal"
        # basis must be orthogonal to each layer's mean direction only up to the small
        # cross-layer rotation; record worst-case leakage for the report
        leak = float((V @ mdirs.T).abs().max())
        out[t] = {"V": V.float(), "s": s.float(), "resid_fracs": fracs.float(),
                  "eff_rank": eff, "max_meandir_cos": leak}
        effs.append(eff)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tasks": out, "rank": 11, "layers": list(layers),
                "source": "label_avg10_L5-15_acts",
                "note": "mean-removed task-level features L5-15, SVD-orthonormalized"}, OUT)
    effs = torch.tensor(effs)
    leaks = torch.tensor([out[t]["max_meandir_cos"] for t in tasks])
    print(f"wrote {OUT}  ({len(out)} tasks)")
    print(f"effective rank: min={effs.min():.2f} median={effs.median():.2f} max={effs.max():.2f}")
    print(f"max |cos(basis, any layer mean dir)|: median={leaks.median():.3f} max={leaks.max():.3f}")


if __name__ == "__main__":
    main()
