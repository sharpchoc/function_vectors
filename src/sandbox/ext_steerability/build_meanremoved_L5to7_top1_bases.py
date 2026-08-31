#!/usr/bin/env python
"""Top-1 task-unique bases built from layers 5-7 ONLY (band-restriction + rank-1 combo).

USER REQUEST 2026-08-31: same construction as build_meanremoved_L6to9_top3_bases.py but
with KEEP_LAYERS=(5, 6, 7) and RANK=1 — the single top singular vector of the 3
unit-normed mean-removed task-level directions. Tests whether one direction from a
narrow early band reproduces the L5-15 top-1 (own 0.103) / L6-9 top-3 (own 0.096)
ablation results. Ablation protocol unchanged (all 28 block inputs, demo-target tokens).

Output: artifacts/69_task_run/bottom_up_ablation/meanremoved_L5to7_top1_bases.pt
  {tasks: {task: {"V": (1, 4096), "s": (3,), "energy_top1": float}},
   "rank": 1, "layers": [5, 6, 7], "source": "label_avg10_L5-15_acts"}
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
OUT = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "meanremoved_L5to7_top1_bases.pt"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"
KEEP_LAYERS = (5, 6, 7)
RANK = 1


def main():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    feats, layers = {}, None
    for t in tasks:
        d = torch.load(ACTS / f"{t}.pt", map_location="cpu", weights_only=False)
        layers = list(d["layers"])
        feats[t] = d["acts"].double().mean(dim=0)
    idx = [layers.index(l) for l in KEEP_LAYERS]
    X = torch.stack([feats[t][idx] for t in tasks])        # (69, 3, 4096)
    mdirs = X.mean(dim=0)
    mdirs = mdirs / mdirs.norm(dim=1, keepdim=True)
    R = X - (X * mdirs).sum(-1, keepdim=True) * mdirs

    out, energies = {}, []
    for ti, t in enumerate(tasks):
        U = R[ti] / R[ti].norm(dim=1, keepdim=True)        # (3, 4096) unit-normed
        _, s, Vh = torch.linalg.svd(U, full_matrices=False)
        V = Vh[:RANK]
        energy = float((s[:RANK] ** 2).sum() / (s ** 2).sum())
        g = V @ V.T - torch.eye(RANK, dtype=torch.float64)
        assert float(g.abs().max()) < 1e-8, f"{t}: basis not orthonormal"
        out[t] = {"V": V.float(), "s": s.float(), "energy_top1": energy,
                  "max_meandir_cos": float((V @ mdirs.T).abs().max())}
        energies.append(energy)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tasks": out, "rank": RANK, "layers": list(KEEP_LAYERS),
                "source": "label_avg10_L5-15_acts",
                "note": "top-1 SVD of the 3 unit-normed mean-removed dirs, layers 5-7"}, OUT)
    energies = torch.tensor(energies)
    leaks = torch.tensor([out[t]["max_meandir_cos"] for t in tasks])
    print(f"wrote {OUT}  ({len(out)} tasks)")
    print(f"top-1 energy fraction (of 3): min={energies.min():.4f} "
          f"median={energies.median():.4f} max={energies.max():.4f}")
    print(f"max |cos(basis, any kept-layer mean dir)|: median={leaks.median():.3f} "
          f"max={leaks.max():.3f}")


if __name__ == "__main__":
    main()
