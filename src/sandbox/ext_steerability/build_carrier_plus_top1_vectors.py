#!/usr/bin/env python
"""Steering vectors mirroring the ablation decomposition: carrier + norm-matched L5-7 top-1.

USER-ADJUDICATED 2026-09-02. Per task A (bank a, label_resid_means):
    mbar_A = mean over layers 5,6,7 of m_A(l)               (task read feature, L5-7 avg)
    c      = mean over all 69 tasks of mbar_A                (task-agnostic CARRIER)
    v1     = bankA/L5to7_top1_bases.pt unit direction        (the ablation's headline object)
    n_A    = <mbar_A - c, v1>                                (natural task-unique coordinate)
    u_A    = c + n_A * v1                                    (steering vector; sign-invariant)
Steering runs sweep one global alpha on u_A.

Note: the ablation removed each layer's mean DIRECTION before the SVD; here the carrier is
subtracted as the layer-averaged mean VECTOR. Immaterial for n_A because v1 is carrier-free
(|cos(v1, layer-mean dirs)| <= 0.05), so <c, v1> ~ 0 — printed below as a sanity check.

Output: artifacts/69_task_run/bottom_up_ablation/bankA/carrier_plus_top1_vectors.pt
  {tasks: {task: {"vec": (4096,) fp32, "n_A", "carrier_norm", "vec_norm", "c_dot_v1"}},
   "carrier": (4096,) fp32, "definition": ...}
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

RM = ARTIFACTS_ROOT / "69_task_run" / "label_resid_means"
BASES = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA" / "L5to7_top1_bases.pt"
OUT = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA" / "carrier_plus_top1_vectors.pt"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"
LAYERS = (5, 6, 7)


def main():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    bases = torch.load(BASES, map_location="cpu", weights_only=False)["tasks"]
    mbar = {}
    for t in tasks:
        rm = torch.load(RM / f"{t}.pt", map_location="cpu",
                        weights_only=False)["resid_means"].double()   # (28, 4096)
        mbar[t] = rm[list(LAYERS)].mean(dim=0)
    c = torch.stack([mbar[t] for t in tasks]).mean(dim=0)              # carrier (4096,)

    out = {}
    for t in tasks:
        v1 = bases[t]["V"][0].double()
        assert abs(float(v1.norm()) - 1.0) < 1e-4
        n_A = float((mbar[t] - c) @ v1)
        vec = c + n_A * v1
        out[t] = {"vec": vec.float(), "n_A": round(n_A, 4),
                  "carrier_norm": round(float(c.norm()), 3),
                  "vec_norm": round(float(vec.norm()), 3),
                  "c_dot_v1": round(float(c @ v1), 4)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tasks": out, "carrier": c.float(),
                "definition": "u_A = c + <mbar_A - c, v1> * v1; mbar_A = mean_{l=5,6,7} m_A(l) "
                              "(label_resid_means), c = 69-task mean of mbar_A, v1 = bankA "
                              "L5to7_top1 unit direction"}, OUT)
    n = torch.tensor([out[t]["n_A"] for t in tasks])
    cdv = torch.tensor([abs(out[t]["c_dot_v1"]) for t in tasks])
    vn = torch.tensor([out[t]["vec_norm"] for t in tasks])
    print(f"wrote {OUT} ({len(out)} tasks)")
    print(f"||c|| = {float(c.norm()):.2f} | n_A: min={n.min():.2f} median={n.median():.2f} "
          f"max={n.max():.2f} | |<c,v1>|/||c||: median={float((cdv / c.norm()).median()):.4f} "
          f"max={float((cdv / c.norm()).max()):.4f} | ||u_A|| median={vn.median():.2f}")


if __name__ == "__main__":
    main()
