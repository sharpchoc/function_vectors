#!/usr/bin/env python
"""Steering vectors: task mean averaged over L6+L7, plus the L5-7 top-1 direction on top.

USER REQUEST 2026-08-31 (adjudicated): per task A,
    base_A = 0.5 * (m_A(L6) + m_A(L7))          (raw target-token means, block outputs)
    n_A    = <base_A, v1_A>                      (natural projection onto the unit top-1
                                                  L5-7 task-unique direction v1_A)
    w_A    = base_A + n_A * v1_A                 (doubles the task-unique component)
The steering runs sweep one global alpha on w_A.

Output: artifacts/69_task_run/bottom_up_ablation/l67_plus_top1_vectors.pt
  {tasks: {task: {"vec": (4096,) fp32, "n_A": float, "base_norm": float, "vec_norm": float}},
   "definition": ...}
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
BASES = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "meanremoved_L5to7_top1_bases.pt"
OUT = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "l67_plus_top1_vectors.pt"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"


def main():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    bases = torch.load(BASES, map_location="cpu", weights_only=False)["tasks"]
    out = {}
    for t in tasks:
        rm = torch.load(RM / f"{t}.pt", map_location="cpu",
                        weights_only=False)["resid_means"].double()   # (28, 4096)
        base = 0.5 * (rm[6] + rm[7])
        v1 = bases[t]["V"][0].double()
        assert abs(float(v1.norm()) - 1.0) < 1e-4
        n_A = float(base @ v1)
        vec = base + n_A * v1
        out[t] = {"vec": vec.float(), "n_A": round(n_A, 4),
                  "base_norm": round(float(base.norm()), 3),
                  "vec_norm": round(float(vec.norm()), 3)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tasks": out,
                "definition": "0.5*(m_A(L6)+m_A(L7)) + <base, v1>*v1, v1 = L5-7 top-1 "
                              "task-unique unit direction (meanremoved_L5to7_top1_bases)"}, OUT)
    n = torch.tensor([out[t]["n_A"] for t in tasks])
    b = torch.tensor([out[t]["base_norm"] for t in tasks])
    print(f"wrote {OUT} ({len(out)} tasks)")
    print(f"n_A: min={n.min():.2f} median={n.median():.2f} max={n.max():.2f} "
          f"| base norm median={b.median():.1f} | n_A/base median={float((n/b).median()):.4f}")


if __name__ == "__main__":
    main()
