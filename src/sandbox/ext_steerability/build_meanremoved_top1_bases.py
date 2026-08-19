#!/usr/bin/env python
"""Top-1 slice of the task-unique SVD bases (single top direction per task).

USER REQUEST 2026-08-19: rerun the task-unique ablation with ONLY the top singular
vector of the unit-normed 11 mean-removed directions. Derived by slicing
meanremoved_top3_bases.pt (same SVD, row 0), so the direction is bit-identical to the
top-3 run's first direction.

Output: artifacts/69_task_run/bottom_up_ablation/meanremoved_top1_bases.pt
  {tasks: {task: {"V": (1, 4096) fp32, "s": (11,), "energy_top1": float}},
   "rank": 1, "layers": [5..15], "source": "label_avg10_L5-15_acts"}
"""
import sys
from pathlib import Path

import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT  # noqa: E402

SRC = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "meanremoved_top3_bases.pt"
OUT = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "meanremoved_top1_bases.pt"


def main():
    src = torch.load(SRC, map_location="cpu", weights_only=False)
    out = {}
    energies = []
    for t, b in src["tasks"].items():
        s = b["s"].double()
        energy = float(s[0] ** 2 / (s ** 2).sum())
        v = b["V"][:1]
        assert abs(float(v.norm()) - 1.0) < 1e-5
        out[t] = {"V": v, "s": b["s"], "energy_top1": energy}
        energies.append(energy)
    torch.save({"tasks": out, "rank": 1, "layers": src["layers"],
                "source": src["source"],
                "note": "top-1 slice of meanremoved_top3_bases (same SVD)"}, OUT)
    energies = torch.tensor(energies)
    print(f"wrote {OUT}  ({len(out)} tasks)")
    print(f"top-1 energy fraction: min={energies.min():.4f} median={energies.median():.4f} "
          f"max={energies.max():.4f}")


if __name__ == "__main__":
    main()
