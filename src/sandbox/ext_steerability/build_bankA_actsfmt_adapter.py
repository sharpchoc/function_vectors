#!/usr/bin/env python
"""Adapter: bank-(a) task means in the label_avg10 acts format (migration 2026-09-01).

steer_taskunique_svd.py and steer_twoknob_dummy.py consume acts files only via
d["acts"].mean(dim=0) and d["layers"]. This writes per-task files with
acts = (1, 11, 4096) = label_resid_means rows 5..15, so those scripts compute their
sign-fixing / natural coordinates from bank (a) with zero code changes.

Output: artifacts/69_task_run/bottom_up_ablation/bankA/actsfmt/<task>.pt
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
OUT = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA" / "actsfmt"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"
LAYERS = list(range(5, 16))


def main():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    OUT.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        rm = torch.load(RM / f"{t}.pt", map_location="cpu",
                        weights_only=False)["resid_means"]
        torch.save({"acts": rm[LAYERS].unsqueeze(0).float(), "layers": LAYERS,
                    "note": "bank-(a) adapter: acts.mean(0) == label_resid_means rows"},
                   OUT / f"{t}.pt")
    print(f"wrote {len(tasks)} adapter files to {OUT}")


if __name__ == "__main__":
    main()
