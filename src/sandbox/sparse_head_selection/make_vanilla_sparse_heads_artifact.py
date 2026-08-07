#!/usr/bin/env python
"""SANDBOX: write the vanilla_sparse_opt23 head-selection artifact (the 23 heads with c > 0.8
from the sparse-optimization run). The 'vanilla sparse optimisation FV' is the UNWEIGHTED sum
of these heads' varicl mean outputs -- coefficients select heads only. NOT a repo default."""
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import ARTIFACTS_ROOT

THRESHOLD = 0.8
sel_root = ARTIFACTS_ROOT / "sandbox" / "sparse_head_selection"
d = torch.load(sel_root / "coeffs_final.pt", map_location="cpu", weights_only=False)
c = d["c"].view(28, 16)
heads = sorted(((l, h, round(c[l, h].item(), 6)) for l in range(28) for h in range(16)
                if c[l, h] > THRESHOLD), key=lambda t: -t[2])
assert len(heads) == 23, f"expected 23 heads above {THRESHOLD}, got {len(heads)}"

artifact = {
    "top_heads": heads,
    "sandbox": True,
    "name": "vanilla_sparse_opt23",
    "note": ("SANDBOX 'vanilla sparse optimisation FV' head set: the 23 heads with c > 0.8 from "
             "the sparse-optimization selection (lambda=0.01, LOTO CV; see selection.json). "
             "FV construction is the UNWEIGHTED sum of these heads' varicl mean outputs "
             "(coefficients used for selection only). NOT a repo-default head set."),
    "source": str(sel_root / "coeffs_final.pt"),
    "threshold": THRESHOLD,
}
out = sel_root / "vanilla_sparse_opt23_heads.pt"
torch.save(artifact, out)
with open(sel_root / "vanilla_sparse_opt23_heads.json", "w") as f:
    json.dump(artifact, f, indent=2)
print(f"wrote {out} ({len(heads)} heads); top-5: {heads[:5]}")
