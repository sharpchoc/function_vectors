#!/usr/bin/env python
"""Simplified task-unique object (USER PROPOSAL 2026-09-02): mean of the carrier-removed
L5-7 read features, WITH its natural magnitude — no SVD, no unit-norming, no n_A.

Per task A (bank a, label_resid_means), layers l in {5, 6, 7}:
    c_hat(l) = unit cross-task mean direction at layer l
    r_A(l)   = m_A(l) - <m_A(l), c_hat(l)> c_hat(l)        (carrier projected out)
    u_A      = (1/3) * sum_l r_A(l)                          (task-unique VECTOR, natural size)
    c        = (1/69) * sum_A' mbar_A'   with mbar_A' = (1/3) sum_l m_A'(l)   (carrier)
Ablation direction = u_A / ||u_A||  (ablate_readdir_pc5.py needs a rank-1 unit basis).
Steering vector    = c + u_A        (no separate coefficient; alpha scales the whole).

Outputs (bankA/):
  meanresid_top1_bases.pt        {tasks: {task: {"V": (1,4096) unit, "norm_u": float,
                                  "cos_to_svd_v1": float}}, "rank": 1, ...}
  carrier_plus_meanresid_vectors.pt  {tasks: {task: {"vec", "u_norm", "carrier_norm",
                                  "vec_norm"}}, "carrier", "definition"}
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
BA = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"
LAYERS = (5, 6, 7)


def main():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    X = torch.stack([torch.load(RM / f"{t}.pt", map_location="cpu",
                                weights_only=False)["resid_means"][list(LAYERS)].double()
                     for t in tasks])                                # (69, 3, d)
    cd = X.mean(0)
    cd = cd / cd.norm(dim=1, keepdim=True)                           # per-layer carrier dirs
    R = X - (X * cd).sum(-1, keepdim=True) * cd                      # carrier projected out
    U = R.mean(dim=1)                                                # (69, d) task-unique vectors
    mbar = X.mean(dim=1)
    c = mbar.mean(0)                                                 # layer-averaged carrier

    svd = torch.load(BA / "L5to7_top1_bases.pt", map_location="cpu", weights_only=False)["tasks"]
    bases, vecs, cs, un = {}, {}, [], []
    for i, t in enumerate(tasks):
        u = U[i]
        uh = u / u.norm()
        cos_svd = float(abs(uh @ svd[t]["V"][0].double()))
        bases[t] = {"V": uh.float().unsqueeze(0), "norm_u": round(float(u.norm()), 3),
                    "cos_to_svd_v1": round(cos_svd, 4),
                    "max_carrier_cos": round(float((uh @ cd.T).abs().max()), 4)}
        vec = c + u
        vecs[t] = {"vec": vec.float(), "u_norm": round(float(u.norm()), 3),
                   "carrier_norm": round(float(c.norm()), 3),
                   "vec_norm": round(float(vec.norm()), 3)}
        cs.append(cos_svd); un.append(float(u.norm()))
    BA.mkdir(parents=True, exist_ok=True)
    torch.save({"tasks": bases, "rank": 1, "layers": list(LAYERS), "source": "label_resid_means",
                "note": "unit direction of the mean carrier-removed L5-7 residual (no SVD)"},
               BA / "meanresid_top1_bases.pt")
    torch.save({"tasks": vecs, "carrier": c.float(),
                "definition": "c + u_A, u_A = mean_l [m_A(l) - <m_A(l),c_hat(l)> c_hat(l)], "
                              "l in 5..7; c = 69-task mean of the L5-7 mean read feature"},
               BA / "carrier_plus_meanresid_vectors.pt")
    cs, un = torch.tensor(cs), torch.tensor(un)
    print(f"wrote meanresid_top1_bases.pt + carrier_plus_meanresid_vectors.pt ({len(tasks)} tasks)")
    print(f"|cos(u_hat, svd v1)|: median {cs.median():.4f} min {cs.min():.4f} | ||u_A||: median "
          f"{un.median():.2f} (n_A median was 28.2) | ||c|| {float(c.norm()):.2f}")


if __name__ == "__main__":
    main()
