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
  l67_plus_meanresid_vectors.pt  w_A variant: 0.5*(m_A(6)+m_A(7)) + u_A  (own L6/7 mean plus the
                                  task-unique part once more) {tasks: {"vec","u_norm","base_norm",
                                  "vec_norm"}}
  meanresid_swap_bases.pt        for steer_taskunique_svd.py: {"V": (1,4096) unit u_hat,
                                  "s": [||u_A||]} so alpha=1 swaps in u_A at natural magnitude
With --bank b the same u_A is built from the ten-site-average estimator
(label_avg10_L5-15_acts) and written to bottom_up_ablation/meanresid_top1_bases_avg10.pt
(Appendix K estimator comparison only).
"""
import argparse
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


def load_bank_b(tasks):
    """(69, 3, d) L5-7 task means from the ten-site-average estimator."""
    root = ARTIFACTS_ROOT / "69_task_run" / "label_avg10_L5-15_acts"
    rows = []
    for t in tasks:
        d = torch.load(root / f"{t}.pt", map_location="cpu", weights_only=False)
        idx = [list(d["layers"]).index(l) for l in LAYERS]
        rows.append(d["acts"].double().mean(dim=0)[idx])
    return torch.stack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", choices=("a", "b"), default="a")
    args = ap.parse_args()
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    if args.bank == "b":
        X = load_bank_b(tasks)
        cd = X.mean(0); cd = cd / cd.norm(dim=1, keepdim=True)
        U = (X - (X * cd).sum(-1, keepdim=True) * cd).mean(dim=1)
        ref = torch.load(BA / "meanresid_top1_bases.pt", map_location="cpu", weights_only=False)["tasks"]
        bases, cs = {}, []
        for i, t in enumerate(tasks):
            uh = U[i] / U[i].norm()
            cos = float(abs(uh @ ref[t]["V"][0].double()))
            bases[t] = {"V": uh.float().unsqueeze(0), "norm_u": round(float(U[i].norm()), 3),
                        "cos_to_bankA_uhat": round(cos, 4)}
            cs.append(cos)
        out = BA.parent / "meanresid_top1_bases_avg10.pt"
        torch.save({"tasks": bases, "rank": 1, "layers": list(LAYERS), "source": "label_avg10_L5-15_acts",
                    "note": "bank-(b) estimator variant of the mean carrier-removed L5-7 residual"}, out)
        cs = torch.tensor(cs)
        print(f"wrote {out.name}: |cos(u_hat^avg10, u_hat^final)| median {cs.median():.4f} "
              f"min {cs.min():.4f} max {cs.max():.4f}")
        return
    X = torch.stack([torch.load(RM / f"{t}.pt", map_location="cpu",
                                weights_only=False)["resid_means"][list(LAYERS)].double()
                     for t in tasks])                                # (69, 3, d)
    M67 = torch.stack([torch.load(RM / f"{t}.pt", map_location="cpu",
                                  weights_only=False)["resid_means"][[6, 7]].double().mean(0)
                       for t in tasks])                              # (69, d) own L6/7 mean
    cd = X.mean(0)
    cd = cd / cd.norm(dim=1, keepdim=True)                           # per-layer carrier dirs
    R = X - (X * cd).sum(-1, keepdim=True) * cd                      # carrier projected out
    U = R.mean(dim=1)                                                # (69, d) task-unique vectors
    mbar = X.mean(dim=1)
    c = mbar.mean(0)                                                 # layer-averaged carrier

    svd = torch.load(BA / "L5to7_top1_bases.pt", map_location="cpu", weights_only=False)["tasks"]
    bases, vecs, wvecs, swap, cs, un = {}, {}, {}, {}, [], []
    for i, t in enumerate(tasks):
        u = U[i]
        uh = u / u.norm()
        wv = M67[i] + u
        wvecs[t] = {"vec": wv.float(), "u_norm": round(float(u.norm()), 3),
                    "base_norm": round(float(M67[i].norm()), 3), "vec_norm": round(float(wv.norm()), 3)}
        swap[t] = {"V": uh.float().unsqueeze(0), "s": torch.tensor([float(u.norm())])}
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
    torch.save({"tasks": wvecs, "definition": "0.5*(m_A(6)+m_A(7)) + u_A (own L6/7 mean plus "
                "the mean carrier-removed L5-7 residual); bank a"}, BA / "l67_plus_meanresid_vectors.pt")
    torch.save({"tasks": swap, "rank": 1, "layers": list(LAYERS), "source": "label_resid_means",
                "note": "V = u_hat_A, s = ||u_A||: steer_taskunique_svd.py swaps in alpha*u_A"},
               BA / "meanresid_swap_bases.pt")
    cs, un = torch.tensor(cs), torch.tensor(un)
    print(f"wrote meanresid_top1_bases.pt + carrier_plus_meanresid_vectors.pt + "
          f"l67_plus_meanresid_vectors.pt + meanresid_swap_bases.pt ({len(tasks)} tasks)")
    print(f"|cos(u_hat, svd v1)|: median {cs.median():.4f} min {cs.min():.4f} | ||u_A||: median "
          f"{un.median():.2f} (n_A median was 28.2) | ||c|| {float(c.norm()):.2f}")


if __name__ == "__main__":
    main()
