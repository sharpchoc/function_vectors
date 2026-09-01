#!/usr/bin/env python
"""Bank-(a) rebuild of ALL task-unique ablation bases (read-feature convention migration).

USER DECISION 2026-09-01: the read feature standardises on bank (a) = label_resid_means
(final-demo-target last token, mean over the 150 clean 10-shot prompts, all 28 layers).
This script rebuilds the whole ablation-ladder basis family from bank (a) task means,
construction-identical to the original label_avg10 builders:

  mr11        layers 5-15, SVD of the RAW mean-removed residuals, keep 11 (orthonormal)
  top3        layers 5-15, SVD of the UNIT-NORMED residuals, keep 3
  top1        layers 5-15, SVD of the UNIT-NORMED residuals, keep 1
  L6to9_top3  layers 6-9,  SVD of the UNIT-NORMED residuals, keep 3
  L5to7_top1  layers 5-7,  SVD of the UNIT-NORMED residuals, keep 1

Mean removal: per layer, project out the cross-task (69-task) mean direction of the
bank-(a) means. SVD on CPU float64 (CUDA gesvdj inaccuracy; see DECISIONS).

Outputs: artifacts/69_task_run/bottom_up_ablation/bankA/<name>_bases.pt
  {tasks: {task: {"V": (rank, 4096) fp32, "s": (...) fp32, ...}},
   "rank", "layers", "source": "label_resid_means", "unitnorm": bool}
Also: bankA/l67_plus_top1_vectors.pt — w_A = 0.5*(m_A(6)+m_A(7)) + <base, v1>*v1 with v1
from the bank-(a) L5to7_top1 basis (base was already bank (a); only v1 changes).
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
OUTDIR = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"

CONFIGS = [
    ("mr11",       tuple(range(5, 16)), 11, False),
    ("top3",       tuple(range(5, 16)), 3,  True),
    ("top1",       tuple(range(5, 16)), 1,  True),
    ("L6to9_top3", (6, 7, 8, 9),        3,  True),
    ("L5to7_top1", (5, 6, 7),           1,  True),
]


def main():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    rm = {t: torch.load(RM / f"{t}.pt", map_location="cpu",
                        weights_only=False)["resid_means"].double() for t in tasks}
    OUTDIR.mkdir(parents=True, exist_ok=True)

    for name, layers, rank, unitnorm in CONFIGS:
        X = torch.stack([rm[t][list(layers)] for t in tasks])   # (69, nL, 4096)
        mdirs = X.mean(dim=0)
        mdirs = mdirs / mdirs.norm(dim=1, keepdim=True)
        R = X - (X * mdirs).sum(-1, keepdim=True) * mdirs
        out = {}
        for ti, t in enumerate(tasks):
            res = R[ti]
            src = res / res.norm(dim=1, keepdim=True) if unitnorm else res
            _, s, Vh = torch.linalg.svd(src, full_matrices=False)
            V = Vh[:rank]
            g = V @ V.T - torch.eye(rank, dtype=torch.float64)
            assert float(g.abs().max()) < 1e-8, f"{name}/{t}: basis not orthonormal"
            e = (s ** 2) / (s ** 2).sum()
            out[t] = {"V": V.float(), "s": s.float(),
                      "energy_kept": float(e[:rank].sum()),
                      "eff_rank": float(1 / (e ** 2).sum()),
                      "max_meandir_cos": float((V @ mdirs.T).abs().max()),
                      "resid_fracs": (res.norm(dim=1) / X[ti].norm(dim=1)).float()}
        p = OUTDIR / f"{name}_bases.pt"
        torch.save({"tasks": out, "rank": rank, "layers": list(layers),
                    "source": "label_resid_means", "unitnorm": unitnorm,
                    "note": f"bank-(a) rebuild, construction-identical to the "
                            f"label_avg10 {name} basis"}, p)
        en = torch.tensor([out[t]["energy_kept"] for t in tasks])
        lk = torch.tensor([out[t]["max_meandir_cos"] for t in tasks])
        print(f"{name}: wrote {p.name} | energy kept median={en.median():.3f} "
              f"| carrier leak max={lk.max():.3f}")

    # w_A rebuild with the bank-(a) v1
    v1b = torch.load(OUTDIR / "L5to7_top1_bases.pt", map_location="cpu",
                     weights_only=False)["tasks"]
    wout = {}
    for t in tasks:
        base = 0.5 * (rm[t][6] + rm[t][7])
        v1 = v1b[t]["V"][0].double()
        n_A = float(base @ v1)
        vec = base + n_A * v1
        wout[t] = {"vec": vec.float(), "n_A": round(n_A, 4),
                   "base_norm": round(float(base.norm()), 3),
                   "vec_norm": round(float(vec.norm()), 3)}
    p = OUTDIR / "l67_plus_top1_vectors.pt"
    torch.save({"tasks": wout,
                "definition": "0.5*(m_A(L6)+m_A(L7)) + <base,v1>*v1, v1 = bank-(a) "
                              "L5to7_top1 unit direction"}, p)
    n = torch.tensor([wout[t]["n_A"] for t in tasks])
    print(f"w_A: wrote {p.name} | n_A median={n.median():.2f} "
          f"| |n_A|/base median={float((n.abs()/torch.tensor([wout[t]['base_norm'] for t in tasks])).median()):.4f}")

    # how close are the bank-(a) v1s to the old bank-(b) v1s?
    old = torch.load(ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation"
                     / "meanremoved_L5to7_top1_bases.pt", map_location="cpu",
                     weights_only=False)["tasks"]
    cos = torch.tensor([float((v1b[t]["V"][0].double() @ old[t]["V"][0].double()).abs())
                        for t in tasks])
    print(f"|cos(v1_bankA, v1_bankB)|: min={cos.min():.3f} median={cos.median():.3f} "
          f"max={cos.max():.3f}")


if __name__ == "__main__":
    main()
