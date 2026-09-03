#!/usr/bin/env python
"""Cross-task similarity of the NEW task-unique read direction vs the centered write feature.

Companion to understanding_read_write_linear_map/centered_cossim_hists.py, but the read side
is the Claim-3 task-unique object rather than the grand-mean-centered L6 mean:
  read : v1_A, the top SVD direction of the carrier-removed L5-7 read features
         (bankA/L5to7_top1_bases.pt), oriented along the task's own carrier-removed L5-7
         mean (SVD sign is arbitrary; sign = sign(n_A), n_A from carrier_plus_top1_vectors.pt)
  write: v_A - mean_A' v_A', the task FV centered on the 69-task grand mean (as before)
Pairwise cosines over all 2346 task pairs, plus the pair-by-pair agreement between the two
families (Pearson / Spearman of the two 2346-vectors), which is the "same shape" test used
for the rotation argument.

Writes results/69_task_run/understanding_read_write_linear_map/ctop1/:
  taskunique_vs_write_cossim.png, cossim_summary.csv, pairwise_cos.npz
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR  # noqa: E402

import os
_MR = os.environ.get("MEANRESID") == "1"   # mean-residual task-unique part u_hat_A instead of SVD v1
BASES = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA" / ("meanresid_top1_bases.pt" if _MR else "L5to7_top1_bases.pt")
VECS = ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA" / ("carrier_plus_meanresid_vectors.pt" if _MR else "carrier_plus_top1_vectors.pt")
UL = "\\hat u_A" if _MR else "v_1"
FV_ROOT = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
SPLIT = REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"
OUT = TASK69_RUN_DIR / "understanding_read_write_linear_map" / ("meanresid" if _MR else "ctop1")


def pairwise_cos(X):
    Xn = X / X.norm(dim=1, keepdim=True)
    G = (Xn @ Xn.T).numpy()
    iu = np.triu_indices(len(X), k=1)
    return G[iu]


def main():
    split = json.load(open(SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    bases = torch.load(BASES, map_location="cpu", weights_only=False)["tasks"]
    vecs = torch.load(VECS, map_location="cpu", weights_only=False)["tasks"]
    V1 = torch.stack([bases[t]["V"][0].double() * (1.0 if _MR else np.sign(vecs[t]["n_A"])) for t in tasks])  # signed (u_hat_A already oriented)
    FV = torch.stack([torch.load(FV_ROOT / f"{t}.pt", map_location="cpu",
                                 weights_only=False)["fv"].double().mean(0) for t in tasks])
    FVc = FV - FV.mean(0, keepdim=True)

    cr = pairwise_cos(V1)
    cw = pairwise_cos(FVc)
    cw_raw = pairwise_cos(FV)
    pr, sr = pearsonr(cr, cw)[0], spearmanr(cr, cw)[0]

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "pairwise_cos.npz", tasks=np.array(tasks), read_v1=cr, write_centered=cw,
             write_raw=cw_raw)
    with open(OUT / "cossim_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "mean", "median", "sd", "min", "max"])
        for name, a in (("read_taskunique_v1", cr), ("write_centered", cw), ("write_raw", cw_raw)):
            w.writerow([name, f"{a.mean():.4f}", f"{np.median(a):.4f}", f"{a.std():.4f}",
                        f"{a.min():.4f}", f"{a.max():.4f}"])
        w.writerow(["pair_agreement_pearson", f"{pr:.4f}", "", "", "", ""])
        w.writerow(["pair_agreement_spearman", f"{sr:.4f}", "", "", "", ""])

    TEAL, BLUE = "#2f9e8a", "#4472d6"
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=180,
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})
    bins = np.linspace(-1, 1, 81)
    for ax, a, col, title in ((axes[0], cr, TEAL,
                               f"Read: task-unique direction ${UL}$\n(carrier removed, L5–7)"),
                              (axes[1], cw, BLUE,
                               "Write: function vector $v_A$, centered\n"
                               f"(uncentered mean {cw_raw.mean():.3f})")):
        ax.hist(a, bins=bins, color=col)
        ax.axvline(a.mean(), color="k", ls="--", lw=1.4,
                   label=f"mean {a.mean():.3f}\nmedian {np.median(a):.3f}")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"pairwise cosine ({len(a)} task pairs)")
        ax.legend(frameon=False, fontsize=10)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].set_ylabel("pairs")
    ax = axes[2]
    ax.scatter(cr, cw, s=6, alpha=0.35, color="0.3", linewidths=0)
    lim = (-0.6, 1.0)
    ax.plot(lim, lim, color="0.7", lw=1, ls=":")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"read: cos(${UL}^A$, ${UL}^B$)")
    ax.set_ylabel("write: cos(centered $v_A$, centered $v_B$)")
    ax.set_title(f"Same pair, both families\nPearson {pr:.2f}, Spearman {sr:.2f}", fontsize=12)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.suptitle("Cross-task similarity: task-unique read direction vs centered write feature (69 tasks)",
                 fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "taskunique_vs_write_cossim.png")
    print(f"read v1: mean {cr.mean():.3f} median {np.median(cr):.3f} | write centered: mean {cw.mean():.3f} "
          f"median {np.median(cw):.3f} | pair agreement Pearson {pr:.3f} Spearman {sr:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
