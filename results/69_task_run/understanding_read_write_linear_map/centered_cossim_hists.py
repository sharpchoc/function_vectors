"""Centered pairwise cosine similarity of task-level read features vs write features.

Before fitting/interpreting any read->write linear map: how similar are the 69
task-level vectors within each family once the shared (grand-mean) component is
removed?

- Read feature m_A(L):  artifacts/69_task_run/label_resid_means/<task>.pt,
  resid_means[L] (task-mean label-token residual; L6 = canonical read layer,
  L13 = linear-map peak layer, kept as a secondary variant in the CSV/npz).
- Write feature v_A:    artifacts/69_task_run/perprompt_fvs/<task>.pt,
  fv.mean(0) over the 150 per-prompt FVs (task FV).
- Centering: subtract the equal-weighted mean across the 69 task vectors
  (each family centered on its own grand mean). Cosines over all 2346 pairs.

Outputs (this folder): centered_cossim_hists.png, cossim_summary.csv,
pairwise_cos.npz. Uncentered values are computed alongside for reference
(should reproduce bottom_up_read_features/ablation/debugging: read .727, fv .393).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT  # noqa: E402

READ_ROOT = ARTIFACTS_ROOT / "69_task_run" / "label_resid_means"
FV_ROOT = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
OUT = RESULTS_ROOT / "69_task_run" / "understanding_read_write_linear_map"
READ_LAYERS = [6, 13]

tasks = sorted(p.stem for p in READ_ROOT.glob("*.pt"))
assert len(tasks) == 69, f"expected 69 tasks, got {len(tasks)}"

reads = {L: [] for L in READ_LAYERS}
fvs = []
for t in tasks:
    r = torch.load(READ_ROOT / f"{t}.pt", map_location="cpu", weights_only=False)
    for L in READ_LAYERS:
        reads[L].append(r["resid_means"][L].double().numpy())
    f = torch.load(FV_ROOT / f"{t}.pt", map_location="cpu", weights_only=False)
    fv = f["fv"]
    assert fv.shape == (150, 4096), fv.shape
    fvs.append(fv.double().mean(0).numpy())

families = {f"read_L{L}": np.stack(reads[L]) for L in READ_LAYERS}
families["write_fv"] = np.stack(fvs)


def pairwise_cos(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = Xn @ Xn.T
    iu = np.triu_indices(len(X), k=1)
    return C[iu]


rows = []
store = {"tasks": np.array(tasks)}
for name, X in families.items():
    for variant, Xv in [("uncentered", X), ("centered", X - X.mean(0))]:
        c = pairwise_cos(Xv)
        store[f"{name}__{variant}"] = c
        rows.append(
            (name, variant, len(c), c.mean(), np.median(c),
             np.percentile(c, 5), np.percentile(c, 95), c.min(), c.max())
        )

with open(OUT / "cossim_summary.csv", "w") as fh:
    fh.write("family,variant,n_pairs,mean,median,p5,p95,min,max\n")
    for r in rows:
        fh.write(",".join([r[0], r[1], str(r[2])] + [f"{v:.4f}" for v in r[3:]]) + "\n")
np.savez_compressed(OUT / "pairwise_cos.npz", **store)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
bins = np.linspace(-1, 1, 81)
panels = [
    ("read_L6", "Read feature $m_A$(L6), centered", "#0e7c6b"),
    ("write_fv", "Write feature $v_A$ (task FV), centered", "#2a5fd1"),
]
for ax, (name, title, color) in zip(axes, panels):
    c = store[f"{name}__centered"]
    ax.hist(c, bins=bins, color=color, alpha=0.85)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.axvline(c.mean(), color="k", lw=1.2, ls="--",
               label=f"mean {c.mean():.3f}\nmedian {np.median(c):.3f}")
    u = store[f"{name}__uncentered"]
    ax.set_title(f"{title}\n(uncentered mean {u.mean():.3f})", fontsize=10)
    ax.set_xlabel("pairwise cosine (2346 task pairs)")
    ax.legend(frameon=False, fontsize=9)
axes[0].set_ylabel("pairs")
fig.suptitle("Cross-task similarity after removing each family's grand mean (69 tasks)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "centered_cossim_hists.png", dpi=180)
print(open(OUT / "cossim_summary.csv").read())
