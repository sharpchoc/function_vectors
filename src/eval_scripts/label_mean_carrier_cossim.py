#!/usr/bin/env python
"""Shared component of label-token means vs function vectors, GPT-J and Qwen2.5-7B-Instruct (paper Appendix C).

Backs the main-text claim that label-token task means m_A(l) "share a significant common component", which
motivates splitting them into a shared carrier c and a task-specific component u_A. For each model, over all
tasks of its pool, we compare the distributions of pairwise (uncentered) cosines between tasks of:
  m_A(L_id)  label-token task mean at the identification layer (GPT-J 6, Qwen 12)
  v_A        task FV (mean of the per-prompt FVs)
  u_A        carrier-removed band average (paper definition: training-task carrier g_l per layer of the
             extraction band B, u_A = mean_{l in B} [m_A(l) - g_hat_l g_hat_l^T m_A(l)])
Layers are zero-indexed block outputs (resid_means[l]).

Writes <model results dir>/label_mean_cossim/{summary.csv, pairwise_cos.npz} and the paper figure
paper_materials/uploads/figures/appendix_c_identification_controls/label_mean_cossim.pdf. CPU only.
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

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import (ARTIFACTS_ROOT, REPO_ROOT, TASK69_RUN_DIR, QWEN25_FV_DIR,  # noqa: E402
                             QWEN25_READ_ARTIFACTS_DIR, QWEN25_SPLIT)
from utils.paper_style import apply_paper_style, C  # noqa: E402

CFG = {
    "GPT-J": dict(means=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means",
                  fv=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs",
                  split=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json",
                  band=(5, 6, 7), id_layer=6,
                  out=TASK69_RUN_DIR / "understanding_read_write_linear_map" / "label_mean_cossim"),
    "Qwen2.5": dict(means=QWEN25_READ_ARTIFACTS_DIR / "label_resid_means",
                    fv=QWEN25_READ_ARTIFACTS_DIR / "perprompt_fvs",
                    split=QWEN25_SPLIT, band=(11, 12, 13), id_layer=12,
                    out=QWEN25_FV_DIR / "read_write_map" / "label_mean_cossim"),
}
FIG = REPO_ROOT / "paper_materials" / "uploads" / "figures" / "appendix_c_identification_controls" / "label_mean_cossim.pdf"
NAMES = {"label_mean": "label mean $m_A$", "task_fv": "task FV $v_A$", "task_specific_u": "task-specific $u_A$"}


def pairwise_cos(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return (Xn @ Xn.T)[np.triu_indices(len(X), k=1)]


def model_vectors(cfg):
    split = json.load(open(cfg["split"]))
    train, test = sorted(split["train_tasks"]), sorted(split["heldout_tasks"])
    tasks = sorted(train + test)
    band = list(cfg["band"])
    M = {t: torch.load(cfg["means"] / f"{t}.pt", map_location="cpu", weights_only=False)["resid_means"].double().numpy()
         for t in tasks}                                                     # (28, d) per task
    V = np.stack([torch.load(cfg["fv"] / f"{t}.pt", map_location="cpu", weights_only=False)["fv"].double().mean(0).numpy()
                  for t in tasks])
    g = np.stack([M[t][band] for t in train]).mean(0)                         # (|B|, d) training carrier per layer
    gh = g / np.linalg.norm(g, axis=1, keepdims=True)
    U = np.stack([(M[t][band] - (M[t][band] * gh).sum(1, keepdims=True) * gh).mean(0) for t in tasks])
    Mid = np.stack([M[t][cfg["id_layer"]] for t in tasks])
    return tasks, {"label_mean": Mid, "task_fv": V, "task_specific_u": U}


def main():
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    colors = {"label_mean": C.read, "task_fv": C.write, "task_specific_u": C.accent}
    bins = np.linspace(-0.4, 1.0, 71)
    for ax, (model, cfg) in zip(axes, CFG.items()):
        tasks, vecs = model_vectors(cfg)
        cfg["out"].mkdir(parents=True, exist_ok=True)
        rows, store = [], {"tasks": np.array(tasks)}
        for key, X in vecs.items():
            c = pairwise_cos(X)
            store[key] = c
            rows.append(dict(model=model, vector=key, n_tasks=len(tasks), n_pairs=len(c), mean=c.mean(),
                             median=np.median(c), p5=np.percentile(c, 5), p95=np.percentile(c, 95)))
            ax.hist(c, bins=bins, density=True, color=colors[key], alpha=0.55,
                    label=f"{NAMES[key]} (mean {c.mean():.2f})")
        with open(cfg["out"] / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        np.savez_compressed(cfg["out"] / "pairwise_cos.npz", **store)
        for r in rows:
            print(model, r["vector"], r["n_tasks"], round(r["mean"], 3), round(r["median"], 3),
                  round(r["p5"], 3), round(r["p95"], 3))
        ax.set_title(f"{model} ({len(tasks)} tasks, layer {cfg['id_layer']})", loc="left")
        ax.set_xlabel("pairwise cosine between tasks")
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        ax.set_yticks([])
    axes[0].set_ylabel("density")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
