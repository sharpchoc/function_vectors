#!/usr/bin/env python
"""SANDBOX: dimensionality summary figure for the sparse-23 per-prompt read directions.

ONE grid PNG (repo figure prefs): (A) per-task rank90 of each task's 170 stacked rank90-variant
read directions; (B) cumulative sigma^2 energy curves of the three pooled 4,930-row stacks
(per-prompt FVs v23, rank90 read dirs, literal read dirs), with SR / rank90 milestones.
Spectra computed here (sigma-only SVDs, CPU ~1 min); per-task rank90 recomputed from the banks.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT

FV_ROOT = ARTIFACTS_ROOT / "sandbox" / "sparse_head_selection" / "perprompt_fv_sparse23"
RD_ROOT = ARTIFACTS_ROOT / "sandbox" / "sparse_head_selection" / "perprompt_read_dirs_sparse23"
OUT = RESULTS_ROOT / "sandbox" / "sparse_head_selection"

INK = "#1E242B"
MUTED = "#5A6572"
FAINT = "#8B95A0"
HAIR = "#DDE2E7"
C_FV = "#0E7368"      # teal   - per-prompt FVs
C_R90 = "#3B5BA5"     # blue   - rank90 read dirs
C_LIT = "#B0613A"     # sienna - literal read dirs

tasks = sorted(p.stem for p in RD_ROOT.glob("*.pt") if p.stem != "build_summary")


def rank90_of(X):
    s = np.linalg.svd(X, compute_uv=False)
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(e, 0.90) + 1), s


fv_stack, r90_stack, lit_stack, per_task = [], [], [], []
for t in tasks:
    d = torch.load(RD_ROOT / f"{t}.pt", map_location="cpu", weights_only=False)
    f = torch.load(FV_ROOT / f"{t}.pt", map_location="cpu", weights_only=False)
    fv_stack.append(torch.cat([f["train"]["fvs"], f["test"]["fvs"]]))
    R = torch.cat([d["rank90"]["train"]["r"], d["rank90"]["test"]["r"]])
    L = torch.cat([d["literal"]["train"]["r"], d["literal"]["test"]["r"]])
    r90_stack.append(R)
    lit_stack.append(L)
    per_task.append((t, rank90_of(R.numpy().astype(np.float32))[0]))

curves = {}
for name, stack, color in (("per-prompt FVs (v23)", fv_stack, C_FV),
                           ("read dirs, rank90 variant", r90_stack, C_R90),
                           ("read dirs, literal variant", lit_stack, C_LIT)):
    X = torch.cat(stack).numpy().astype(np.float32)
    r90, s = rank90_of(X)
    sr = float((s ** 2).sum() / s[0] ** 2)
    curves[name] = {"energy": np.cumsum(s ** 2) / np.sum(s ** 2), "r90": r90, "sr": sr, "color": color}
    print(f"{name}: SR={sr:.1f} rank90={r90}")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 6.4), dpi=200,
                               gridspec_kw={"width_ratios": [1, 1.15]})
fig.patch.set_facecolor("white")

# --- A: per-task rank90 bars (rank90 variant) ---
per_task.sort(key=lambda x: x[1])
names = [t for t, _ in per_task]
vals = [v for _, v in per_task]
ys = range(len(names))
axA.barh(ys, vals, height=0.72, color=C_R90, zorder=3)
for y, v in zip(ys, vals):
    axA.text(v + 1.2, y, str(v), va="center", fontsize=7.5, color=INK)
axA.set_yticks(list(ys))
axA.set_yticklabels(names, fontsize=7.5)
axA.set_xlabel("rank90 of the task's 170 read directions", fontsize=9)
axA.set_title("Per-task spread of read directions\n(rank90 variant)", fontsize=10.5)
axA.grid(axis="x", color=HAIR, lw=0.7, zorder=0)
axA.spines[["top", "right"]].set_visible(False)
axA.tick_params(labelsize=8)
axA.set_xlim(0, 92)

# --- B: cumulative energy curves, pooled stacks ---
ks = np.arange(1, 501)
for name, c in curves.items():
    axB.plot(ks, c["energy"][:500] * 100, color=c["color"], lw=2, zorder=3)
for i, (name, c) in enumerate(curves.items()):
    y = 40 - i * 9
    axB.plot([255, 285], [y, y], color=c["color"], lw=2.5)
    axB.text(295, y, f"{name}  —  SR {c['sr']:.1f} · rank90 {c['r90']}",
             fontsize=8.5, color=c["color"], va="center", fontweight="bold")
axB.axhline(90, color=FAINT, ls=(0, (5, 4)), lw=1.1, zorder=2)
axB.text(6, 91.2, "90% energy", fontsize=8, color=MUTED)
for name in ("per-prompt FVs (v23)", "read dirs, rank90 variant"):
    c = curves[name]
    axB.plot([c["r90"]], [90], "o", color=c["color"], ms=5, zorder=4)
    axB.text(c["r90"], 86.5, str(c["r90"]), ha="center", fontsize=8, color=c["color"], fontweight="bold")
axB.set_xlim(1, 500)
axB.set_ylim(0, 102)
axB.set_xlabel("number of singular directions k", fontsize=9)
axB.set_ylabel("cumulative σ² energy (%)", fontsize=9)
axB.set_title("Pooled spectra — all 4,930 prompts, 29 tasks\n(one shared direction + task-structured tail)", fontsize=10.5)
axB.grid(color=HAIR, lw=0.7, zorder=0)
axB.spines[["top", "right"]].set_visible(False)
axB.tick_params(labelsize=8)
axB.margins(x=0)

fig.suptitle("SANDBOX per-prompt read directions (23-head sparse FV circuit, GPT-J) — dimensionality overview",
             fontsize=11, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.97))
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / "read_dirs_dimensionality.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
