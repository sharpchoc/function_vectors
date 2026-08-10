#!/usr/bin/env python
"""Histograms of per-prompt function-vector norms ||v^j_A|| across 27 tasks (CPU).

v^j_A = sum_{h in H} W_O[:, h_slice] @ z_j(l,h) -- the per-prompt FV over the pooled
top-40 varicl heads, built from the per-prompt head activations captured by
capture_perprompt_head_activations.py (full-pool queries, 170 prompts/task, variants
fixed10 and varicl4to10). Norm = L2 of the 4096-dim vector, math in fp64.

Figures (in FV_FORMATION_DIR/attention_head_analysis/perprompt_fv_norms/):
  1. fvnorm_hist_pooled_<variant>.png  -- one histogram over all 27x170 norms
  2. fvnorm_hist_pertask_<variant>.png -- 27-panel grid (shared bins), panels ordered by
     median norm, + per-task median +/- IQR summary panel
  3. fvnorm_vs_shots.png               -- supplementary: norm vs shot count (varicl4to10),
     with fixed10 medians as the 10-shot reference
Data: fvnorm_perprompt_<variant>.npz + fvnorm_summary.csv (regenerate views on request).

Advisory anchor: per task, ||mean_j v^j_A|| must equal the norm of the FV rebuilt from
the capture's own task-mean activations (linearity check, dev < 1e-6 relative).
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR

HD = 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture_root", type=Path,
                   default=ARTIFACTS_ROOT / "perprompt_head_activations" / "gptj_27tasks_170prompts")
    p.add_argument("--pooled_heads_path", type=Path,
                   default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl" / "multitask_top_aie_heads.pt")
    p.add_argument("--variants", nargs="+", default=["fixed10", "varicl4to10"])
    p.add_argument("--out_dir", type=Path,
                   default=FV_FORMATION_DIR / "attention_head_analysis" / "perprompt_fv_norms")
    return p.parse_args()


def load_w_o_slices(heads):
    """mmap the GPT-J checkpoint for the 40 out_proj column slices (16 GB cgroup cap:
    never full-load the model locally)."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    bins = glob.glob(str(hf_home / "hub" / "models--EleutherAI--gpt-j-6b" /
                         "snapshots" / "*" / "pytorch_model.bin"))
    assert bins, "GPT-J pytorch_model.bin not found in HF cache"
    sd = torch.load(bins[0], map_location="cpu", weights_only=True, mmap=True)
    w_o = {(l, h): sd[f"transformer.h.{l}.attn.out_proj.weight"][:, h * HD:(h + 1) * HD].double()
           for l, h in heads}
    del sd
    return w_o


def main():
    args = parse_args()
    heads = [(int(l), int(h)) for l, h, _ in
             torch.load(args.pooled_heads_path, weights_only=False)["top_heads"]]
    w_o = load_w_o_slices(heads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_norms = {}      # variant -> {task: (170,) norms}
    all_shots = {}      # variant -> {task: (170,) n_shots}
    task_order = None
    for variant in args.variants:
        vdir = args.capture_root / variant
        tasks = sorted(p.stem for p in vdir.glob("*.pt"))
        assert len(tasks) == 27, f"{variant}: expected 27 task files, found {len(tasks)}"
        if task_order is None:
            task_order = tasks
        norms_v, shots_v = {}, {}
        for task in tasks:
            d = torch.load(vdir / f"{task}.pt", weights_only=False)
            acts = d["activations"].double()          # (170, 28, 16, 256)
            n = acts.shape[0]
            fvs = torch.zeros(n, 4096, dtype=torch.float64)
            for (l, h) in heads:
                fvs += acts[:, l, h] @ w_o[(l, h)].T
            norms_v[task] = fvs.norm(dim=1).numpy()
            shots_v[task] = np.array([m["n_shots"] for m in d["metadata"]])

            # Advisory linearity anchor: ||FV of the capture's task mean|| vs mean of per-prompt FVs
            mean_fv = torch.zeros(4096, dtype=torch.float64)
            tm = d["task_mean_fp32"].double()
            for (l, h) in heads:
                mean_fv += w_o[(l, h)] @ tm[l, h]
            rel = (fvs.mean(0) - mean_fv).norm() / mean_fv.norm()
            if rel > 5e-3:   # fp16 storage of per-prompt acts vs fp32 task mean
                print(f"  WARNING {variant}/{task}: linearity anchor rel dev {rel:.2e}")
        all_norms[variant], all_shots[variant] = norms_v, shots_v
        print(f"[{variant}] loaded {len(tasks)} tasks x {n} prompts")

    # shared bins across variants for comparability
    pooled_all = np.concatenate([v for var in args.variants for v in all_norms[var].values()])
    bins = np.linspace(0, pooled_all.max() * 1.02, 61)

    # ---- fig 1: pooled histogram per variant ----
    for variant in args.variants:
        pooled = np.concatenate(list(all_norms[variant].values()))
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        ax.hist(pooled, bins=bins, color="tab:blue", alpha=0.8)
        ax.axvline(np.median(pooled), color="k", ls="--", lw=1.2,
                   label=f"median {np.median(pooled):.1f}")
        ax.set_xlabel(r"$\|v^j_A\|_2$  (per-prompt function vector norm)")
        ax.set_ylabel("prompts")
        ax.set_title(f"Per-prompt FV norms, all 27 tasks pooled ({len(pooled)} prompts) — {variant}\n"
                     r"$v^j_A = \sum_{h\in H} W_O\,h(p^j_A)$ at the final cue token, pooled top-40 heads",
                     fontsize=10.5, loc="left")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = args.out_dir / f"fvnorm_hist_pooled_{variant}.png"
        fig.savefig(out); plt.close(fig)
        print(f"wrote {out}")

    # ---- fig 2: per-task grid + summary panel, per variant ----
    for variant in args.variants:
        norms_v = all_norms[variant]
        med = {t: float(np.median(norms_v[t])) for t in task_order}
        order = sorted(task_order, key=lambda t: med[t])
        fig, axes = plt.subplots(7, 4, figsize=(16, 18), dpi=150)
        axes = axes.ravel()
        for ax, task in zip(axes, order):
            ax.hist(norms_v[task], bins=bins, color="tab:blue", alpha=0.85)
            ax.axvline(med[task], color="crimson", ls="--", lw=1.2)
            ax.set_title(f"{task}  (med {med[task]:.1f})", fontsize=8.5)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)
        # summary panel in the last slot: median +/- IQR, sorted
        ax = axes[27]
        q1 = [np.percentile(norms_v[t], 25) for t in order]
        q3 = [np.percentile(norms_v[t], 75) for t in order]
        y = np.arange(len(order))
        ax.errorbar([med[t] for t in order], y,
                    xerr=[np.array([med[t] for t in order]) - q1,
                          q3 - np.array([med[t] for t in order])],
                    fmt="o", ms=3, color="tab:blue", ecolor="grey", elinewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=6)
        ax.set_xlabel(r"median $\|v^j_A\|$ $\pm$ IQR", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        fig.suptitle(f"Per-prompt FV norm by task ({variant}) — panels sorted by median; "
                     "shared bins; last panel = summary", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = args.out_dir / f"fvnorm_hist_pertask_{variant}.png"
        fig.savefig(out); plt.close(fig)
        print(f"wrote {out}")

    # ---- fig 3: norm vs shot count (varicl4to10), fixed10 as 10-shot reference ----
    if "varicl4to10" in args.variants:
        norms_v, shots_v = all_norms["varicl4to10"], all_shots["varicl4to10"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
        cmap = plt.get_cmap("tab20")
        colors = {t: (cmap(i % 20) if i < 20 else plt.get_cmap("Dark2")(i - 20))
                  for i, t in enumerate(task_order)}
        for t in task_order:
            ss = sorted(set(shots_v[t]))
            ax1.plot(ss, [np.median(norms_v[t][shots_v[t] == s]) for s in ss],
                     "-o", ms=2.5, lw=0.9, color=colors[t], alpha=0.75, label=t)
            if "fixed10" in args.variants:
                ax1.plot([10.35], [np.median(all_norms["fixed10"][t])], "s", ms=3.5,
                         color=colors[t], alpha=0.9)
        ax1.set_xlabel("n_shots (squares at 10.35 = fixed10 run reference)")
        ax1.set_ylabel(r"median $\|v^j_A\|$")
        ax1.set_title("Per-task median FV norm vs shot count (varicl4to10)", fontsize=10)
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=5, ncol=2, loc="upper left")
        # pooled: box per shot count
        pooled_shots = np.concatenate(list(shots_v.values()))
        pooled_norms = np.concatenate(list(norms_v.values()))
        ss = sorted(set(pooled_shots))
        ax2.boxplot([pooled_norms[pooled_shots == s] for s in ss], positions=ss,
                    widths=0.6, showfliers=False)
        ax2.set_xlabel("n_shots")
        ax2.set_ylabel(r"$\|v^j_A\|$")
        ax2.set_title("Pooled over tasks: FV norm by shot count", fontsize=10)
        ax2.grid(alpha=0.25)
        fig.suptitle("Shot-count dependence of per-prompt FV norms", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = args.out_dir / "fvnorm_vs_shots.png"
        fig.savefig(out); plt.close(fig)
        print(f"wrote {out}")

    # ---- data dumps ----
    for variant in args.variants:
        np.savez(args.out_dir / f"fvnorm_perprompt_{variant}.npz",
                 tasks=np.array(task_order),
                 norms=np.stack([all_norms[variant][t] for t in task_order]),
                 n_shots=np.stack([all_shots[variant][t] for t in task_order]))
    with open(args.out_dir / "fvnorm_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "task", "n", "mean", "median", "std", "q25", "q75", "min", "max"])
        for variant in args.variants:
            for t in task_order:
                x = all_norms[variant][t]
                w.writerow([variant, t, len(x), f"{x.mean():.3f}", f"{np.median(x):.3f}",
                            f"{x.std():.3f}", f"{np.percentile(x, 25):.3f}",
                            f"{np.percentile(x, 75):.3f}", f"{x.min():.3f}", f"{x.max():.3f}"])
    print(f"wrote {args.out_dir / 'fvnorm_summary.csv'}")

    # console summary: tasks sorted by median (varicl variant last so it's freshest on screen)
    for variant in args.variants:
        norms_v = all_norms[variant]
        print(f"\n[{variant}] tasks by median ||v^j_A||:")
        for t in sorted(task_order, key=lambda t: np.median(norms_v[t])):
            x = norms_v[t]
            print(f"  {t:28s} med {np.median(x):7.2f}  IQR [{np.percentile(x,25):7.2f}, "
                  f"{np.percentile(x,75):7.2f}]  min {x.min():6.2f} max {x.max():7.2f}")


if __name__ == "__main__":
    main()
