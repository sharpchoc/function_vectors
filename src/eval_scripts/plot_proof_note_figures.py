#!/usr/bin/env python
"""Write-up figures for the 'FV heads carry the map' proof note (2026-10-06).

Reads the per-model summary CSVs (no numbers hardcoded) and writes PNGs that paste into a
write-up:
  head_mean_ablation_both_models.png   one grouped bar chart, GPT-J vs Qwen2.5-7B-Instruct:
                                       0-shot | 6-shot | FV heads mean-ablated | random heads
  table_head_mean_ablation.png         the step-1 table as an image
  table_ov_circuit_map.png             the step-3 table (OV-circuit map cosines) as an image
Output dir: results/qwen25_fv/cross_model/proof_note/ (cross-model material lives in the
Qwen bucket, whose README already carries the Qwen-vs-GPT-J tables).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import TASK69_RUN_DIR, QWEN25_FV_DIR  # noqa: E402

MODELS = [("GPT-J-6B", TASK69_RUN_DIR, "#2a78d6"), ("Qwen2.5-7B-Instruct", QWEN25_FV_DIR, "#eb6834")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def read_summary(path):
    rows = list(csv.DictReader(open(path)))
    return {(r["condition"], r["task_group"]): r for r in rows}


def read_ov(path):
    return {r["metric"]: r for r in csv.DictReader(open(path))}


def per_task_sem(path, col):
    v = np.array([float(r[col]) for r in csv.DictReader(open(path))])
    return float(v.std(ddof=1) / np.sqrt(len(v)))


def bar_chart(out):
    conds = [("zero_shot", "0-shot\n(no demos)"), ("real_6shot", "6-shot\nunablated"),
             ("fvheads_mean", "6-shot, FV heads\nmean-ablated at cue"),
             ("random_mean", "6-shot, random\nsame-size non-FV heads")]
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=200)
    x = np.arange(len(conds)); w = 0.36
    for k, (name, root, color) in enumerate(MODELS):
        s = read_summary(root / "FV_ablation" / "head_mean_ablation" / "summary.csv")
        pt = root / "FV_ablation" / "head_mean_ablation" / "per_task_acc.csv"
        n_tasks = int(s[("real_6shot", "all")]["n_tasks"])
        n_heads = {"GPT-J-6B": 37, "Qwen2.5-7B-Instruct": 140}[name]
        means = [float(s[(c, "all")]["mean_acc"]) for c, _ in conds]
        sems = [per_task_sem(pt, c) for c, _ in conds]
        xs = x + (k - 0.5) * (w + 0.04)
        ax.bar(xs, means, w, color=color, yerr=sems, capsize=2.5, error_kw={"lw": 1, "ecolor": MUTED},
               label=f"{name} ({n_heads} FV heads, {n_tasks} tasks)", zorder=3)
        for xi, m in zip(xs, means):
            ax.text(xi, m + 0.022, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, [lab for _, lab in conds], fontsize=9)
    ax.set_ylabel("mean T=1 exact-match accuracy")
    ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title("Mean-ablating the FV heads at the final cue token (error bars: SEM over tasks)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def table_png(out, header, rows, col_widths, shade=(), row_h=0.42, header_h=None):
    """Hand-laid-out table: col_widths in inches, header may contain newlines; shade = {(r, c)}
    cells (1-based data rows) to tint + bold. Text is placed with fig.text, so nothing clips."""
    from matplotlib.patches import Rectangle
    n_hl = max(h.count("\n") + 1 for h in header)
    header_h = header_h or (0.22 + 0.19 * n_hl)
    W = sum(col_widths); H = header_h + row_h * len(rows)
    fig = plt.figure(figsize=(W, H), dpi=220)
    tr = fig.dpi_scale_trans
    x0 = np.concatenate([[0], np.cumsum(col_widths)])
    pad = 0.09
    fig.patches.append(Rectangle((0, H - header_h), W, header_h, transform=tr, facecolor="#f0efec", edgecolor="none"))
    def line(y):
        fig.add_artist(plt.Line2D([0, W], [y, y], transform=tr, color=GRID, lw=0.9))
    line(H); line(H - header_h); line(0)
    for c, h in enumerate(header):
        x = x0[c] + pad if c == 0 else x0[c + 1] - pad
        fig.text(x, H - header_h / 2, h, transform=tr, ha="left" if c == 0 else "right", va="center",
                 fontsize=9.3, weight="bold", color=INK, linespacing=1.15)
    for r, row in enumerate(rows):
        yc = H - header_h - row_h * (r + 0.5)
        for c, val in enumerate(row):
            if (r + 1, c) in shade:
                fig.patches.append(Rectangle((x0[c], yc - row_h / 2), col_widths[c], row_h, transform=tr,
                                             facecolor="#e6eef9", edgecolor="none"))
            x = x0[c] + pad if c == 0 else x0[c + 1] - pad
            fig.text(x, yc, val, transform=tr, ha="left" if c == 0 else "right", va="center", fontsize=9.6,
                     weight="bold" if (r + 1, c) in shade else "normal", color=INK)
        line(yc - row_h / 2)
    fig.savefig(out, bbox_inches="tight", facecolor="white", pad_inches=0.06); plt.close(fig)


def table_head_ablation(out):
    header = ["6-shot accuracy, mean over tasks", "0-shot", "unablated", "FV heads\nmean-ablated", "random same-\nsize heads"]
    rows, shade = [], set()
    for i, (name, root, _) in enumerate(MODELS):
        s = read_summary(root / "FV_ablation" / "head_mean_ablation" / "summary.csv")
        n_tasks = int(s[("real_6shot", "all")]["n_tasks"]); n_heads = {"GPT-J-6B": 37, "Qwen2.5-7B-Instruct": 140}[name]
        g = lambda c: f"{float(s[(c, 'all')]['mean_acc']):.3f}"
        rows.append([f"{name}, {n_heads} heads, {n_tasks} tasks", g("zero_shot"), g("real_6shot"), g("fvheads_mean"), g("random_mean")])
        shade.add((i + 1, 3))
    table_png(out, header, rows, [3.6, 0.85, 1.05, 1.35, 1.45], shade=shade)


def table_ov(out):
    ov = {name: read_ov(root / "read_write_relationship" / "ov_circuit_map" / "summary.csv") for name, root, _ in MODELS}
    f = lambda m, k: float(ov[m][k]["mean_all"])
    header = ["mean cosine with the task's own FV", "GPT-J-6B", "Qwen2.5-7B-Instruct"]
    rows = []
    rows.append(["$u_A$ through the FV heads"] + [f"{f(m,'u_A_noskip'):.3f}" for m, _, _ in MODELS])
    rows.append(["$s_A = c + u_A$ through the FV heads"] + [f"{f(m,'s_A_noskip'):.3f}" for m, _, _ in MODELS])
    rows.append(["same, with skip connection (+x): $u_A$ / $s_A$"] + [f"{f(m,'u_A_skip'):.3f} / {f(m,'s_A_skip'):.3f}" for m, _, _ in MODELS])
    rows.append(["input alone, no heads: $u_A$ / $s_A$"] + [f"{f(m,'u_A_input_only'):.3f} / {f(m,'s_A_input_only'):.3f}" for m, _, _ in MODELS])
    table_png(out, header, rows, [3.6, 1.3, 1.8], shade={(2, 1), (2, 2)})   # strongest own-FV cosines (s_A row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path, default=QWEN25_FV_DIR / "cross_model" / "proof_note")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    bar_chart(a.out_dir / "head_mean_ablation_both_models.png")
    table_head_ablation(a.out_dir / "table_head_mean_ablation.png")
    table_ov(a.out_dir / "table_ov_circuit_map.png")
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
