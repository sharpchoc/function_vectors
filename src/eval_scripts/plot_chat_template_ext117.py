#!/usr/bin/env python
"""Aggregate + plot the chat-template transfer 6-shot accuracies (Qwen2.5-7B-Instruct,
117-task extended pool, three formats — see eval_chat_template_ext117.py).

Reads artifacts/chat_template_transfer/ext117_6shot/<format>/<task>.json plus the GPT-J
reference (n=6 rows of the extended n-shot sweep CSV) and writes to
results/chat_template_transfer/ext117_6shot_accuracy/:
  accuracy_6shot.csv   task, origin, lane, acc per arm, acc_gptj
  summary.csv          mean/median per arm (+ GPT-J reference)
  bar_6shot_<arm>.png  ranked ascending bar chart per arm (style of the GPT-J
                       nshot_bar_6shot.png: origin coloring, 30% reference line)
  arm_comparison_6shot.png  sorted per-task accuracy curves, all arms + GPT-J
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.paths import ARTIFACTS_ROOT, CHAT_TEMPLATE_TRANSFER_DIR, GENERAL_DIR  # noqa: E402

IN_ROOT = ARTIFACTS_ROOT / "chat_template_transfer" / "ext117_6shot"
OUT_DIR = CHAT_TEMPLATE_TRANSFER_DIR / "ext117_6shot_accuracy"
GPTJ_CSV = GENERAL_DIR / "extended_tasks_nshot_sweep" / "nshot_accuracy.csv"
ARMS = ["chat_blank_system", "chat_no_system", "plain"]
ARM_TITLES = {"chat_blank_system": "chat template, blank system prompt",
              "chat_no_system": "chat template, no system block",
              "plain": 'plain "Q:/A:" format'}

# validated reference palette (matches plot_extended_nshot_bar.py)
C_NEW, C_ORIG = "#2a78d6", "#eb6834"
C_PRUNED = "#c3c8ce"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
PRUNE_AT = 0.30
ARM_COLORS = {"chat_blank_system": "#2a78d6", "chat_no_system": "#7c4dbe",
              "plain": "#eb6834", "gptj": "#52514e"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    pool = json.load(open(REPO_ROOT / "dataset_files" / "extended_tasks" / "manifest.json"))["tasks"]
    meta, gptj = {}, {}
    for r in csv.DictReader(open(GPTJ_CSV)):
        if int(r["n_shots"]) == args.n and r["task"] in pool:
            meta[r["task"]] = {"origin": r["origin"], "lane": r["lane"]}
            gptj[r["task"]] = float(r["accuracy"])

    acc = {}
    for arm in ARMS:
        acc[arm] = {}
        for t in pool:
            f = IN_ROOT / arm / f"{t}.json"
            if not f.exists():
                continue
            recs = json.load(open(f))
            acc[arm][t] = sum(r["match"] for r in recs) / len(recs)
        missing = [t for t in pool if t not in acc[arm]]
        if missing:
            print(f"WARNING: {arm}: {len(missing)} tasks missing (e.g. {missing[:4]})")

    tasks = [t for t in sorted(pool) if all(t in acc[a] for a in ARMS)]
    print(f"{len(tasks)}/{len(pool)} tasks complete in all arms")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"accuracy_{args.n}shot.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "origin", "lane"] + [f"acc_{a}" for a in ARMS] + ["acc_gptj"])
        for t in tasks:
            w.writerow([t, meta[t]["origin"], meta[t]["lane"]]
                       + [f"{acc[a][t]:.4f}" for a in ARMS]
                       + [f"{gptj[t]:.4f}" if t in gptj else ""])

    stats = {}
    for name, vals in [(a, [acc[a][t] for t in tasks]) for a in ARMS] + \
                      [("gptj", [gptj[t] for t in tasks if t in gptj])]:
        vs = sorted(vals)
        stats[name] = {"mean": sum(vs) / len(vs), "median": vs[len(vs) // 2], "n_tasks": len(vs)}
    with open(OUT_DIR / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "model", "mean", "median", "n_tasks"])
        for name, s in stats.items():
            model = "EleutherAI/gpt-j-6b" if name == "gptj" else "Qwen/Qwen2.5-7B-Instruct"
            w.writerow([name, model, f"{s['mean']:.4f}", f"{s['median']:.4f}", s["n_tasks"]])
    for name, s in stats.items():
        print(f"{name:>18}: mean {s['mean']:.3f}  median {s['median']:.3f}  (n={s['n_tasks']})")

    # ranked ascending bar chart per arm (style of plot_extended_nshot_bar.py)
    for arm in ARMS:
        rows = sorted(tasks, key=lambda t: acc[arm][t])
        accs = [acc[arm][t] for t in rows]
        # same pruning rule as the GPT-J reference plot: tasks under 30% in THIS arm are greyed
        colors = [C_PRUNED if acc[arm][t] < PRUNE_AT
                  else (C_NEW if meta[t]["origin"] == "new" else C_ORIG) for t in rows]
        n_pruned = sum(1 for a in accs if a < PRUNE_AT)
        fig, ax = plt.subplots(figsize=(30, 6.5))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        ax.bar(range(len(rows)), accs, color=colors, width=0.82)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(rows, rotation=90, fontsize=5.2, color=INK2)
        ax.set_xlim(-0.8, len(rows) - 0.2)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(f"{args.n}-shot accuracy", fontsize=11, color=INK)
        ax.set_title(f"Qwen2.5-7B-Instruct {args.n}-shot accuracy by task — {ARM_TITLES[arm]} "
                     f"({len(rows)} tasks), ascending. T=1.0 sampled generation, full-label match, "
                     f"50 prompts/task.", fontsize=12, color=INK)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="y", labelsize=9, colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.axhline(PRUNE_AT, color=INK2, lw=1.1, ls="--", alpha=0.7)
        ax.text(1, PRUNE_AT + 0.012, "pruning threshold (30%)", fontsize=9, color=INK2)
        n_new = sum(1 for t in rows if meta[t]["origin"] == "new" and acc[arm][t] >= PRUNE_AT)
        n_orig = sum(1 for t in rows if meta[t]["origin"] != "new" and acc[arm][t] >= PRUNE_AT)
        ax.legend(handles=[Patch(color=C_NEW, label=f"new task ({n_new})"),
                           Patch(color=C_ORIG, label=f"original abstractive ({n_orig})"),
                           Patch(color=C_PRUNED, label=f"pruned tasks, <30% ({n_pruned})")],
                  loc="upper left", fontsize=10, frameon=False)
        print(f"{arm}: {n_pruned} pruned (<30%), {len(rows) - n_pruned} kept "
              f"({n_new} new + {n_orig} original)")
        fig.tight_layout()
        out = OUT_DIR / f"bar_{args.n}shot_{arm}.png"
        fig.savefig(out, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {out}")

    # arm comparison: per-task accuracies sorted within each arm (distribution profiles)
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    series = [(a, [acc[a][t] for t in tasks]) for a in ARMS]
    series.append(("gptj", [gptj[t] for t in tasks if t in gptj]))
    for name, vals in series:
        vs = sorted(vals)
        label = ("GPT-J, plain Q:/A: (reference)" if name == "gptj"
                 else f"Qwen2.5, {ARM_TITLES[name]}")
        ax.plot(range(len(vs)), vs, lw=2.2, color=ARM_COLORS[name],
                label=f"{label} — mean {stats[name]['mean']:.3f}")
    ax.set_xlabel("task rank (each arm sorted ascending independently)", fontsize=10, color=INK)
    ax.set_ylabel(f"{args.n}-shot accuracy", fontsize=10, color=INK)
    ax.set_title(f"{args.n}-shot ICL accuracy profiles over {len(tasks)} extended tasks",
                 fontsize=12, color=INK)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / f"arm_comparison_{args.n}shot.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
