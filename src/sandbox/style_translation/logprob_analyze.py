#!/usr/bin/env python
"""Step 3b analysis — first-token log-prob margin vs k per family and pole (logprob_margin.py records).
margin = log p(first token of the context's rendering) - log p(first token of the other rendering) at the cue; top1 = the argmax
next token IS the context rendering (classifier-free accuracy). Outputs -> <results>/logprob/: logprob_margin.csv,
logprob_margin_by_k.png (grid, mean ± 95 % CI, one line per pole), logprob_top1_by_k.png (grid), logprob_summary.png (pooled)."""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.code_families import CODE_FAMILY
from src.sandbox.style_translation.family_groups import grouped_grid, grouped_order

C = {"nat": "#1f77b4", "alt": "#d62728"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="*", default=None); ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    MP = model_paths(args.model); LP = MP["prompts"].parent / "logprob"; OUT = (MP["results"] / args.tag if args.tag else MP["results"]) / "logprob"; OUT.mkdir(parents=True, exist_ok=True)
    fams = [f for f in CODE_FAMILY if (LP / f"{f}.json").exists() and (args.families is None or f in args.families)]
    S = {}; rows = []
    for fam in fams:
        recs = json.load(open(LP / f"{fam}.json"))
        for style in ("nat", "alt"):
            for k in sorted({r["k"] for r in recs}):
                m = np.array([r["margin"] for r in recs if r["style"] == style and r["k"] == k]); t = np.array([r["top1_ctx"] for r in recs if r["style"] == style and r["k"] == k])
                if not len(m):
                    continue
                ci = 1.96 * m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else 0.0
                S[(fam, style, k)] = dict(n=len(m), mean=float(m.mean()), ci=float(ci), median=float(np.median(m)), pos=float((m > 0).mean()), top1=float(t.mean()))
                rows.append(dict(family=fam, style=style, k=k, **{a: round(b, 4) if isinstance(b, float) else b for a, b in S[(fam, style, k)].items()}))
    with open(OUT / "logprob_margin.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    KS = sorted({k for (_, _, k) in S})

    def grid(metric, fname, ylabel, title, ylim=None, zero=False, err=False):
        fig, gax = grouped_grid(fams, panel_w=3.6, panel_h=2.8, top=0.9, sharex=True, sharey=True)
        for fam in grouped_order(fams):
            ax = gax[(fam, 0)]
            for style in ("nat", "alt"):
                ks = [k for k in KS if (fam, style, k) in S]; y = [S[(fam, style, k)][metric] for k in ks]
                if err:
                    ax.errorbar(ks, y, yerr=[S[(fam, style, k)]["ci"] for k in ks], color=C[style], marker="o", ms=3, lw=1.5, capsize=2, label=f"{style} context")
                else:
                    ax.plot(ks, y, color=C[style], marker="o", ms=3, lw=1.5, label=f"{style} context")
            if zero:
                ax.axhline(0, color="#9e9e9e", lw=1, ls="dashed")
            ax.set_title(fam, fontsize=9); ax.set_xticks(KS); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
            if ylim: ax.set_ylim(*ylim)
        for ax in fig.axes_meta["bottom"]:
            ax.set_xlabel("k (opportunities shown in context)", fontsize=8)
        for ax in fig.axes_meta["left"]:
            ax.set_ylabel(ylabel, fontsize=8)
        h, l = next(iter(gax.values())).get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, fontsize=10, frameon=False)
        fig.suptitle(title, fontsize=12, y=0.99); fig.savefig(OUT / fname, dpi=150); plt.close(fig)

    grid("mean", "logprob_margin_by_k.png", "mean margin (nats)", f"{MP['label']}: first-token log-prob margin at the cue vs k — log p(context's rendering) − log p(other rendering), mean ± 95 % CI, 200 prompts per point", zero=True, err=True)
    grid("top1", "logprob_top1_by_k.png", "P(top-1 token = context rendering)", f"{MP['label']}: classifier-free accuracy vs k — share of prompts whose most likely next token is the context's rendering", ylim=(0, 1))
    # pooled summary
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, lab in zip(axes, ("mean", "top1"), ("mean first-token margin (nats)", "P(top-1 = context rendering)")):
        for style in ("nat", "alt"):
            M = np.array([[S[(f, style, k)][metric] for k in KS] for f in fams if all((f, style, k) in S for k in KS)])
            med = np.median(M, 0); lo, hi = np.percentile(M, 25, 0), np.percentile(M, 75, 0)
            ax.plot(KS, med, color=C[style], marker="o", lw=1.8, label=f"{style} context (median over families, IQR)"); ax.fill_between(KS, lo, hi, color=C[style], alpha=0.15)
        if metric == "mean": ax.axhline(0, color="#9e9e9e", ls="dashed", lw=1)
        else: ax.set_ylim(0, 1)
        ax.set_xticks(KS); ax.set_xlabel("k"); ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{MP['label']}: first-token log-prob margin across {len(fams)} code families", fontsize=11); fig.tight_layout(); fig.savefig(OUT / "logprob_summary.png", dpi=150); plt.close(fig)
    print(f"{'family':18s} " + " ".join(f"{s}{k}" for s in ('n', 'a') for k in KS))
    for fam in fams:
        print(f"{fam:18s} " + " ".join(f"{S[(fam, s, k)]['top1']:.2f}" if (fam, s, k) in S else "  - " for s in ('nat', 'alt') for k in KS))
    print("->", OUT)


if __name__ == "__main__":
    main()
