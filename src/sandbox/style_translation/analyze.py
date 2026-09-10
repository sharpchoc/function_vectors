#!/usr/bin/env python
"""Step 3 — accuracy vs k per family and style.

accuracy(family, style, k) = P(style_ok AND judge ok) over the 200 texts.
Breakdown: correct | wrong_style (decision == other style) | unscorable (decision None) |
style_ok_but_unfaithful (judge NOT OK). Also capped rate and Wilson 95% CI.
Outputs -> results/style_translation/: accuracy_by_k.png, unscorable_by_k.png, unscorable.csv,
summary.csv, records.npz.
"""
import csv
import json
import math
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
from src.utils.paths import ARTIFACTS_ROOT, STYLE_TRANSLATION_RESULTS
from src.sandbox.style_translation.families import FAMILIES, FAMILY
from src.sandbox.style_translation.family_groups import grouped_grid, grouped_order

ROLL = ARTIFACTS_ROOT / "style_translation" / "rollouts"
OUT = STYLE_TRANSLATION_RESULTS
KS = [0, 1, 2, 3, 4]


def wilson(p, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, npz = [], {}
    fams = [f.name for f in FAMILIES if (ROLL / f"{f.name}.json").exists()]
    data = {}
    for fam in fams:
        recs = json.load(open(ROLL / f"{fam}.json"))
        judged = [r for r in recs if r.get("judge")]
        data[fam] = {}
        for style in ("nat", "alt"):
            for k in KS:
                sel = [r for r in judged if r["style"] == style and r["k"] == k]
                n = len(sel)
                if n == 0:
                    continue
                correct = sum(r["style_ok"] and r["judge"]["ok"] for r in sel)
                wrong = sum(r["decision"] not in (None, style) for r in sel)
                unsc = sum(r["decision"] is None for r in sel)
                unf = sum(r["style_ok"] and not r["judge"]["ok"] for r in sel)
                capped = sum(r["capped"] for r in sel)
                acc = correct / n
                lo, hi = wilson(acc, n)
                data[fam][(style, k)] = (acc, lo, hi, n, unsc / n)
                rows.append(dict(family=fam, style=style, k=k, n=n, accuracy=round(acc, 3), ci_lo=round(lo, 3),
                                 ci_hi=round(hi, 3), style_ok=round(sum(r["style_ok"] for r in sel) / n, 3),
                                 wrong_style=round(wrong / n, 3), unscorable=round(unsc / n, 3),
                                 style_ok_but_unfaithful=round(unf / n, 3), judge_ok=round(sum(r["judge"]["ok"] for r in sel) / n, 3),
                                 capped=round(capped / n, 3), unjudged=sum(1 for r in recs if r["style"] == style and r["k"] == k) - n))
        npz[f"{fam}__style"] = np.array([r["style"] == "nat" for r in recs])
        npz[f"{fam}__k"] = np.array([r["k"] for r in recs])
        npz[f"{fam}__decision"] = np.array([{"nat": 1, "alt": 0, None: -1}[r["decision"]] for r in recs])
        npz[f"{fam}__judge_ok"] = np.array([(r["judge"] or {}).get("ok", -1) if r.get("judge") else -1 for r in recs])
        npz[f"{fam}__capped"] = np.array([r["capped"] for r in recs])

    with open(OUT / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(OUT / "records.npz", **npz)

    ncol = 4; nrow = int(math.ceil(len(fams) / ncol))
    C = {"nat": "#1f77b4", "alt": "#d62728"}
    for metric, fname, ylabel, title in (
        (0, "accuracy_by_k.png", "accuracy",
         "Does GPT-J learn a writing convention from in-context examples while translating Spanish to English?\n"
         "Accuracy = uses the convention shown in context AND translates faithfully (n = 200 per point, 95% CI)"),
        (4, "unscorable_by_k.png", "unscorable share",
         "Share of completions that use neither convention (counted as inaccurate in the accuracy plot)")):
        fig, gax = grouped_grid(fams, panel_w=3.6, panel_h=2.9, top=0.89, sharex=True, sharey=True)
        for fam in grouped_order(fams):
            ax = gax[(fam, 0)]
            for style in ("nat", "alt"):
                pts = [(k,) + data[fam][(style, k)] for k in KS if (style, k) in data[fam]]
                if not pts:
                    continue
                xs = [p[0] for p in pts]; ys = [p[1 + metric] if metric == 0 else p[5] for p in pts]
                lab = f"{style}: {FAMILY[fam].nat if style == 'nat' else FAMILY[fam].alt}"
                if metric == 0:
                    lo = [p[1] - p[2] for p in pts]; hi = [p[3] - p[1] for p in pts]
                    ax.errorbar(xs, ys, yerr=[lo, hi], color=C[style], marker="o", ms=5, capsize=3, lw=1.8, label=lab[:60])
                else:
                    ax.plot(xs, ys, color=C[style], marker="o", ms=5, lw=1.8, label=lab[:60])
            ax.set_title(fam, fontsize=10); ax.set_xticks(KS); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
            ax.legend(fontsize=6.5, loc="lower right" if metric == 0 else "upper right", frameon=False)
        for ax in fig.axes_meta["bottom"]:
            ax.set_xlabel("k = in-context examples of the convention", fontsize=8)
        for ax in fig.axes_meta["left"]:
            ax.set_ylabel(ylabel, fontsize=9)
        fig.suptitle(title, fontsize=13, y=0.99)
        fig.savefig(OUT / fname, dpi=150); plt.close(fig)

    # unscorable table: per family x style, pooled over k and per k
    with open(OUT / "unscorable.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["family", "style", "unscorable_pooled"] + [f"unscorable_k{k}" for k in KS])
        for fam in fams:
            for style in ("nat", "alt"):
                per = [data[fam][(style, k)][4] for k in KS if (style, k) in data[fam]]
                w.writerow([fam, style, round(float(np.mean(per)), 3)] + [round(x, 3) for x in per])
    print(f"{'family':14s} {'style':5s} " + " ".join(f"acc k={k}".rjust(8) for k in KS) + "   unscorable pooled  judge_ok pooled")
    for fam in fams:
        for style in ("nat", "alt"):
            accs = [data[fam][(style, k)][0] for k in KS if (style, k) in data[fam]]
            un = np.mean([data[fam][(style, k)][4] for k in KS if (style, k) in data[fam]])
            jo = np.mean([r["judge_ok"] for r in rows if r["family"] == fam and r["style"] == style])
            print(f"{fam:14s} {style:5s} " + " ".join(f"{a:8.2f}" for a in accs) + f"   {un:17.2f} {jo:15.2f}")
    print("->", OUT)


if __name__ == "__main__":
    main()
