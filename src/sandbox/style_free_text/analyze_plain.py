#!/usr/bin/env python
"""style_free_text — accuracy vs k (plain English, no translation scaffold).

accuracy(family, style, k) = P(style_ok AND coherent) ; same breakdown/plots as
style_translation.analyze, plus a comparison figure of STYLE-ONLY adoption vs k against the
translation-scaffold run (read from results/style_translation/summary.csv, read-only).
Outputs -> results/style_free_text/: accuracy_by_k.png, unscorable_by_k.png, unscorable.csv,
summary.csv, records.npz, style_only_vs_translation.png
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
from src.utils.paths import ARTIFACTS_ROOT, STYLE_FREE_TEXT_RESULTS, STYLE_TRANSLATION_RESULTS
from src.sandbox.style_translation.families import FAMILIES, FAMILY
from src.sandbox.style_translation.analyze import wilson

ROLL = ARTIFACTS_ROOT / "style_free_text" / "rollouts"
OUT = STYLE_FREE_TEXT_RESULTS
KS = [0, 1, 2, 3, 4]
C = {"nat": "#1f77b4", "alt": "#d62728"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, npz, data = [], {}, {}
    fams = [f.name for f in FAMILIES if (ROLL / f"{f.name}.json").exists()]
    import re
    PUNCT_ONLY = re.compile(r'^\s*["\u201c\u201d]?[.,!?]+["\u201c\u201d]?\s*$')
    for fam in fams:
        recs = json.load(open(ROLL / f"{fam}.json"))
        judged = [r for r in recs if r.get("judge")]
        # A completion that is ONLY the closing punctuation of the current sentence (e.g. the
        # quote_punct decision `."` / `".`) is a coherent continuation — the sentence cutter stops
        # right there, and the judge, seeing no reference, calls bare punctuation "not a
        # continuation". Count these as coherent when a style decision was made. (In the
        # translation run the reference was the same string, so the judge accepted them.)
        for r in judged:
            if not r["judge"]["ok"] and r["decision"] is not None and PUNCT_ONLY.match(r["tail"] or ""):
                r["judge"] = dict(r["judge"], ok=True, notes="punctuation-only completion closing the sentence (rule)")
        data[fam] = {}
        for style in ("nat", "alt"):
            for k in KS:
                sel = [r for r in judged if r["style"] == style and r["k"] == k]
                n = len(sel)
                if n == 0:
                    continue
                correct = sum(r["style_ok"] and r["judge"]["ok"] for r in sel)
                acc = correct / n; lo, hi = wilson(acc, n)
                unsc = sum(r["decision"] is None for r in sel) / n
                data[fam][(style, k)] = dict(acc=acc, lo=lo, hi=hi, n=n, unsc=unsc,
                                            style_ok=sum(r["style_ok"] for r in sel) / n)
                rows.append(dict(family=fam, style=style, k=k, n=n, accuracy=round(acc, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                                 style_ok=round(sum(r["style_ok"] for r in sel) / n, 3),
                                 wrong_style=round(sum(r["decision"] not in (None, style) for r in sel) / n, 3),
                                 unscorable=round(unsc, 3),
                                 style_ok_but_incoherent=round(sum(r["style_ok"] and not r["judge"]["ok"] for r in sel) / n, 3),
                                 judge_ok=round(sum(r["judge"]["ok"] for r in sel) / n, 3),
                                 capped=round(sum(r["capped"] for r in sel) / n, 3),
                                 unjudged=sum(1 for r in recs if r["style"] == style and r["k"] == k) - n))
        npz[f"{fam}__style"] = np.array([r["style"] == "nat" for r in recs])
        npz[f"{fam}__k"] = np.array([r["k"] for r in recs])
        npz[f"{fam}__decision"] = np.array([{"nat": 1, "alt": 0, None: -1}[r["decision"]] for r in recs])
        npz[f"{fam}__judge_ok"] = np.array([r["judge"]["ok"] if r.get("judge") else -1 for r in recs])
        npz[f"{fam}__capped"] = np.array([r["capped"] for r in recs])
    with open(OUT / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(OUT / "records.npz", **npz)

    ncol = 4; nrow = int(math.ceil(len(fams) / ncol))

    def grid(fname, title, ylabel, draw, shared_legend=False):
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, fam in zip(axes, fams):
            draw(ax, fam)
            ax.set_title(fam, fontsize=10); ax.set_xticks(KS); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
            if not shared_legend:
                ax.legend(fontsize=6.5, frameon=False)
        for ax in axes[len(fams):]:
            ax.axis("off")
        if shared_legend:
            h, l = axes[0].get_legend_handles_labels()
            axes[len(fams)].legend(h, l, loc="center", fontsize=10, frameon=False)
        for ax in axes[-ncol:]:
            ax.set_xlabel("k = number of in-context examples of the convention")
        for ax in axes[::ncol]:
            ax.set_ylabel(ylabel, fontsize=9)
        fig.suptitle(title, fontsize=13, y=0.995); fig.tight_layout(rect=(0, 0, 1, 0.955))
        fig.savefig(OUT / fname, dpi=150); plt.close(fig)

    def draw_acc(ax, fam):
        for s in ("nat", "alt"):
            pts = [(k, data[fam][(s, k)]) for k in KS if (s, k) in data[fam]]
            ax.errorbar([k for k, _ in pts], [d["acc"] for _, d in pts],
                        yerr=[[d["acc"] - d["lo"] for _, d in pts], [d["hi"] - d["acc"] for _, d in pts]],
                        color=C[s], marker="o", ms=5, capsize=3, lw=1.8,
                        label=f"{s}: {FAMILY[fam].nat if s == 'nat' else FAMILY[fam].alt}"[:60])
        ax.legend(loc="lower right", fontsize=6.5, frameon=False)

    def draw_unsc(ax, fam):
        for s in ("nat", "alt"):
            pts = [(k, data[fam][(s, k)]) for k in KS if (s, k) in data[fam]]
            ax.plot([k for k, _ in pts], [d["unsc"] for _, d in pts], color=C[s], marker="o", ms=5, lw=1.8,
                    label=f"{s}: {FAMILY[fam].nat if s == 'nat' else FAMILY[fam].alt}"[:60])

    grid("accuracy_by_k.png",
         "Does GPT-J learn a writing convention from in-context examples? Plain English text, no translation scaffold\n"
         "Accuracy = uses the convention shown in context AND continues coherently (n = 200 per point, 95% CI)",
         "accuracy", draw_acc)
    grid("unscorable_by_k.png",
         "Share of completions that use neither convention (counted as inaccurate in the accuracy plot) — plain English variation",
         "unscorable share", draw_unsc)

    # comparison with the translation-scaffold run: style-only adoption (judge-independent)
    tr = STYLE_TRANSLATION_RESULTS / "summary.csv"
    if tr.exists():
        trans = {(r["family"], r["style"], int(r["k"])): float(r["style_ok"]) for r in csv.DictReader(open(tr))}

        def draw_cmp(ax, fam):
            for s in ("nat", "alt"):
                ks = [k for k in KS if (s, k) in data[fam]]
                ax.plot(ks, [data[fam][(s, k)]["style_ok"] for k in ks], color=C[s], marker="o", ms=5, lw=2.0,
                        label=f"{s}, plain English")
                tk = [k for k in ks if (fam, s, k) in trans]
                ax.plot(tk, [trans[(fam, s, k)] for k in tk], color=C[s], marker="s", ms=4, lw=1.2, ls="--", alpha=0.8,
                        label=f"{s}, with Spanish source")
        grid("style_only_vs_translation.png",
             "Convention adoption vs k: plain English (solid) vs the same prompts with the Spanish source and header (dashed)\n"
             "style-only rate = P(completion uses the convention shown in context), judge-independent; n = 200 per point",
             "convention adoption", draw_cmp, shared_legend=True)

    with open(OUT / "unscorable.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["family", "style", "unscorable_pooled"] + [f"unscorable_k{k}" for k in KS])
        for fam in fams:
            for s in ("nat", "alt"):
                per = [data[fam][(s, k)]["unsc"] for k in KS if (s, k) in data[fam]]
                w.writerow([fam, s, round(float(np.mean(per)), 3)] + [round(x, 3) for x in per])
    print(f"{'family':14s} {'style':5s} " + " ".join(f"acc k={k}".rjust(8) for k in KS) + "   style-only k0→k4  unscorable  judge_ok")
    for fam in fams:
        for s in ("nat", "alt"):
            d = [data[fam][(s, k)] for k in KS if (s, k) in data[fam]]
            jo = np.mean([r["judge_ok"] for r in rows if r["family"] == fam and r["style"] == s])
            print(f"{fam:14s} {s:5s} " + " ".join(f"{x['acc']:8.2f}" for x in d) + f"   {d[0]['style_ok']:.2f}→{d[-1]['style_ok']:.2f}        {np.mean([x['unsc'] for x in d]):.2f}      {jo:.2f}")
    print("->", OUT)


if __name__ == "__main__":
    main()
