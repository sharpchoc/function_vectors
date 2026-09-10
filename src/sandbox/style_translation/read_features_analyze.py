#!/usr/bin/env python
"""Step 6c — read features: summary of the evidence-token means per family and layer.

Per family and layer: |r_nat - r_alt| / |mean|, split-half cosine of the difference, cosine between
the read difference (r_nat - r_alt) and the step-4 PAIRED steering (write) vector v_nat at the same
layer, evidence tokens per prompt. Outputs -> results/style_translation/read_features/:
read_features.csv, evidence_tokens.csv, read_feature_summary.png (grouped LEXICAL / FIXED layout).
"""
import collections
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, STYLE_TRANSLATION_RESULTS
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.steer_screen import SCREEN_LAYERS
from src.sandbox.style_translation.family_groups import grouped_grid, grouped_order

RF = ARTIFACTS_ROOT / "style_translation" / "read_features"
VEC = ARTIFACTS_ROOT / "style_translation" / "steering" / "vectors"
OUT = STYLE_TRANSLATION_RESULTS / "read_features"


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fams = [f.name for f in FAMILIES if (RF / f"{f.name}.npz").exists()]
    rows, S = [], {}
    for fam in fams:
        d = np.load(RF / f"{fam}.npz"); v = np.load(VEC / f"{fam}.npz")["v_nat"]
        r = d["mean_nat"] - d["mean_alt"]
        for L in range(1, 29):
            rows.append(dict(family=fam, layer=L, rel_norm=round(float(d["norm_diff"][L - 1] / d["norm_mean"][L - 1]), 4),
                             split_half_cos=round(float(d["split_half_cos"][L - 1]), 3), cos_read_write=round(cos(r[L - 1], v[L - 1]), 3),
                             n_nat=int(d["n_nat"]), n_alt=int(d["n_alt"]), tokens_per_prompt_nat=round(float(d["tokens_per_prompt_nat"][0]), 2),
                             tokens_per_prompt_alt=round(float(d["tokens_per_prompt_alt"][0]), 2)))
        S[fam] = rows[-28:]
    with open(OUT / "read_features.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # evidence token audit
    with open(OUT / "evidence_tokens.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["family", "pole", "evidence_text", "count", "tokens"])
        for fam in fams:
            c = collections.Counter()
            for rec in json.load(open(RF / "evidence" / f"{fam}.json")):
                for e in rec["instances"]:
                    c[(rec["pole"], "".join(e["toks"]), len(e["toks"]))] += 1
            for pole in ("nat", "alt"):
                for (p_, txt, nt), n in sorted(((k, n) for k, n in c.items() if k[0] == pole), key=lambda kv: -kv[1])[:5]:
                    w.writerow([fam, pole, txt, n, nt])

    # figure
    fig, gax = grouped_grid(fams, panel_w=3.6, panel_h=2.8, top=0.86, sharex=True, sharey=True)
    for fam in grouped_order(fams):
        ax = gax[(fam, 0)]; Ls = list(range(1, 29))
        rel = [S[fam][L - 1]["rel_norm"] for L in Ls]; cr = [S[fam][L - 1]["cos_read_write"] for L in Ls]; sh = [S[fam][L - 1]["split_half_cos"] for L in Ls]
        ax.plot(Ls, cr, color="#1f6c80", lw=1.8, marker="o", ms=3, label="cos(read diff, write vector)")
        ax.plot(Ls, np.array(rel) / max(rel), color="#b8860b", lw=1.4, marker="s", ms=2.5, label="|read diff| / |mean|, scaled to its max")
        ax.plot(Ls, sh, color="#9e9e9e", lw=1.2, ls="dashed", label="split-half cos of read diff")
        for L in SCREEN_LAYERS:
            ax.axvline(L, color="#dddddd", lw=0.6, zorder=0)
        ax.set_title(f"{fam}  (max |diff|/|mean| = {max(rel):.2f})", fontsize=9); ax.set_ylim(-0.25, 1.05); ax.set_xticks(SCREEN_LAYERS); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    for ax in fig.axes_meta["bottom"]:
        ax.set_xlabel("layer", fontsize=8)
    h, l = next(iter(gax.values())).get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, fontsize=9.5, frameon=False)
    fig.suptitle("Read features: mean activation at the evidence tokens (k = 4 prompts, 200 texts per pole) — nat minus alt, by layer\n"
                 "cosine with the cue-token steering vector of the same family and layer; vertical lines = the 9 screened layers", fontsize=12, y=0.995)
    fig.savefig(OUT / "read_feature_summary.png", dpi=150); plt.close(fig)

    print(f"{'family':14s} {'tok/prompt':>10s} {'rel|diff| L6/12/20/24':>22s} {'split-half L6/12/20/24':>22s} {'cos(read,write) L6/12/20/24':>28s}")
    for fam in fams:
        g = lambda k, Ls=(6, 12, 20, 24): "/".join(f"{S[fam][L-1][k]:.2f}" for L in Ls)
        print(f"{fam:14s} {S[fam][0]['tokens_per_prompt_nat']:4.1f}/{S[fam][0]['tokens_per_prompt_alt']:<5.1f} {g('rel_norm'):>22s} {g('split_half_cos'):>22s} {g('cos_read_write'):>28s}")
    print("->", OUT)


if __name__ == "__main__":
    main()
