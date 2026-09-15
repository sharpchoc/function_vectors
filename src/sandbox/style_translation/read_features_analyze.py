#!/usr/bin/env python
"""Step 6c — read features: summary of the evidence-token means per family and layer.

Per family and layer L (0 = embedding output, 1..n = block outputs): |r_nat - r_alt| / |mean|, split-half
cosine of the difference, cosine between the read difference (r_nat - r_alt) and the step-4 PAIRED cue-token
steering (write) vector v_nat at the SAME layer, and at L24 (the settled write-feature layer), evidence tokens
per prompt. Outputs -> results/style_translation/<model>[/<tag>]/read_features/: read_features.csv,
evidence_tokens.csv (top-5 evidence strings per pole), evidence_stats.csv (per-family evidence audit),
read_feature_summary.png (per-family curves), read_write_cos_by_layer.png (mean over families).
"""
import collections
import csv
import json
import statistics
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
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.ml_families import ML_FAMILIES, ML_FAMILY
from src.sandbox.style_translation.family_groups import CODE as CODE_NAMES, grouped_grid, grouped_order

RF = ARTIFACTS_ROOT / "style_translation" / "read_features"
VEC = ARTIFACTS_ROOT / "style_translation" / "steering" / "vectors"
OUT = STYLE_TRANSLATION_RESULTS / "read_features"
WRITE_LAYER = 24


def configure(model="gptj", tag=None):
    global RF, VEC, OUT
    MP = model_paths(model); RF, VEC = MP["read_features"], MP["steering"] / "vectors"
    OUT = (MP["results"] / tag if tag else MP["results"]) / "read_features"


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gptj", help="models.MODELS key")
    ap.add_argument("--tag", default=None, help="results sub-bucket (results/<model>/<tag>/read_features)")
    ap.add_argument("--families", nargs="*", default=None, help="restrict to these families (default: every family with a capture)")
    args = ap.parse_args()
    configure(args.model, args.tag)
    OUT.mkdir(parents=True, exist_ok=True)
    fams = [f.name for f in list(FAMILIES) + list(ML_FAMILIES) if (RF / f"{f.name}.npz").exists() and (args.families is None or f.name in args.families)]
    code_only = all(f in CODE_NAMES for f in fams)
    rows, S = [], {}
    for fam in fams:
        d = np.load(RF / f"{fam}.npz"); v = np.load(VEC / f"{fam}.npz")["v_nat"]; NL = d["mean_nat"].shape[0]
        r = d["mean_nat"] - d["mean_alt"]; vw = v[WRITE_LAYER - 1]
        S[fam] = {}
        common = dict(family=fam, n_nat=int(d["n_nat"]), n_alt=int(d["n_alt"]), tokens_per_prompt_nat=round(float(d["tokens_per_prompt_nat"][0]), 2),
                      tokens_per_prompt_alt=round(float(d["tokens_per_prompt_alt"][0]), 2))
        if "mean_nat_L0" in d:
            r0 = d["mean_nat_L0"] - d["mean_alt_L0"]
            S[fam][0] = dict(common, layer=0, rel_norm=round(float(np.linalg.norm(r0) / (np.linalg.norm(d["mean_nat_L0"]) + 1e-9)), 4),
                             split_half_cos=None, cos_read_write=None, cos_read_write_L24=round(cos(r0, vw), 3))
            rows.append(S[fam][0])
        for L in range(1, NL + 1):
            S[fam][L] = dict(common, layer=L, rel_norm=round(float(d["norm_diff"][L - 1] / d["norm_mean"][L - 1]), 4),
                             split_half_cos=round(float(d["split_half_cos"][L - 1]), 3), cos_read_write=round(cos(r[L - 1], v[L - 1]), 3),
                             cos_read_write_L24=round(cos(r[L - 1], vw), 3))
            rows.append(S[fam][L])
    with open(OUT / "read_features.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # evidence token audit
    with open(OUT / "evidence_tokens.csv", "w", newline="") as fh, open(OUT / "evidence_stats.csv", "w", newline="") as fh2:
        w = csv.writer(fh); w.writerow(["family", "pole", "evidence_text", "count", "tokens"])
        w2 = csv.writer(fh2); w2.writerow(["family", "prompts", "instances", "median_tokens_per_instance", "whitespace_only_share", "outside_prompt_share", "shared_tokens_dropped"])
        for fam in fams:
            c = collections.Counter(); insts = []; nrec = 0
            for rec in json.load(open(RF / "evidence" / f"{fam}.json")):
                nrec += 1
                for e in rec["instances"]:
                    insts.append(e); c[(rec["pole"], "".join(e["toks"]), len(e["toks"]))] += 1
            for pole in ("nat", "alt"):
                for (p_, txt, nt), n in sorted(((k, n) for k, n in c.items() if k[0] == pole), key=lambda kv: -kv[1])[:5]:
                    w.writerow([fam, pole, txt, n, nt])
            w2.writerow([fam, nrec, len(insts), statistics.median([len(e["idx"]) for e in insts]) if insts else 0,
                         round(float(np.mean([bool(e["idx"]) and "".join(e["toks"]).strip() == "" for e in insts])), 3) if insts else 0,
                         round(float(np.mean([e.get("outside_prompt", False) for e in insts])), 4) if insts else 0,
                         sum(e.get("n_shared_dropped", 0) for e in insts)])

    # per-family figure
    Ls = sorted(next(iter(S.values())).keys()); ticks = [L for L in (0, 4, 8, 12, 16, 20, 24, 28) if L <= max(Ls)]
    fig, gax = grouped_grid(fams, panel_w=3.6, panel_h=2.8, top=0.86, sharex=True, sharey=True)
    for fam in grouped_order(fams):
        ax = gax[(fam, 0)]
        rel = [S[fam][L]["rel_norm"] for L in Ls]
        ax.plot(Ls, [S[fam][L]["cos_read_write_L24"] for L in Ls], color="#1f6c80", lw=1.8, marker="o", ms=3, label="cos(read diff at L, write vector at L24)")
        ax.plot([L for L in Ls if L > 0], [S[fam][L]["cos_read_write"] for L in Ls if L > 0], color="#7fb3c4", lw=1.2, marker="^", ms=2.5, ls="dotted", label="cos(read diff at L, write vector at L)")
        ax.plot(Ls, np.array(rel) / max(rel), color="#b8860b", lw=1.4, marker="s", ms=2.5, label="|read diff| / |mean|, scaled to its max")
        ax.plot([L for L in Ls if L > 0], [S[fam][L]["split_half_cos"] for L in Ls if L > 0], color="#9e9e9e", lw=1.2, ls="dashed", label="split-half cos of read diff")
        ax.set_title(f"{fam}  (max |diff|/|mean| = {max(rel):.2f})", fontsize=9); ax.set_ylim(-0.25, 1.05); ax.set_xticks(ticks); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    for ax in fig.axes_meta["bottom"]:
        ax.set_xlabel("layer (0 = embeddings)", fontsize=8)
    h, l = next(iter(gax.values())).get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, fontsize=9.5, frameon=False)
    what = "in-context code" if code_only else "texts"
    fig.suptitle(f"Read features: mean activation at the evidence tokens (k = 4 prompts, paired {what} per pole) — nat minus alt, by layer\n"
                 "cosine with the cue-token steering (write) vector of the same family, at the same layer and at L24", fontsize=12, y=0.995)
    fig.savefig(OUT / "read_feature_summary.png", dpi=150); plt.close(fig)

    # mean over families
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for key, col, lab, Lsub in (("cos_read_write_L24", "#1f6c80", "cos(read diff at L, write vector at L24)", Ls),
                                ("cos_read_write", "#7fb3c4", "cos(read diff at L, write vector at L)", [L for L in Ls if L > 0]),
                                ("split_half_cos", "#9e9e9e", "split-half cos of the read diff", [L for L in Ls if L > 0])):
        M = np.array([[S[f][L][key] for L in Lsub] for f in fams], float)
        med = np.median(M, 0); lo, hi = np.percentile(M, 25, 0), np.percentile(M, 75, 0)
        ax.plot(Lsub, med, color=col, lw=1.8, marker="o", ms=3, label=lab); ax.fill_between(Lsub, lo, hi, color=col, alpha=0.15)
    ax.set_xticks(ticks); ax.set_ylim(-0.2, 1.02); ax.grid(alpha=0.3); ax.set_xlabel("read layer (0 = embeddings)"); ax.set_ylabel("cosine (median over families, IQR band)")
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    ax.set_title(f"Read–write geometry, {len(fams)} {'coding-convention' if code_only else ''} families: where does the evidence-token difference align with the cue-token write vector?", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "read_write_cos_by_layer.png", dpi=150); plt.close(fig)

    print(f"{'family':18s} {'tok/prompt':>10s} {'rel|diff| L6/12/20/24':>22s} {'split-half L6/12/20/24':>22s} {'cos(read,write L24) L0/6/12/20/24':>34s}")
    for fam in fams:
        g = lambda k, Ls=(6, 12, 20, 24): "/".join(f"{S[fam][L][k]:.2f}" for L in Ls)
        c24 = "/".join(f"{S[fam][L]['cos_read_write_L24']:.2f}" for L in (0, 6, 12, 20, 24) if L in S[fam])
        print(f"{fam:18s} {S[fam][1]['tokens_per_prompt_nat']:4.1f}/{S[fam][1]['tokens_per_prompt_alt']:<5.1f} {g('rel_norm'):>22s} {g('split_half_cos'):>22s} {c24:>34s}")
    print("->", OUT)


if __name__ == "__main__":
    main()
