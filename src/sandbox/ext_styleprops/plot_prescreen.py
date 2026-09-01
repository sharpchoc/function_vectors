#!/usr/bin/env python
"""Stage A4 pre-screen analysis: (k, distance) adherence surfaces + pass/fail table.

Reads artifacts/style_properties/prescreen/<prop>.json and produces
  results/style_properties/behavioral_prescreen/
    adherence_by_k.png        grid: P(continuation classified nat | ctx, k) per property
    separation_by_dist.png    grid: separation vs token distance since last manifestation
    prescreen_summary.csv     per-property stats + PROPOSED pass/fail
    prescreen_records.npz     raw per-record arrays to regenerate views

Proposed gate (thresholds are Stage-0 adjudication item 6):
  (a) separation s = P(nat|nat ctx) - P(nat|alt ctx) >= 0.3 at k>=4
  (b) s monotone increasing over k bins 0..4 (Spearman > 0)
  (c) adherence to the prior-DISFAVORED polarity at k>=4 >= 0.4
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, STYLE_PROPERTIES_DIR

IN_DIR = ARTIFACTS_ROOT / "style_properties" / "prescreen"
OUT_DIR = STYLE_PROPERTIES_DIR / "behavioral_prescreen"
K_BINS = [0, 1, 2, 3, 4]          # last bin = k>=4
DIST_BINS = [(1, 15), (16, 40), (41, 90), (91, 10_000)]


def kbin(k):
    return min(k, 4)


def rate_nat(recs):
    sc = [r for r in recs if r["label"] is not None]
    if not sc:
        return np.nan, 0
    return float(np.mean([r["label"] == "nat" for r in sc])), len(sc)


def rate_ctx(recs):
    """Adherence to the CONTEXT's own polarity (the ICL accuracy-vs-shots analog)."""
    sc = [r for r in recs if r["label"] is not None]
    if not sc:
        return np.nan, 0
    return float(np.mean([r["label"] == r["pol"] for r in sc])), len(sc)


def main():
    paths = sorted(IN_DIR.glob("*.json"))
    assert paths, f"no prescreen outputs in {IN_DIR}"
    props = [json.load(open(p)) for p in paths]
    n = len(props)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    fig2, ax2 = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    fig3, ax3 = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    rows = []
    npz = {}
    for pi, pdata in enumerate(props):
        name, recs = pdata["property"], pdata["records"]
        npz[f"{name}_k"] = np.array([r["k"] for r in recs])
        npz[f"{name}_dist"] = np.array([r["dist"] for r in recs])
        npz[f"{name}_pol"] = np.array([r["pol"] == "nat" for r in recs])
        npz[f"{name}_label"] = np.array([{"nat": 1, "alt": 0, None: -1}[r["label"]] for r in recs])

        curves = {}
        for pol in ("nat", "alt"):
            xs, ys, ns = [], [], []
            for kb in K_BINS:
                sub = [r for r in recs if r["pol"] == pol and kbin(r["k"]) == kb]
                v, m = rate_nat(sub)
                xs.append(kb); ys.append(v); ns.append(m)
            curves[pol] = (xs, ys, ns)
        s_curve = [a - b if not (np.isnan(a) or np.isnan(b)) else np.nan
                   for a, b in zip(curves["nat"][1], curves["alt"][1])]

        a = ax1[pi // ncol][pi % ncol]
        a.plot(K_BINS, curves["nat"][1], "o-", label="nat ctx")
        a.plot(K_BINS, curves["alt"][1], "s-", label="alt ctx")
        a.plot(K_BINS, s_curve, "k--", lw=1, label="separation (nat − alt)")
        a.set_title(name, fontsize=9)
        a.set_ylim(-0.05, 1.05)
        a.set_xlabel("k prior manifestations (4=4+)")
        if pi % ncol == 0:
            a.set_ylabel("P(continuation = nat pole)", fontsize=8)
        if pi == 0:
            a.legend(fontsize=7)

        # accuracy analog: adherence to the context's own polarity, by k.
        # cells with <20 scorable samples get open markers (thin-data flag).
        a = ax3[pi // ncol][pi % ncol]
        for pol, mk, col, lab in (("nat", "o", "#1f77b4", "nat-convention doc"),
                                  ("alt", "s", "#ff7f0e", "alt-convention doc")):
            ys, ms = [], []
            for kb in K_BINS:
                sub = [r for r in recs if r["pol"] == pol and kbin(r["k"]) == kb]
                v, m = rate_ctx(sub)
                ys.append(v)
                ms.append(m)
            a.plot(K_BINS, ys, "-", color=col, alpha=0.8, label=lab)
            solid = [i for i in range(len(ys)) if ms[i] >= 20]
            thin = [i for i in range(len(ys)) if 0 < ms[i] < 20]
            a.plot([K_BINS[i] for i in solid], [ys[i] for i in solid], mk, color=col)
            a.plot([K_BINS[i] for i in thin], [ys[i] for i in thin], mk, color=col,
                   mfc="none")
        a.set_title(name, fontsize=9)
        a.set_ylim(-0.05, 1.05)
        a.set_xlabel("k prior manifestations (4=4+)")
        if pi % ncol == 0:
            a.set_ylabel("P(continuation follows ctx polarity)", fontsize=8)
        if pi == 0:
            a.legend(fontsize=7)

        # distance effect at k>=2
        a = ax2[pi // ncol][pi % ncol]
        dx, dsep = [], []
        for lo, hi in DIST_BINS:
            sub_n = [r for r in recs if r["pol"] == "nat" and r["k"] >= 2 and lo <= r["dist"] <= hi]
            sub_a = [r for r in recs if r["pol"] == "alt" and r["k"] >= 2 and lo <= r["dist"] <= hi]
            vn, _ = rate_nat(sub_n)
            va, _ = rate_nat(sub_a)
            dx.append(f"{lo}-{hi if hi < 10_000 else '+'}")
            dsep.append(vn - va if not (np.isnan(vn) or np.isnan(va)) else np.nan)
        a.bar(dx, dsep)
        a.set_title(name, fontsize=9)
        a.set_ylim(-0.2, 1.0)
        a.set_xlabel("tok distance since last manifestation (k>=2)")
        if pi % ncol == 0:
            a.set_ylabel("separation s", fontsize=8)

        # gate
        scorable = np.mean([r["label"] is not None for r in recs])
        s_k4 = s_curve[-1]
        valid = [(k, s) for k, s in zip(K_BINS, s_curve) if not np.isnan(s)]
        rho = spearmanr([v[0] for v in valid], [v[1] for v in valid]).statistic if len(valid) >= 3 else np.nan
        nat_k0, alt_k0 = curves["nat"][1][0], curves["alt"][1][0]
        # prior-favored pole = the pole the model already picks at k=0
        prior_nat = np.nanmean([nat_k0, alt_k0]) > 0.5
        disfavored = "alt" if prior_nat else "nat"
        sub = [r for r in recs if r["pol"] == disfavored and r["k"] >= 4]
        v, m = rate_nat(sub)
        adh_disf = (1 - v) if disfavored == "alt" else v
        passed = (not np.isnan(s_k4) and s_k4 >= 0.3
                  and (np.isnan(rho) or rho > 0) and adh_disf >= 0.4)
        rows.append(dict(property=name, n=len(recs), scorable=round(float(scorable), 3),
                         s_k0=round(s_curve[0], 3) if not np.isnan(s_curve[0]) else "",
                         s_k4=round(s_k4, 3) if not np.isnan(s_k4) else "",
                         spearman_k=round(float(rho), 3) if not np.isnan(rho) else "",
                         disfavored_pole=disfavored,
                         adh_disfavored_k4=round(adh_disf, 3) if not np.isnan(adh_disf) else "",
                         PASS=passed))

    for k in range(len(props), nrow * ncol):
        ax1[k // ncol][k % ncol].axis("off")
        ax2[k // ncol][k % ncol].axis("off")
        ax3[k // ncol][k % ncol].axis("off")
    fig3.suptitle("Style-following accuracy vs. number of prior examples "
                  "(GPT-J, T=1 sampling)", fontsize=14, fontweight="bold", y=0.985)
    fig3.text(0.5, 0.955,
              "Each point: the fraction of sampled continuations at decision points that "
              "match the document's own convention,\nafter k earlier occurrences of that "
              "convention in the document. Open markers: fewer than 20 scorable samples.",
              ha="center", fontsize=10)
    fig1.suptitle("Sampled adherence at decision points: fraction of T=1 continuations "
                  "classified as the nat pole, under nat-polarity vs alt-polarity context "
                  "(dashed = their difference, the context separation)", fontsize=11)
    fig2.suptitle("Context separation s = P(nat | nat ctx) − P(nat | alt ctx) at decision "
                  "points, by token distance since the property last manifested", fontsize=11)
    for f_, name_, top in ((fig1, "adherence_by_k.png", 0.96),
                           (fig2, "separation_by_dist.png", 0.96),
                           (fig3, "accuracy_by_k.png", 0.93)):
        f_.tight_layout(rect=(0, 0, 1, top))
        f_.savefig(OUT_DIR / name_, dpi=150)
    np.savez(OUT_DIR / "prescreen_records.npz", **npz)

    import csv
    with open(OUT_DIR / "prescreen_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pool = {"description": "style-property pool passing the A4 behavioral pre-screen "
                           "(gate thresholds: adjudication memo item 4)",
            "pass": [r["property"] for r in rows if r["PASS"]],
            "fail": [r["property"] for r in rows if not r["PASS"]]}
    json.dump(pool, open(REPO_ROOT / "task_splits" / "style_properties_pool.json", "w"),
              indent=2)
    for r in rows:
        print("  ".join(f"{k}={v}" for k, v in r.items()))
    print(f"pool: {len(pool['pass'])} pass -> task_splits/style_properties_pool.json")
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
