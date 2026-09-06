#!/usr/bin/env python
"""Translation-framing variant: accuracy vs k, compared with the English-only prescreen.

Inputs (artifacts, gitignored):
  artifacts/style_properties/prescreen/<prop>.json            English-only (2026-09-01/02 run,
                                                              refs backfilled, trans_exact added)
  artifacts/style_properties/prescreen_translate/<prop>.json  Spanish:/English: framing, judged

Metrics (k = number of prior manifestations visible in the English prefix; x-axis exact k 0..5):
  style      P(continuation classified as the context's own convention | style-scorable)
             — identical to behavioral_prescreen/accuracy_by_k.png.
  judge      P(Gemini judge says the fragment is a correct continuation of the translation
             | judged)  — translate framing only (undefined for free continuation).
  exact      P(normalised fragment matches either twin's reference on >= 8 chars) — both framings.
  joint      P(style adopted AND judge-correct | style-scorable and judged) — the headline
             "accuracy" of the translation framing (user spec: style continuation AND correct
             translation).
Outputs -> results/style_properties/translation_framing/: accuracy_by_k.png, translation_by_k.png,
  summary.csv, records.npz. CPU only (numpy + matplotlib).
"""
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
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, STYLE_PROPERTIES_DIR

PLAIN_DIR = ARTIFACTS_ROOT / "style_properties" / "prescreen"
TRANS_DIR = ARTIFACTS_ROOT / "style_properties" / "prescreen_translate"
OUT_DIR = STYLE_PROPERTIES_DIR / "translation_framing"
POOL = json.load(open(REPO_ROOT / "task_splits" / "style_properties_pool.json"))["pass"]
PLOT_BINS = list(range(6))
MIN_N = 20
LABEL_CODE = {"nat": 1, "alt": 0, None: -1}


def load(path):
    recs = json.load(open(path))["records"]
    n = len(recs)
    a = {
        "k": np.array([r["k"] for r in recs]),
        "pol": np.array([r["pol"] == "nat" for r in recs]),            # True = nat context
        "label": np.array([LABEL_CODE[r["label"]] for r in recs]),
        "exact": np.array([bool(r.get("trans_exact", False)) for r in recs]),
        "judge": np.array([{True: 1, False: 0}.get(r.get("trans_judge"), -1) for r in recs]),
    }
    a["adopt"] = (a["label"] == a["pol"].astype(int)) & (a["label"] >= 0)
    a["scorable"] = a["label"] >= 0
    return a


def rate(mask_num, mask_den):
    n = int(mask_den.sum())
    return (float(mask_num[mask_den].mean()) if n else np.nan), n


def curves(a, sel):
    """Per exact k: style, judge, exact, joint, unscorable rates (+ denominators)."""
    out = {}
    for k in PLOT_BINS:
        m = sel & (a["k"] == k)
        judged = a["judge"] >= 0
        out[k] = {
            "style": rate(a["adopt"], m & a["scorable"]),
            "judge": rate(a["judge"] == 1, m & judged),
            "exact": rate(a["exact"], m),
            "joint": rate(a["adopt"] & (a["judge"] == 1), m & a["scorable"] & judged),
            "unscorable": rate(~a["scorable"], m),
        }
    return out


def plot_line(ax, cur, key, color, ls, marker, label, filled=True, lw=1.6):
    xs = [k for k in PLOT_BINS if not np.isnan(cur[k][key][0])]
    ys = [cur[k][key][0] for k in xs]
    ax.plot(xs, ys, ls=ls, color=color, lw=lw, label=label, zorder=2)
    for x, y in zip(xs, ys):
        ok = cur[x][key][1] >= MIN_N
        ax.plot(x, y, marker=marker, ms=6, ls="none", color=color,
                mfc=color if (filled and ok) else "white", mew=1.4, zorder=3)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    props = [p for p in POOL if (TRANS_DIR / f"{p}.json").exists()]
    missing = [p for p in POOL if p not in props]
    if missing:
        print("missing translate records:", missing)
    data = {p: {"plain": load(PLAIN_DIR / f"{p}.json"), "trans": load(TRANS_DIR / f"{p}.json")}
            for p in props}

    C_NAT, C_ALT, C_GREY = "#1f77b4", "#d62728", "#7f7f7f"
    ncol = 4
    nrow = int(np.ceil((len(props) + 1) / ncol))

    # ---- fig 1: accuracy by k -------------------------------------------------------------
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), sharex=True, sharey=True)
    axes = axes.ravel()
    summary, npz = [], {}
    for ax, p in zip(axes, props):
        d = data[p]
        for pol_is_nat, color, marker, ctxname in ((True, C_NAT, "o", "nat"), (False, C_ALT, "s", "alt")):
            cp = curves(d["plain"], d["plain"]["pol"] == pol_is_nat)
            ct = curves(d["trans"], d["trans"]["pol"] == pol_is_nat)
            plot_line(ax, cp, "style", C_GREY if pol_is_nat else "#bcbcbc", "--", marker,
                      f"English-only, {ctxname}-convention doc: style", filled=False, lw=1.2)
            plot_line(ax, ct, "style", color, "-", marker,
                      f"Spanish->English, {ctxname}-convention doc: style only", filled=False)
            plot_line(ax, ct, "joint", color, "-", marker,
                      f"Spanish->English, {ctxname}-convention doc: style AND correct translation",
                      filled=True, lw=2.2)
            for framing, cur in (("plain", cp), ("trans", ct)):
                row = {"property": p, "framing": framing, "ctx": ctxname}
                for k in PLOT_BINS:
                    for key in ("style", "judge", "exact", "joint", "unscorable"):
                        v, n = cur[k][key]
                        row[f"{key}_k{k}"] = "" if np.isnan(v) else round(v, 3)
                        row[f"n_{key}_k{k}"] = n
                a = d[framing]
                sel = (a["pol"] == pol_is_nat) & (a["k"] >= 4)
                judged = a["judge"] >= 0
                row["style_k4plus"] = round(rate(a["adopt"], sel & a["scorable"])[0], 3)
                row["joint_k4plus"] = round(rate(a["adopt"] & (a["judge"] == 1),
                                                 sel & a["scorable"] & judged)[0], 3) \
                    if framing == "trans" else ""
                row["judge_all"] = round(rate(a["judge"] == 1, (a["pol"] == pol_is_nat) & judged)[0], 3) \
                    if framing == "trans" else ""
                row["exact_all"] = round(rate(a["exact"], a["pol"] == pol_is_nat)[0], 3)
                row["scorable_all"] = round(rate(a["scorable"], a["pol"] == pol_is_nat)[0], 3)
                row["judge_fail"] = int(((a["judge"] < 0) & (a["pol"] == pol_is_nat)).sum()) \
                    if framing == "trans" else ""
                row["n"] = int((a["pol"] == pol_is_nat).sum())
                summary.append(row)
        for framing in ("plain", "trans"):
            for key in ("k", "pol", "label", "exact", "judge"):
                npz[f"{p}__{framing}__{key}"] = data[p][framing][key]
        ax.set_title(p, fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(PLOT_BINS)
        ax.grid(alpha=0.3)
    for ax in axes[len(props):]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    axes[len(props)].legend(h, l, fontsize=7.5, loc="center", frameon=False)
    for ax in axes[-ncol:]:
        ax.set_xlabel("k = prior manifestations in the English prefix")
    for ax in axes[::ncol]:
        ax.set_ylabel("P(adopt context convention)")
    fig.suptitle("Style continuation vs k: English-only prefix vs translation framing "
                 "(Spanish source, then English translation in the convention). "
                 "Filled = style AND judge-correct translation; open markers: n < 20. GPT-J-6B, T=1.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "accuracy_by_k.png", dpi=150)
    plt.close(fig)

    # ---- fig 2: translation correctness by k ---------------------------------------------
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, p in zip(axes, props):
        d = data[p]
        allp = curves(d["plain"], np.ones_like(d["plain"]["k"], dtype=bool))
        allt = curves(d["trans"], np.ones_like(d["trans"]["k"], dtype=bool))
        plot_line(ax, allt, "judge", "#2ca02c", "-", "o", "Spanish->English: judge-correct translation", lw=2.2)
        plot_line(ax, allt, "exact", "#2ca02c", "--", "o", "Spanish->English: exact match to reference",
                  filled=False, lw=1.3)
        plot_line(ax, allp, "exact", C_GREY, "--", "o", "English-only: exact match to reference",
                  filled=False, lw=1.3)
        plot_line(ax, allt, "unscorable", "#9467bd", ":", "^", "Spanish->English: style-unscorable fraction",
                  filled=False, lw=1.3)
        plot_line(ax, allp, "unscorable", "#c5b0d5", ":", "^", "English-only: style-unscorable fraction",
                  filled=False, lw=1.3)
        ax.set_title(p, fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(PLOT_BINS)
        ax.grid(alpha=0.3)
    for ax in axes[len(props):]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    axes[len(props)].legend(h, l, fontsize=7.5, loc="center", frameon=False)
    for ax in axes[-ncol:]:
        ax.set_xlabel("k = prior manifestations in the English prefix")
    for ax in axes[::ncol]:
        ax.set_ylabel("rate (all items, both polarities)")
    fig.suptitle("Translation correctness of the sampled fragment vs k (both context polarities pooled). "
                 "Judge = Gemini 2.5 Flash, conventions ignored; exact = normalised prefix match.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "translation_by_k.png", dpi=150)
    plt.close(fig)

    cols = list(summary[0].keys())
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)
    np.savez_compressed(OUT_DIR / "records.npz", **npz)

    # console digest: k=0 vs k>=4, alt-convention docs (the disfavoured pole for most props)
    print(f"{'property':14s} {'framing':6s} {'ctx':3s} {'style k0':>8s} {'style k4+':>9s} "
          f"{'joint k0':>8s} {'joint k4+':>9s} {'judge':>6s} {'exact':>6s} {'unscor':>6s}")
    for r in summary:
        print(f"{r['property']:14s} {r['framing']:6s} {r['ctx']:3s} {str(r['style_k0']):>8s} "
              f"{str(r['style_k4plus']):>9s} {str(r['joint_k0']):>8s} {str(r['joint_k4plus']):>9s} "
              f"{str(r['judge_all']):>6s} {str(r['exact_all']):>6s} "
              f"{str(round(1 - r['scorable_all'], 3)):>6s}")
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
