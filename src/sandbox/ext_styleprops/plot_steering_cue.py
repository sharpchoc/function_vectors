#!/usr/bin/env python
"""Cue-token steering results — the write-feature analog.

Reads artifacts/style_properties/steering/{sweep_cue,full_cue}/<prop>.json and
artifacts/style_properties/sparse_heads_cue/<prop>.npz. Outputs in
results/style_properties/steering/cue/:
  headline_cue.png            THE result: per property unsteered / steered at the cue / natural reference
  steering_by_layer_cue.png   adherence vs injection layer at the cue, one line per dose
                              (solid = cue-derived vector, dashed = evidence-derived vector)
  appendix_cue_all_conditions.png  every cue-site condition (control, reverse, raw mean, head-sum, other vector)
  steering_cue_summary.csv, sparse_heads_cue_summary.csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, STYLE_PROPERTIES_DIR
from src.sandbox.ext_styleprops.plot_steering import icl_heads

STEER = ARTIFACTS_ROOT / "style_properties" / "steering"
HEADS = ARTIFACTS_ROOT / "style_properties" / "sparse_heads_cue"
PRE = ARTIFACTS_ROOT / "style_properties" / "prescreen"
OUT = STYLE_PROPERTIES_DIR / "steering" / "cue"
LAYERS = (2, 4, 6, 8, 10, 12, 16, 20, 24)
ALPHAS = (2.0, 4.0, 8.0, 16.0, 32.0)
ACOL = {2.0: "#a8dadc", 4.0: "#457b9d", 8.0: "#e63946", 16.0: "#6a040f", 32.0: "#222222"}


def main():
    pool = set(json.load(open(REPO_ROOT / "task_splits" / "style_properties_pool.json"))["pass"])
    props = sorted(p.stem for p in (STEER / "full_cue").glob("*.json") if p.stem in pool)
    OUT.mkdir(parents=True, exist_ok=True)
    n, ncol = len(props), 4
    nrow = int(np.ceil(n / ncol))

    # ---- headline: unsteered / steered at cue / natural reference
    trip, rows = [], []
    for name in props:
        fl = json.load(open(STEER / "full_cue" / f"{name}.json"))
        c, b = fl["conditions"], fl["best_from_sweep"]
        # cue-derived vector only (user decision); if the sweep's overall best was the
        # evidence-derived vector, the cue-derived one was run at ITS best setting
        if b["vector"] != "cuediff":
            sw = json.load(open(STEER / "sweep_cue" / f"{name}.json"))["conditions"]
            cb = max((k for k in sw if k.startswith("cuediff") and not np.isnan(sw[k]["adherence_tgt"])),
                     key=lambda k: sw[k]["adherence_tgt"])
            b = {"vector": "cuediff", "L": int(cb.split("_L")[1].split("_")[0]),
                 "alpha": float(cb.split("_a")[1])}
            steered_key = "cuediff_cue_nat2alt_best"
        else:
            steered_key = "cuediff_cue_nat2alt"
        recs = json.load(open(PRE / f"{name}.json"))["records"]
        ref = [r for r in recs if r["pol"] == "alt" and r["k"] >= 4 and r["label"]]
        ref_v = float(np.mean([r["label"] == "alt" for r in ref])) if ref else np.nan
        trip.append((name, c["baseline_nat2alt"]["adherence_tgt"],
                     c[steered_key]["adherence_tgt"], ref_v, b))
        row = {"property": name, "vector": b["vector"], "L": b["L"], "alpha": b["alpha"],
               "cf_property": fl["cf_property"]}
        for k, v in c.items():
            row[k] = v["adherence_tgt"]
        rows.append(row)
    trip.sort(key=lambda t: -t[2])
    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = np.arange(len(trip)); w = 0.27
    ax.bar(x - w, [t[1] for t in trip], w, color="#bdbdbd", label="unsteered (standard-convention text)")
    ax.bar(x, [t[2] for t in trip], w, color="#e63946",
           label="steered: cue-derived property vector added at the CUE token only")
    ax.bar(x + w, [t[3] for t in trip], w, color="#457b9d",
           label="reference: model reading genuine ALT-convention context (k ≥ 4)")
    for xi, t in zip(x, trip):
        ax.text(xi, t[2] + 0.02, f"L{t[4]['L']}", ha="center", fontsize=6.5, color="#555")
    ax.set_xticks(x, [t[0] for t in trip], rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fraction of sampled continuations\nin the ALT convention", fontsize=10)
    ax.set_title("Write-feature analog: a property vector added at the cue token alone makes GPT-J "
                 "switch style convention\n(GPT-J, T=1 sampling at the cue; all cue tokens incl. k=0; "
                 "label above bar = injection layer)", fontsize=11.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8.5, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "headline_cue.png", dpi=170, bbox_inches="tight")

    # ---- by layer: solid = cue-derived vector, dashed = evidence-derived vector
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.0 * nrow), squeeze=False,
                             sharex=True, sharey=True)
    for pi, name in enumerate(props):
        sw = json.load(open(STEER / "sweep_cue" / f"{name}.json"))["conditions"]
        ax = axes[pi // ncol][pi % ncol]
        for a in ALPHAS:
            ys = [sw.get(f"cuediff_cue_L{L}_a{a}", {}).get("adherence_tgt", np.nan) for L in LAYERS]
            ax.plot(LAYERS, ys, "-", marker="o", ms=3, lw=1.3, color=ACOL[a], label=f"α={a:g}")
        ax.axhline(sw["baseline_nat2alt"]["adherence_tgt"], color="#888", ls=":", lw=1, label="unsteered")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_ylim(-0.03, 1.03); ax.set_xticks(LAYERS); ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
        if pi % ncol == 0:
            ax.set_ylabel("adherence to ALT convention", fontsize=8)
        if pi + ncol >= n:
            ax.set_xlabel("injection layer (at the cue token)", fontsize=8)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", bbox_to_anchor=(0.97, 0.05), fontsize=9,
               title="dose α (× cue-derived vector)")
    fig.suptitle("Cue-token steering by injection layer (cue-derived vector added at the cue token only)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "steering_by_layer_cue.png", dpi=150)

    # ---- appendix: all cue-site conditions
    show = [("baseline_nat2alt", "unsteered", "#9e9e9e"),
            ("BEST", "cue-derived vector at cue", "#e63946"),
            ("CF", "control: other property's vector", "#457b9d"),
            ("RAW", "raw alt-mean at cue (best α)", "#2a9d8f"),
            ("HS", "cue-trained head-sum (best α)", "#9b5de5"),
            ("baseline_alt2nat", "reverse baseline", "#cfcfcf"),
            ("REV", "reverse steering", "#e76f51")]
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.4 * nrow), squeeze=False)
    for pi, name in enumerate(props):
        fl = json.load(open(STEER / "full_cue" / f"{name}.json")); c = fl["conditions"]; vs = fl["best_from_sweep"]["vector"]
        def best_of(prefix):
            ks = [k for k in c if k.startswith(prefix)]
            return max((c[k]["adherence_tgt"] for k in ks), default=np.nan)
        cue_based = vs == "cuediff"
        vals = {"baseline_nat2alt": c["baseline_nat2alt"]["adherence_tgt"],
                "BEST": c["cuediff_cue_nat2alt"]["adherence_tgt"] if cue_based
                        else best_of("cuediff_cue_nat2alt_best"),
                "CF": c["cfprop_cue_nat2alt"]["adherence_tgt"] if cue_based else np.nan,
                "RAW": best_of("rawalt_cue") if cue_based else np.nan, "HS": best_of("headsum"),
                "baseline_alt2nat": c["baseline_alt2nat"]["adherence_tgt"],
                "REV": c["cuediff_cue_alt2nat"]["adherence_tgt"] if cue_based else np.nan}
        ax = axes[pi // ncol][pi % ncol]
        for i, (k, lab, col) in enumerate(show):
            v = vals[k]
            if not np.isnan(v):
                ax.bar(i, v, color=col); ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6)
        ax.set_xticks([]); ax.set_ylim(0, 1.12)
        ax.set_title(name + ("" if cue_based else "  (control/reverse not run with cue vector)"), fontsize=8.5)
        if pi % ncol == 0:
            ax.set_ylabel("adherence to target", fontsize=8)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.legend(handles=[Patch(facecolor=col, label=lab) for _, lab, col in show], loc="lower right",
               bbox_to_anchor=(0.98, 0.04), fontsize=8.5, ncol=2, title="bar colours")
    fig.suptitle("Cue-token steering — all conditions (appendix)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "appendix_cue_all_conditions.png", dpi=150)

    with open(OUT / "steering_cue_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    icl, _ = icl_heads(); hrows = []
    for name in props:
        pth = HEADS / f"{name}.npz"
        if pth.exists():
            z = np.load(pth, allow_pickle=True); sel = set(int(v) for v in z["selected"])
            hrows.append({"property": name, "n_heads": len(sel), "inject_block": int(z["inject_block"]),
                          "overlap_with_37_icl": len(sel & icl) if icl else "",
                          "expected_by_chance": round(37 * len(sel) / 448, 1)})
    if hrows:
        with open(OUT / "sparse_heads_cue_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hrows[0])); w.writeheader(); w.writerows(hrows)
    for t in trip:
        print(f"{t[0]:15s} base={t[1]:.2f} steered@cue={t[2]:.2f} ref={t[3]:.2f}  ({t[4]['vector']} L{t[4]['L']} α{t[4]['alpha']:g})")
    for r in hrows:
        print(r)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
