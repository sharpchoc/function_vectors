#!/usr/bin/env python
"""Steering results: sweep heatmaps, headline bars, sparse-head summary.

Reads artifacts/style_properties/steering/{sweep,full}/<prop>.json and
artifacts/style_properties/sparse_heads/<prop>.npz. Outputs in
results/style_properties/steering/:
  sweep_heatmaps.png    layer x alpha adherence-to-alt per property (evid injection)
  headline_bars.png     per property: baselines vs meandiff (evid/dec), cf control,
                        rawalt best, headsum best; both directions for meandiff
  steering_summary.csv  the numbers behind the bars
  sparse_heads_summary.csv  heads per property + overlap with the 37 ICL FV heads
  head_overlap_matrix.png   property x property selected-head overlap (Jaccard)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT, STYLE_PROPERTIES_DIR
from src.sandbox.ext_styleprops.properties import PROPS

STEER = ARTIFACTS_ROOT / "style_properties" / "steering"
HEADS = ARTIFACTS_ROOT / "style_properties" / "sparse_heads"
OUT = STYLE_PROPERTIES_DIR / "steering"
ICL_SELECTION = ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43" / "pooled_sparse" / "selection.json"
SWEEP_LAYERS = (2, 4, 6, 8, 10, 12, 16, 20, 24)
SWEEP_ALPHAS = (2.0, 4.0, 8.0, 16.0)


def icl_heads():
    for cand in (ICL_SELECTION,):
        if cand.exists():
            sel = json.load(open(cand))
            heads = sel.get("selected_heads") or sel.get("heads") or sel.get("selected")
            out = set()
            for h in heads:
                if isinstance(h, (list, tuple)):
                    out.add(int(h[0]) * 16 + int(h[1]))
                elif isinstance(h, dict):
                    out.add(int(h["layer"]) * 16 + int(h["head"]))
                else:
                    out.add(int(h))
            return out, str(cand)
    return None, None


def main():
    pool_pass = set(json.load(open(REPO_ROOT / "task_splits" / "style_properties_pool.json"))["pass"])
    props = sorted(p.stem for p in (STEER / "full").glob("*.json") if p.stem in pool_pass)
    OUT.mkdir(parents=True, exist_ok=True)

    # sweep heatmaps
    n = len(props)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow), squeeze=False)
    for pi, name in enumerate(props):
        sw = json.load(open(STEER / "sweep" / f"{name}.json"))["conditions"]
        base = sw["baseline_nat2alt"]["adherence_tgt"]
        M = np.full((len(SWEEP_ALPHAS), len(SWEEP_LAYERS)), np.nan)
        for li, L in enumerate(SWEEP_LAYERS):
            for ai, a in enumerate(SWEEP_ALPHAS):
                c = sw.get(f"meandiff_evid_L{L}_a{a}")
                if c:
                    M[ai, li] = c["adherence_tgt"]
        ax = axes[pi // ncol][pi % ncol]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(SWEEP_LAYERS)), SWEEP_LAYERS, fontsize=7)
        ax.set_yticks(range(len(SWEEP_ALPHAS)), SWEEP_ALPHAS, fontsize=7)
        ax.set_title(f"{name} (base {base:.2f})", fontsize=9)
        ax.set_xlabel("capture layer", fontsize=7)
        ax.set_ylabel("alpha", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.04)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Layer × dose sweep of the mean-difference steering vector (added at evidence tokens of standard-convention text)\n"
                 "colour = fraction of T=1 samples following the ALT convention at decision points (panel title: unsteered baseline); "
                 "capture layer 0 = embedding, l = output of block l−1", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "sweep_heatmaps.png", dpi=150)

    # headline bars + summary csv
    # (condition key, legend text, color, hatch, x-slot). Two groups: make the ALT
    # convention appear in standard (nat) text; then the reverse direction.
    conds_show = [
        ("baseline_nat2alt", "unsteered baseline: how often the model follows the ALT convention on its own",
         "#9e9e9e", None, 0),
        ("meandiff_evid_nat2alt", "mean-DIFFERENCE vector (alt − nat evidence-token mean) added at the EVIDENCE tokens",
         "#e63946", None, 1),
        ("meandiff_dec_nat2alt", "same mean-difference vector, added at the DECISION token only",
         "#f4a261", None, 2),
        ("cfprop_evid_nat2alt", "CONTROL: a different property's mean-difference vector at the evidence tokens (should stay near baseline)",
         "#457b9d", "//", 3),
        ("rawalt_best", "RAW alt-polarity mean vector (not the difference), best α of {0.5, 1, 2}",
         "#2a9d8f", None, 4),
        ("headsum_best", "sparse head-sum vector (Σ selected heads' evidence-token differences), best α of {1, 2, 4, 8}",
         "#9b5de5", None, 5),
        ("baseline_alt2nat", "REVERSE baseline: alt-convention text, how often the model reverts to NAT on its own",
         "#cfcfcf", None, 7),
        ("meandiff_evid_alt2nat", "REVERSE steering: negated mean-difference vector at the evidence tokens of alt text",
         "#e76f51", None, 8),
    ]
    rows = []
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), squeeze=False)
    for pi, name in enumerate(props):
        fl = json.load(open(STEER / "full" / f"{name}.json"))
        c = fl["conditions"]
        raw_best = max((k for k in c if k.startswith("rawalt")),
                       key=lambda k: c[k]["adherence_tgt"], default=None)
        hs_best = max((k for k in c if k.startswith("headsum")),
                      key=lambda k: c[k]["adherence_tgt"], default=None)
        get = {"rawalt_best": raw_best, "headsum_best": hs_best}
        row = {"property": name, "best_L": fl["best_from_sweep"]["L"],
               "best_alpha": fl["best_from_sweep"]["alpha"], "cf_property": fl["cf_property"]}
        ax = axes[pi // ncol][pi % ncol]
        for key, lab, color, hatch, slot in conds_show:
            k = get.get(key, key)
            v = c[k]["adherence_tgt"] if k and k in c else np.nan
            row[key] = round(v, 3) if not np.isnan(v) else ""
            if key in ("rawalt_best", "headsum_best") and k:
                row[key + "_cond"] = k
            if not np.isnan(v):
                ax.bar(slot, v, color=color, hatch=hatch, edgecolor="white" if hatch else color,
                       width=0.8)
                ax.text(slot, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        rows.append(row)
        ax.axvline(6.0, color="#666", lw=0.8, ls=":")
        ax.set_xticks([2.5, 7.5], ["make ALT appear\nin nat text", "reverse:\nalt → nat"],
                      fontsize=7)
        ax.set_xlim(-0.6, 8.6)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="y", labelsize=7)
        if pi % ncol == 0:
            ax.set_ylabel("adherence to TARGET\nconvention", fontsize=8)
        prop = PROPS[name]
        import textwrap
        t1 = textwrap.fill(f"{name}:  {prop.nat_label}  →  {prop.alt_label}", 58)
        ax.set_title(t1 + f"\nvector from capture layer {row['best_L']}, α={row['best_alpha']:g}; "
                     f"control vector = {row['cf_property']}", fontsize=7)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=col, hatch=h, edgecolor="white" if h else col, label=lab)
               for _, lab, col, h, _ in conds_show]
    wrapped = [textwrap.fill(hh.get_label(), 62) for hh in handles]
    fig.suptitle("Can a steering vector make GPT-J switch style convention?\n"
                 "Each bar = fraction of T=1 sampled continuations that follow the TARGET convention at the decision points "
                 "(scorable samples only; 80 docs × ≤8 sites per property). Higher = steering worked.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    n_empty = nrow * ncol - n
    if n_empty >= 2:
        # legend in the empty slots of the last row
        left = axes[nrow - 1][ncol - n_empty].get_position()
        right = axes[nrow - 1][ncol - 1].get_position()
        fig.legend(handles=handles, labels=wrapped, loc="center", fontsize=8, frameon=True,
                   ncol=1 if n_empty < 3 else 2,
                   bbox_to_anchor=((left.x0 + right.x1) / 2, (left.y0 + left.y1) / 2),
                   bbox_transform=fig.transFigure, title="bar colours", title_fontsize=9)
    else:
        fig.legend(handles=handles, labels=wrapped, loc="lower center", ncol=2, fontsize=8,
                   frameon=False)
        fig.subplots_adjust(bottom=0.18)
    fig.savefig(OUT / "headline_bars.png", dpi=150)
    with open(OUT / "steering_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)

    # sparse-head summaries
    icl, icl_src = icl_heads()
    hrows, sels = [], {}
    for name in props:
        p = HEADS / f"{name}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        sel = set(int(x) for x in z["selected"])
        sels[name] = sel
        hrows.append({"property": name, "n_heads": len(sel),
                      "inject_block": int(z["inject_block"]),
                      "overlap_with_37_icl": len(sel & icl) if icl else "",
                      "heads_layer_range": f"{min(s // 16 for s in sel)}-{max(s // 16 for s in sel)}"})
    if hrows:
        with open(OUT / "sparse_heads_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hrows[0]))
            w.writeheader()
            w.writerows(hrows)
        names = list(sels)
        J = np.zeros((len(names), len(names)))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                J[i, j] = len(sels[a] & sels[b]) / max(len(sels[a] | sels[b]), 1)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(J, vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_title("selected-head overlap between properties (Jaccard)")
        fig.colorbar(im, fraction=0.04)
        fig.tight_layout()
        fig.savefig(OUT / "head_overlap_matrix.png", dpi=150)
        print("ICL selection source:", icl_src)
    for r in rows:
        print(r)
    for r in hrows:
        print(r)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
