#!/usr/bin/env python
"""Steering effectiveness by injection layer, per property (first-cue spec).

From artifacts/style_properties/steering/sweep_cuecue1/<prop>.json (cue-derived vector,
first cue token, stored rollouts). Metric matches the headline: STRICT = fraction of
rollouts adopting the ALT convention, with unscorable rollouts counted as NOT adopting.

Because high doses can buy convention at the cost of fluency, a coherence line is overlaid:
for each (property, layer) the best-alpha cell is subsampled (--judge_n rollouts) and rated
FLUENT/GIBBERISH by the same LLM judge as the headline. Verdicts cache into the sweep JSON.

Output: results/style_properties/steering/steering_by_layer.png (+ .csv)
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from src.sandbox.ext_styleprops.judge_coherence import judge_one
from src.sandbox.ext_styleprops.gen_corpus import load_key

SWEEP = ARTIFACTS_ROOT / "style_properties" / "steering" / "sweep_cuecue1"
OUT = STYLE_PROPERTIES_DIR / "steering"
LAYERS = (2, 4, 6, 8, 10, 12, 16, 20, 24)
ALPHAS = (2.0, 4.0, 8.0, 16.0, 32.0)
ACOL = {2.0: "#cfe3ea", 4.0: "#8fbcd4", 8.0: "#457b9d", 16.0: "#e63946", 32.0: "#6a040f"}


def strict(prop, cond):
    labs = [PROPS[prop].classify(t) for t in cond["tails"]]
    return float(np.mean([l == "alt" for l in labs]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge_n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--no_judge", action="store_true")
    args = ap.parse_args()
    pool = set(json.load(open(REPO_ROOT / "task_splits" / "style_properties_pool.json"))["pass"])
    props = sorted(p.stem for p in SWEEP.glob("*.json") if p.stem in pool)
    OUT.mkdir(parents=True, exist_ok=True)
    key = None if args.no_judge else load_key()
    rng = np.random.RandomState(0)

    n, ncol = len(props), 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.1 * nrow), squeeze=False,
                             sharex=True, sharey=True)
    rows = []
    for pi, name in enumerate(props):
        f = SWEEP / f"{name}.json"
        d = json.load(open(f)); C = d["conditions"]; ctx = d["ctx"]
        base = strict(name, C["baseline_nat2alt"])
        grid = {(L, a): strict(name, C[f"cuediff_cue_L{L}_a{a}"])
                for L in LAYERS for a in ALPHAS if f"cuediff_cue_L{L}_a{a}" in C}
        # coherence on the best-alpha cell per layer (subsample, cached)
        coh_line, jobs = {}, []
        for L in LAYERS:
            cells = [(a, grid[(L, a)]) for a in ALPHAS if (L, a) in grid]
            if not cells:
                continue
            besta = max(cells, key=lambda t: t[1])[0]
            cond = C[f"cuediff_cue_L{L}_a{besta}"]
            cached = cond.get("coherent_sub")
            if cached is not None:
                coh_line[L] = float(np.mean([v for v in cached if v is not None])) if any(
                    v is not None for v in cached) else np.nan
            elif key:
                idx = rng.choice(len(cond["tails"]), min(args.judge_n, len(cond["tails"])), replace=False)
                jobs.append((L, besta, list(idx)))
        if jobs and key:
            for L, besta, idx in jobs:
                cond = C[f"cuediff_cue_L{L}_a{besta}"]
                verdicts = [None] * len(idx)
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = {ex.submit(judge_one, key, args.model, ctx[i], cond["tails"][i]): j
                            for j, i in enumerate(idx)}
                    for fu in as_completed(futs):
                        verdicts[futs[fu]] = fu.result()
                cond["coherent_sub"] = verdicts
                ok = [v for v in verdicts if v is not None]
                coh_line[L] = float(np.mean(ok)) if ok else np.nan
            json.dump(d, open(f, "w"), indent=1)

        ax = axes[pi // ncol][pi % ncol]
        for a in ALPHAS:
            ys = [grid.get((L, a), np.nan) for L in LAYERS]
            ax.plot(LAYERS, ys, "o-", ms=3.2, lw=1.4, color=ACOL[a], label=f"α={a:g}")
        if coh_line:
            ax.plot(sorted(coh_line), [coh_line[L] for L in sorted(coh_line)], "s--", ms=2.6,
                    lw=1.1, color="#2a9d8f", label="coherent share (best α)")
        ax.axhline(base, color="#888", ls=":", lw=1, label="unsteered")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_ylim(-0.03, 1.03); ax.set_xticks(LAYERS); ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if pi % ncol == 0:
            ax.set_ylabel("adopt ALT convention\n(unscorable = no)", fontsize=8)
        if pi + ncol >= n:
            ax.set_xlabel("injection layer", fontsize=8)
        for L in LAYERS:
            for a in ALPHAS:
                if (L, a) in grid:
                    rows.append(dict(property=name, layer=L, alpha=a, strict=round(grid[(L, a)], 3),
                                     baseline=round(base, 3),
                                     coherent_best_alpha=round(coh_line[L], 3) if L in coh_line else ""))
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", bbox_to_anchor=(0.98, 0.04), fontsize=8.5,
               title="dose α (× cue-derived vector), added at the first cue token")
    fig.suptitle("Steering effectiveness by injection layer — fraction of rollouts adopting the ALT convention\n"
                 "(first cue token of each document; unscorable rollouts count as NOT adopting; "
                 "green dashed = share judged fluent at that layer's best dose)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "steering_by_layer.png", dpi=150)
    with open(OUT / "steering_by_layer.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"-> {OUT}/steering_by_layer.png")


if __name__ == "__main__":
    main()
