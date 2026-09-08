#!/usr/bin/env python
"""Step 4d — steering results: best (layer, alpha) per family and style, summary figure, heatmaps.

accuracy(arm) = P(target style used at the cue AND judge OK) over the 200 k=0 texts (same grading
as step 3). Final setting per (family, target) = the confirmed top-2 setting with the higher
accuracy. Controls: base (alpha = 0, same seeds), counterfactual-family vector at the top-1 setting.
Outputs -> results/style_translation/steering/: steering_summary.png (headline, no controls),
steering_summary_controls.png (with the other-family control), screen_heatmaps.png,
best_config.csv, steering_summary.csv, screen.csv, records.npz.
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
from src.sandbox.style_translation.steer_screen import SCREEN_LAYERS, ALPHAS

SCREEN = ARTIFACTS_ROOT / "style_translation" / "steering" / "screen"
CONFIRM = ARTIFACTS_ROOT / "style_translation" / "steering" / "confirm"
OUT = STYLE_TRANSLATION_RESULTS / "steering"
STEP3 = STYLE_TRANSLATION_RESULTS / "summary.csv"


def wilson(p, n, z=1.96):
    den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sfams = [f.name for f in FAMILIES if (SCREEN / f"{f.name}.json").exists()]          # screen: judge-free
    fams, pending = [], []
    for f in FAMILIES:                                                                    # confirm: needs the judge
        if not (CONFIRM / f"{f.name}.json").exists():
            continue
        recs = json.load(open(CONFIRM / f"{f.name}.json"))
        judged = sum(r.get("judge") is not None for r in recs)
        (fams if judged >= 0.95 * len(recs) else pending).append(f.name)
        if 0 < len(recs) - judged < 0.05 * len(recs):
            print(f"{f.name}: {len(recs) - judged} unjudged records skipped", flush=True)
    if pending:
        print(f"PENDING JUDGE (excluded from confirm outputs): {pending}", flush=True)
    step3 = {(r["family"], r["style"], int(r["k"])): float(r["accuracy"]) for r in csv.DictReader(open(STEP3))}

    # ---- screen grid csv ----------------------------------------------------------------------
    screen_rows, grid = [], {}
    for fam in sfams:
        recs = json.load(open(SCREEN / f"{fam}.json"))
        base = [r for r in recs if r["alpha"] == 0.0]
        for target in ("nat", "alt"):
            b = np.mean([r["decision"] == target for r in base])
            for layer in SCREEN_LAYERS:
                grid[(fam, target, layer, 0.0)] = b
                screen_rows.append(dict(family=fam, target=target, layer=layer, alpha=0.0, target_rate=round(b, 3), unscorable=round(np.mean([r["decision"] is None for r in base]), 3)))
                for a in ALPHAS:
                    sel = [r for r in recs if r["target"] == target and r["layer"] == layer and r["alpha"] == a]
                    rate = np.mean([r["decision"] == target for r in sel]); grid[(fam, target, layer, a)] = rate
                    screen_rows.append(dict(family=fam, target=target, layer=layer, alpha=a, target_rate=round(rate, 3), unscorable=round(np.mean([r["decision"] is None for r in sel]), 3)))
    with open(OUT / "screen.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(screen_rows[0])); w.writeheader(); w.writerows(screen_rows)

    # ---- confirm: per arm accuracy ------------------------------------------------------------
    rows, best, npz = [], {}, {}
    for fam in fams:
        recs = [r for r in json.load(open(CONFIRM / f"{fam}.json")) if r.get("judge")]
        arms = sorted({r["arm"] for r in recs}, key=lambda a: (a != "base", a))
        base_recs = [r for r in recs if r["arm"] == "base"]
        for arm in arms:
            sel = [r for r in recs if r["arm"] == arm]
            n = len(sel)
            for target in (("nat", "alt") if arm == "base" else (sel[0]["style"],)):
                style_ok = [r["decision"] == target for r in sel]
                acc = np.mean([s and r["judge"]["ok"] for s, r in zip(style_ok, sel)])
                lo, hi = wilson(acc, n)
                rows.append(dict(family=fam, target=target, arm=arm if arm != "base" else "base", layer=sel[0]["layer"], alpha=sel[0]["alpha"],
                                 cf_family=sel[0].get("cf_family"), n=n, accuracy=round(acc, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                                 style_only=round(np.mean(style_ok), 3), unscorable=round(np.mean([r["decision"] is None for r in sel]), 3),
                                 judge_ok=round(np.mean([r["judge"]["ok"] for r in sel]), 3), capped=round(np.mean([r["capped"] for r in sel]), 3),
                                 step3_k0=step3.get((fam, target, 0)), step3_k4=step3.get((fam, target, 4))))
        for target in ("nat", "alt"):
            cands = [r for r in rows if r["family"] == fam and r["target"] == target and r["arm"].startswith(f"{target}_top")]
            best[(fam, target)] = max(cands, key=lambda r: (r["accuracy"], -r["alpha"], -r["layer"]))
        npz[f"{fam}__arm"] = np.array([r["arm"] for r in recs]); npz[f"{fam}__decision"] = np.array([{"nat": 1, "alt": 0, None: -1}[r["decision"]] for r in recs])
        npz[f"{fam}__judge_ok"] = np.array([r["judge"]["ok"] for r in recs])
    with open(OUT / "steering_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT / "best_config.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["family", "target", "layer", "alpha", "accuracy", "ci_lo", "ci_hi", "style_only", "unscorable", "judge_ok",
                                           "base_accuracy", "cf_accuracy", "step3_k0", "step3_k4"])
        w.writeheader()
        for (fam, target), b in sorted(best.items(), key=lambda kv: (fams.index(kv[0][0]), kv[0][1])):
            base = [r for r in rows if r["family"] == fam and r["target"] == target and r["arm"] == "base"][0]
            cf = [r for r in rows if r["family"] == fam and r["target"] == target and r["arm"] == f"{target}_cf"][0]
            w.writerow(dict(family=fam, target=target, layer=b["layer"], alpha=b["alpha"], accuracy=b["accuracy"], ci_lo=b["ci_lo"], ci_hi=b["ci_hi"],
                            style_only=b["style_only"], unscorable=b["unscorable"], judge_ok=b["judge_ok"], base_accuracy=base["accuracy"],
                            cf_accuracy=cf["accuracy"], step3_k0=b["step3_k0"], step3_k4=b["step3_k4"]))
    np.savez_compressed(OUT / "records.npz", **npz)

    # ---- summary figures: headline (no controls) + detailed (with the other-family control) ----
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    C = {"base": "#9e9e9e", "steer": {"nat": "#1f77b4", "alt": "#d62728"}, "cf": "#c7c7c7"}

    def summary_figure(path, with_controls):
        ncol = 4; nrow = math.ceil(len(fams) / ncol); w = 3 if with_controls else 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.0 * nrow), sharey=True)
        axes = axes.ravel()
        for ax, fam in zip(axes, fams):
            x = 0; ticks, labels = [], []
            for target in ("nat", "alt"):
                base = [r for r in rows if r["family"] == fam and r["target"] == target and r["arm"] == "base"][0]
                b = best[(fam, target)]
                cf = [r for r in rows if r["family"] == fam and r["target"] == target and r["arm"] == f"{target}_cf"][0]
                bars = [(base, C["base"], "no steering"), (b, C["steer"][target], f"steered L{b['layer']} α{b['alpha']:g}")]
                if with_controls:
                    bars.append((cf, C["cf"], f"{cf['cf_family']}'s vector"))
                for j, (r, col, lab) in enumerate(bars):
                    ax.bar(x + j, r["accuracy"], color=col, edgecolor="black" if j == 1 else "none", linewidth=0.8,
                           yerr=[[r["accuracy"] - r["ci_lo"]], [r["ci_hi"] - r["accuracy"]]], capsize=2)
                    ticks.append(x + j); labels.append(lab)
                k4 = b["step3_k4"]
                if k4 is not None:
                    ax.hlines(k4, x - 0.5, x + w - 0.5, colors=C["steer"][target], linestyles="dashed", lw=1)
                ax.text(x + (w - 1) / 2, 1.02, f"→ {target}", ha="center", fontsize=8, color=C["steer"][target])
                x += w + 1
            ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
            ax.set_title(fam, fontsize=10); ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=0.3)
        for ax in axes[len(fams):]:
            ax.axis("off")
        for ax in axes[::ncol]:
            ax.set_ylabel("accuracy at k = 0", fontsize=9)
        handles = [Patch(color=C["base"], label="no steering (k = 0 prompt as is)"),
                   Patch(color=C["steer"]["nat"], label="steered toward nat: the family's own vector, best (layer, α)"),
                   Patch(color=C["steer"]["alt"], label="steered toward alt: the family's own vector, best (layer, α)")]
        if with_controls:
            handles.append(Patch(color=C["cf"], label="control: another family's vector (named on the axis) at the same (layer, α)"))
        handles.append(Line2D([], [], color="black", linestyle="dashed",
                              label="reference: accuracy with 4 in-context examples and NO steering (step 3), same colour code"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, fontsize=9, frameon=False)
        fig.suptitle("Can one mean-difference vector, added at the first cue token, induce a convention with no in-context example?\n"
                     "accuracy = target convention used at the cue AND faithful, coherent translation (Gemini judge); n = 200 texts per bar, 95% CI",
                     fontsize=11, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(path, dpi=150); plt.close(fig)

    summary_figure(OUT / "steering_summary.png", with_controls=False)
    summary_figure(OUT / "steering_summary_controls.png", with_controls=True)

    # ---- screen heatmaps -----------------------------------------------------------------------
    fig, axes = plt.subplots(len(sfams), 2, figsize=(7.5, 1.55 * len(sfams)))
    for i, fam in enumerate(sfams):
        for j, target in enumerate(("nat", "alt")):
            ax = axes[i, j]
            M = np.array([[grid[(fam, target, layer, a)] for layer in SCREEN_LAYERS] for a in [0.0] + ALPHAS])
            im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_yticks(range(len(ALPHAS) + 1)); ax.set_yticklabels([f"α={a:g}" for a in [0.0] + ALPHAS], fontsize=6)
            ax.set_xticks(range(len(SCREEN_LAYERS))); ax.set_xticklabels([f"L{l}" for l in SCREEN_LAYERS], fontsize=6)
            ax.set_title(f"{fam} → {target}  (unsteered {grid[(fam, target, SCREEN_LAYERS[0], 0.0)]:.2f})", fontsize=7.5)
            for (r, c), v in np.ndenumerate(M):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=5, color="white" if v < 0.6 else "black")
    fig.suptitle("Screen: target-style rate (style only, 50 texts, 16-token completions) by layer and α", fontsize=10, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.99)); fig.savefig(OUT / "screen_heatmaps.png", dpi=150); plt.close(fig)

    print(f"{'family':14s} {'tgt':3s} {'L':>3s} {'α':>4s} {'base':>5s} {'steer':>5s} {'cf':>5s} {'k4 ref':>6s} {'style-only':>10s} {'unscor':>6s} {'judge':>5s}")
    for (fam, target), b in sorted(best.items(), key=lambda kv: (fams.index(kv[0][0]), kv[0][1])):
        base = [r for r in rows if r["family"] == fam and r["target"] == target and r["arm"] == "base"][0]
        cf = [r for r in rows if r["family"] == fam and r["target"] == target and r["arm"] == f"{target}_cf"][0]
        print(f"{fam:14s} {target:3s} {b['layer']:3d} {b['alpha']:4g} {base['accuracy']:5.2f} {b['accuracy']:5.2f} {cf['accuracy']:5.2f} "
              f"{(b['step3_k4'] or float('nan')):6.2f} {b['style_only']:10.2f} {b['unscorable']:6.2f} {b['judge_ok']:5.2f}")
    print("->", OUT)


if __name__ == "__main__":
    main()
