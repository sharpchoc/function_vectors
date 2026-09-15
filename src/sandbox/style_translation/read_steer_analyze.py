#!/usr/bin/env python
"""Step 7c — results of evidence-token steering with the read-feature difference.

Per family and direction (nat2alt: nat-context k-shot prompt steered toward alt at its evidence tokens; alt2nat: the
reverse): accuracy(arm) = P(TARGET convention at the next decision AND judge OK); unsteered = the base arm of the same
context pole scored toward the target (how often the model already deviates from its context); reference = step-3
accuracy at the same k when the context genuinely was the target pole ("flipped reference").
Gate columns (user decision 2026-09-15): reach = accuracy / reference, pass_50 = reach >= .5 (default rule; the user
judges the final results), sig_vs_unsteered = Wilson CI of the steered accuracy excludes the unsteered rate.
Final setting = the confirmed top-2 setting with the higher accuracy.
Outputs -> results/style_translation/<model>[/<tag>]/read_steer[_k<k>]/: read_steer_summary.png (headline),
read_steer_summary_controls.png (only when a control arm exists), read_steer_layer_alpha.png, best_config.csv,
read_steer_summary.csv, screen.csv, records.npz.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, STYLE_TRANSLATION_RESULTS
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.steer_analyze import wilson
from src.sandbox.style_translation.steer_screen import ALPHAS
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.ml_families import ML_FAMILIES, ML_FAMILY
from src.sandbox.style_translation.read_steer_screen import DIRECTIONS
from src.sandbox.style_translation.family_groups import CODE as CODE_NAMES, grouped_grid, grouped_order

ROOT = ARTIFACTS_ROOT / "style_translation" / "read_steer"
OUT = STYLE_TRANSLATION_RESULTS / "read_steer"
STEP3 = STYLE_TRANSLATION_RESULTS / "summary.csv"
K_CTX = 3
GATE = 0.5


def configure(model="gptj", tag=None, k_ctx=K_CTX):
    global ROOT, OUT, STEP3
    MP = model_paths(model); ROOT = MP["read_steer"]
    R = MP["results"] / tag if tag else MP["results"]
    OUT, STEP3 = R / ("read_steer" if k_ctx == K_CTX else f"read_steer_k{k_ctx}"), R / "summary.csv"


C = {"nat2alt": "#d62728", "alt2nat": "#1f77b4"}      # colour = target pole (alt red, nat blue) as elsewhere


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gptj", help="models.MODELS key")
    ap.add_argument("--tag", default=None, help="results sub-bucket (results/<model>/<tag>/…); step-3 reference from the same bucket")
    ap.add_argument("--families", nargs="*", default=None, help="restrict to these families")
    ap.add_argument("--k_ctx", type=int, default=K_CTX, help="prompt k of the confirm run (3 -> screen/confirm, else screen_k<k>/confirm_k<k>)")
    args = ap.parse_args()
    k = args.k_ctx; configure(args.model, args.tag, k)
    SCREEN = ROOT / ("screen" if k == K_CTX else f"screen_k{k}"); CONFIRM = ROOT / ("confirm" if k == K_CTX else f"confirm_k{k}")
    OUT.mkdir(parents=True, exist_ok=True)
    step3 = {(r["family"], r["style"], int(r["k"])): float(r["accuracy"]) for r in csv.DictReader(open(STEP3))} if STEP3.exists() else {}
    universe = [f.name for f in list(FAMILIES) + list(ML_FAMILIES) if args.families is None or f.name in args.families]
    sfams = [f for f in universe if (SCREEN / f"{f}.json").exists()]
    fams = []
    for f in universe:
        p = CONFIRM / f"{f}.json"
        if p.exists():
            recs = json.load(open(p))
            if sum(r.get("judge") is not None for r in recs) >= 0.95 * len(recs):
                fams.append(f)
            else:
                print(f"PENDING JUDGE: {f}", flush=True)
    code_only = bool(fams or sfams) and all(f in CODE_NAMES for f in (fams or sfams))
    unit = "tasks" if code_only else "texts"

    # ---- screen grid ---------------------------------------------------------------------------
    grid, screen_rows, layers = {}, [], None
    for fam in sfams:
        recs = json.load(open(SCREEN / f"{fam}.json"))
        L_here = sorted({r["layer"] for r in recs if r["direction"] is not None})
        layers = L_here if layers is None else sorted(set(layers) & set(L_here))
        for d, (ctx, target) in DIRECTIONS.items():
            base = [r for r in recs if r["alpha"] == 0.0 and r["context"] == ctx]
            b = float(np.mean([r["decision"] == target for r in base]))
            for layer in L_here:
                grid[(fam, d, layer, 0.0)] = b
                for a in ALPHAS:
                    sel = [r for r in recs if r["direction"] == d and r["layer"] == layer and r["alpha"] == a]
                    grid[(fam, d, layer, a)] = float(np.mean([r["decision"] == target for r in sel]))
                    screen_rows.append(dict(family=fam, direction=d, layer=layer, alpha=a, target_rate=round(grid[(fam, d, layer, a)], 3),
                                            unsteered=round(b, 3), unscorable=round(float(np.mean([r["decision"] is None for r in sel])), 3)))
    if screen_rows:
        with open(OUT / "screen.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(screen_rows[0])); w.writeheader(); w.writerows(screen_rows)

    # ---- confirm -------------------------------------------------------------------------------
    rows, best, npz = [], {}, {}
    for fam in fams:
        recs = [r for r in json.load(open(CONFIRM / f"{fam}.json")) if r.get("judge")]
        for d, (ctx, target) in DIRECTIONS.items():
            arms = [("base", [r for r in recs if r["arm"] == f"base_{ctx}"])] + \
                   [(a, [r for r in recs if r["arm"] == a]) for a in (f"{d}_top1", f"{d}_top2", f"{d}_cf")]
            for arm, sel in arms:
                if not sel:
                    continue
                n = len(sel); acc = float(np.mean([r["decision"] == target and r["judge"]["ok"] for r in sel])); lo, hi = wilson(acc, n)
                rows.append(dict(family=fam, direction=d, context=ctx, target=target, k_ctx=k, arm=arm, layer=sel[0]["layer"], alpha=sel[0]["alpha"],
                                 cf_family=sel[0].get("cf_family"), n=n, accuracy=round(acc, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                                 style_only=round(float(np.mean([r["decision"] == target for r in sel])), 3),
                                 context_style=round(float(np.mean([r["decision"] == ctx for r in sel])), 3),
                                 unscorable=round(float(np.mean([r["decision"] is None for r in sel])), 3),
                                 judge_ok=round(float(np.mean([r["judge"]["ok"] for r in sel])), 3), capped=round(float(np.mean([r["capped"] for r in sel])), 3),
                                 n_positions=round(float(np.mean([r["n_positions"] for r in sel])), 1),
                                 step3_genuine_k=step3.get((fam, target, k)), step3_context_k=step3.get((fam, ctx, k))))
            cands = [r for r in rows if r["family"] == fam and r["direction"] == d and r["arm"].endswith(("top1", "top2"))]
            best[(fam, d)] = max(cands, key=lambda r: (r["accuracy"], -r["alpha"], -r["layer"]))
        npz[f"{fam}__arm"] = np.array([r["arm"] for r in recs]); npz[f"{fam}__decision"] = np.array([{"nat": 1, "alt": 0, None: -1}[r["decision"]] for r in recs])
        npz[f"{fam}__judge_ok"] = np.array([r["judge"]["ok"] for r in recs])
    has_cf = any(r["arm"].endswith("_cf") for r in rows)

    def find(fam, d, arm):
        return next((r for r in rows if r["family"] == fam and r["direction"] == d and r["arm"] == arm), None)

    if rows:
        with open(OUT / "read_steer_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        with open(OUT / "best_config.csv", "w", newline="") as fh:
            keys = ["family", "direction", "context", "target", "k_ctx", "layer", "alpha", "n", "accuracy", "ci_lo", "ci_hi", "style_only", "context_style", "unscorable", "judge_ok",
                    "unsteered_accuracy", "unsteered_ci_lo", "unsteered_ci_hi", "unsteered_judge_ok", "cf_accuracy", "reference", "reach", "pass_50", "sig_vs_unsteered"]
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for (fam, d), b in sorted(best.items(), key=lambda kv: (fams.index(kv[0][0]), kv[0][1])):
                base = find(fam, d, "base"); cf = find(fam, d, f"{d}_cf"); ref = b["step3_genuine_k"]
                reach = round(b["accuracy"] / ref, 3) if ref else None
                w.writerow({kk: b[kk] for kk in keys if kk in b} | dict(unsteered_accuracy=base["accuracy"], unsteered_ci_lo=base["ci_lo"], unsteered_ci_hi=base["ci_hi"],
                                                                      unsteered_judge_ok=base["judge_ok"], cf_accuracy=cf["accuracy"] if cf else None,
                                                                      reference=ref, reach=reach, pass_50=(reach is not None and reach >= GATE),
                                                                      sig_vs_unsteered=b["ci_lo"] > base["accuracy"]))
        np.savez_compressed(OUT / "records.npz", **npz)

        # ---- summary figures ---------------------------------------------------------------------
        def summary_figure(path, with_controls):
            wdt = 3 if with_controls else 2
            fig, gax = grouped_grid(fams, panel_w=3.9, panel_h=3.0, top=0.855, sharey=True)
            for fam in grouped_order(fams):
                ax = gax[(fam, 0)]; x = 0; ticks, labels = [], []
                for d, (ctx, target) in DIRECTIONS.items():
                    base = find(fam, d, "base"); b = best[(fam, d)]; cf = find(fam, d, f"{d}_cf")
                    bars = [(base, "#9e9e9e", "unsteered"), (b, C[d], f"L{b['layer']}, α = {b['alpha']:g}")]
                    if with_controls and cf is not None:
                        bars.append((cf, "#c7c7c7", f"{cf['cf_family']} vector"))
                    for j, (r, col, lab) in enumerate(bars):
                        ax.bar(x + j, r["accuracy"], color=col, edgecolor="black" if j == 1 else "none", linewidth=0.8,
                               yerr=[[r["accuracy"] - r["ci_lo"]], [r["ci_hi"] - r["accuracy"]]], capsize=2)
                        ticks.append(x + j); labels.append(lab)
                    ref = b["step3_genuine_k"]
                    if ref is not None:
                        ax.hlines(ref, x - 0.5, x + wdt - 0.5, colors=C[d], linestyles="dashed", lw=1)
                    ax.text(x + (wdt - 1) / 2, 1.02, f"{ctx} context → {target}", ha="center", fontsize=7.5, color=C[d])
                    x += wdt + 1
                ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
                ax.set_title(fam, fontsize=10); ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=0.3)
            for ax in fig.axes_meta["left"]:
                ax.set_ylabel("accuracy toward the target", fontsize=9)
            handles = [Patch(color="#9e9e9e", label=f"unsteered {k}-shot prompt (rate of the other convention)"),
                       Patch(color=C["nat2alt"], label="nat context steered toward alt at the evidence tokens"),
                       Patch(color=C["alt2nat"], label="alt context steered toward nat at the evidence tokens")]
            if with_controls:
                handles.append(Patch(color="#c7c7c7", label="control: another family's read vector, same positions and (layer, α)"))
            handles.append(Line2D([], [], color="black", linestyle="dashed", label=f"reference: {k}-shot prompt whose context genuinely is the target convention"))
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, fontsize=9.5, frameon=False)
            crit = "the judge finds the code a correct solution" if code_only else "faithful, coherent translation"
            fig.suptitle(f"Steering at the evidence tokens with the read-feature difference: does the model re-read a {k}-shot context as the other convention?\n"
                         f"accuracy = target convention at the next decision AND {crit}; ≤ 200 {unit} per bar, 95% CI", fontsize=12, y=0.995)
            fig.savefig(path, dpi=150); plt.close(fig)
        summary_figure(OUT / "read_steer_summary.png", False)
        if has_cf:
            summary_figure(OUT / "read_steer_summary_controls.png", True)
        elif (OUT / "read_steer_summary_controls.png").exists():
            (OUT / "read_steer_summary_controls.png").unlink()

    # ---- screen line plots -----------------------------------------------------------------------
    if sfams:
        ticks_x = layers if len(layers) <= 10 else [L for L in (0, 4, 8, 12, 16, 20, 24, 27) if L in layers]
        fig, gax = grouped_grid(sfams, slots_per_family=2, fam_cols=(2, 2), panel_w=2.9, panel_h=2.3, top=0.915, bottom=0.04, sharex=True, sharey=True)
        acol = dict(zip(ALPHAS, plt.cm.viridis(np.linspace(0.15, 0.9, len(ALPHAS)))))
        for fam in grouped_order(sfams):
            for slot, d in enumerate(DIRECTIONS):
                ax = gax[(fam, slot)]
                ax.axhline(grid[(fam, d, layers[0], 0.0)], color="#9e9e9e", linestyle="dashed", lw=1.2, label="α = 0 (unsteered)")
                for a in ALPHAS:
                    ax.plot(layers, [grid[(fam, d, layer, a)] for layer in layers], marker="o", ms=2.5, lw=1.3, color=acol[a], label=f"α = {a:g}")
                ax.set_title(f"{fam}: {DIRECTIONS[d][0]} ctx → {DIRECTIONS[d][1]}", fontsize=8, color=C[d])
                ax.set_ylim(-0.03, 1.03); ax.set_xticks(ticks_x); ax.tick_params(labelsize=6.5); ax.grid(alpha=0.3)
        for ax in fig.axes_meta["left"]:
            ax.set_ylabel("target rate", fontsize=8)
        for ax in fig.axes_meta["bottom"]:
            ax.set_xlabel("layer (0 = embeddings)", fontsize=8)
        h, l = next(iter(gax.values())).get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(ALPHAS) + 1, fontsize=9, frameon=False)
        fig.suptitle(f"Screen: rate of the target convention at the next decision vs injection layer (evidence tokens of the {k}-shot prompt steered), one line per α\n"
                     f"style only, 50 {unit} per point, 16-token completions; dashed = unsteered {k}-shot rate", fontsize=11, y=0.995)
        fig.savefig(OUT / "read_steer_layer_alpha.png", dpi=150); plt.close(fig)

    if rows:
        print(f"{'family':18s} {'direction':8s} {'L':>3s} {'α':>4s} {'unst':>5s} {'steer':>5s} {'cf':>5s} {'ref':>5s} {'reach':>5s} {'style':>5s} {'judge':>5s}")
        for (fam, d), b in sorted(best.items(), key=lambda kv: (fams.index(kv[0][0]), kv[0][1])):
            base = find(fam, d, "base"); cf = find(fam, d, f"{d}_cf"); ref = b["step3_genuine_k"]
            print(f"{fam:18s} {d:8s} {b['layer']:3d} {b['alpha']:4g} {base['accuracy']:5.2f} {b['accuracy']:5.2f} {(cf['accuracy'] if cf else float('nan')):5.2f} "
                  f"{(ref if ref is not None else float('nan')):5.2f} {(b['accuracy'] / ref if ref else float('nan')):5.2f} {b['style_only']:5.2f} {b['judge_ok']:5.2f}")
    print("->", OUT)


if __name__ == "__main__":
    main()
