#!/usr/bin/env python
"""Step 5d — results of the sparse-head style steering.

accuracy(arm) = P(alt convention used at the cue AND judge OK) on the 80 held-out texts per family
(doc_id order 120..199), same grading as steps 3/4. References restricted to the SAME 80 texts:
step-4 residual mean-difference vector at its best (L, alpha) (confirm records, arm alt_top* from
best_config.csv), step-4 unsteered base, and step-3 accuracy with 4 in-context examples (rollouts).
Outputs -> results/style_translation/sparse_heads/: sparse_summary.png (headline), sparse_summary.csv,
layer_lambda.csv (copied), heads_vs_accuracy.png, selected_heads_map.png, records.npz.
"""
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, STYLE_TRANSLATION_RESULTS
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.steer_analyze import wilson

ROOT = ARTIFACTS_ROOT / "style_translation" / "sparse_heads"
CONFIRM = ARTIFACTS_ROOT / "style_translation" / "steering" / "confirm"
ROLL = ARTIFACTS_ROOT / "style_translation" / "rollouts"
BEST4 = STYLE_TRANSLATION_RESULTS / "steering" / "best_config.csv"
OUT = STYLE_TRANSLATION_RESULTS / "sparse_heads"
N_FIT = 120
ARMS = ["base", "sparse_w", "sparse_unw", "sparse_unw_hi", "sparse_cf"]
LABEL = {"base": "unsteered", "sparse_w": "sparse heads, learned weights", "sparse_unw": "sparse heads, unweighted sum",
         "sparse_unw_hi": "sparse heads c > 0.8, unweighted", "sparse_cf": "other family's head vector", "resid4": "residual mean-difference vector (step 4)"}
COL = {"base": "#9e9e9e", "sparse_w": "#d62728", "sparse_unw": "#ff9896", "sparse_unw_hi": "#f7c6c5", "sparse_cf": "#c7c7c7", "resid4": "#1f77b4"}


def acc_row(sel, fam, arm, **extra):
    n = len(sel)
    ok = [r["decision"] == "alt" and bool(r.get("judge") and r["judge"]["ok"]) for r in sel]
    a = float(np.mean(ok)) if n else float("nan"); lo, hi = wilson(a, n) if n else (float("nan"),) * 2
    return dict(family=fam, arm=arm, n=n, accuracy=round(a, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                style_only=round(float(np.mean([r["decision"] == "alt" for r in sel])), 3) if n else float("nan"),
                unscorable=round(float(np.mean([r["decision"] is None for r in sel])), 3) if n else float("nan"),
                judge_ok=round(float(np.mean([bool(r.get("judge") and r["judge"]["ok"]) for r in sel])), 3) if n else float("nan"), **extra)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.load(open(ROOT / "selection.json"))
    L, lam, c = sel["layer"], sel["lambda"], np.array(sel["c"])
    best4 = {r["family"]: r for r in csv.DictReader(open(BEST4)) if r["target"] == "alt"}
    fams = [f.name for f in FAMILIES if (ROOT / "eval" / f"{f.name}.json").exists()]
    rows, npz = [], {}
    for fam in fams:
        recs = json.load(open(ROOT / "eval" / f"{fam}.json"))
        judged = [r for r in recs if r.get("judge")]
        test_ids = {r["doc_id"] for r in recs}
        for arm in ARMS:
            s = [r for r in judged if r["arm"] == arm]
            if s:
                rows.append(acc_row(s, fam, arm, layer=L, lam=lam, n_heads=s[0]["n_heads"], cf_family=s[0].get("cf_family")))
        # references on the same 80 texts
        b4 = best4[fam]; conf = json.load(open(CONFIRM / f"{fam}.json"))
        top = [r for r in conf if r["arm"].startswith("alt_top") and r["layer"] == int(b4["layer"]) and r["alpha"] == float(b4["alpha"]) and r["doc_id"] in test_ids and r.get("judge")]
        rows.append(acc_row(top, fam, "resid4", layer=int(b4["layer"]), lam=None, n_heads=None, cf_family=None, alpha=float(b4["alpha"])))
        base4 = [r for r in conf if r["arm"] == "base" and r["doc_id"] in test_ids and r.get("judge")]
        rows.append(acc_row(base4, fam, "base_step4", layer=None, lam=None, n_heads=None, cf_family=None))
        k4 = [r for r in json.load(open(ROLL / f"{fam}.json")) if r["style"] == "alt" and r["k"] == 4 and r["doc_id"] in test_ids and r.get("judge")]
        rows.append(acc_row(k4, fam, "icl_k4", layer=None, lam=None, n_heads=None, cf_family=None))
        npz[f"{fam}__arm"] = np.array([r["arm"] for r in judged]); npz[f"{fam}__decision"] = np.array([{"nat": 1, "alt": 0, None: -1}[r["decision"]] for r in judged])
        npz[f"{fam}__judge_ok"] = np.array([r["judge"]["ok"] for r in judged])
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("family", "arm", "n", "accuracy"), k))
    with open(OUT / "sparse_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    shutil.copy(ROOT / "layer_lambda.csv", OUT / "layer_lambda.csv")
    np.savez_compressed(OUT / "records.npz", **npz, c=c)
    R = {(r["family"], r["arm"]): r for r in rows}

    # ---- headline figure ------------------------------------------------------------------------
    arms_plot = ["base", "sparse_w", "sparse_unw", "sparse_cf", "resid4"]
    ncol = 4; nrow = math.ceil(len(fams) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.0 * nrow), sharey=True); axes = axes.ravel()
    for ax, fam in zip(axes, fams):
        for j, arm in enumerate(arms_plot):
            r = R.get((fam, arm))
            if r is None or math.isnan(r["accuracy"]):
                continue
            ax.bar(j, r["accuracy"], color=COL[arm], edgecolor="black" if arm == "sparse_w" else "none", linewidth=0.8,
                   yerr=[[r["accuracy"] - r["ci_lo"]], [r["ci_hi"] - r["accuracy"]]], capsize=2)
        k4 = R[(fam, "icl_k4")]["accuracy"]
        ax.axhline(k4, color="black", linestyle="dashed", lw=1)
        ax.set_xticks(range(len(arms_plot))); ax.set_xticklabels(["unsteered", f"sparse {sel['heads_02'].__len__()}h", "unweighted", "other family", "residual vec"], rotation=60, ha="right", fontsize=6.5)
        ax.set_title(fam, fontsize=10); ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(fams):]:
        ax.axis("off")
    for ax in axes[::ncol]:
        ax.set_ylabel("accuracy", fontsize=9)
    handles = [Patch(color=COL[a], label=LABEL[a]) for a in arms_plot] + [Line2D([], [], color="black", linestyle="dashed", label="4 in-context examples, no steering")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"Steering the rare convention with a sparse set of {len(sel['heads_02'])} attention-head means at the cue token (layer {L}, no in-context examples)\n"
                 "accuracy = target convention used AND faithful, coherent translation; 80 held-out texts per bar, 95% CI", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(OUT / "sparse_summary.png", dpi=150); plt.close(fig)

    # ---- heads vs accuracy / NLL ------------------------------------------------------------------
    ll = list(csv.DictReader(open(ROOT / "layer_lambda.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for Lr in sorted({int(r["layer"]) for r in ll}):
        pts = sorted([(int(r["n_heads_02"]), float(r["es_nll"]), float(r["lam"])) for r in ll if int(r["layer"]) == Lr])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"layer {Lr}" + (" (selected)" if Lr == L else ""), lw=2 if Lr == L else 1)
        for n, nll, lm in pts:
            ax.annotate(f"λ={lm:g}", (n, nll), fontsize=6, xytext=(3, 3), textcoords="offset points")
    c0 = float(ll[0]["es_nll_c0"]); ax.axhline(c0, color="grey", linestyle="dashed", lw=1, label="no steering")
    ax.set_xlabel("number of heads with c > 0.2"); ax.set_ylabel("validation NLL per label token"); ax.set_title("Fit quality vs sparsity (validation slice)", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    lam_dir = ROOT / "eval_lambda"
    if lam_dir.exists() and any(lam_dir.glob("*.json")):
        curve = {}
        for fam in fams:
            f = lam_dir / f"{fam}.json"
            if not f.exists():
                continue
            for r in json.load(open(f)):
                if r.get("judge"):
                    curve.setdefault((r["lam"], r["n_heads"]), []).append(r["decision"] == "alt" and r["judge"]["ok"])
        pts = sorted((n, float(np.mean(v)), lm) for (lm, n), v in curve.items())
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=COL["sparse_w"])
        for n, a, lm in pts:
            ax.annotate(f"λ={lm:g}", (n, a), fontsize=6, xytext=(3, 3), textcoords="offset points")
        base_mean = np.mean([R[(fam, "base")]["accuracy"] for fam in fams]); ax.axhline(base_mean, color="grey", linestyle="dashed", lw=1, label="no steering")
        ax.axhline(np.mean([R[(fam, "resid4")]["accuracy"] for fam in fams]), color=COL["resid4"], linestyle="dotted", lw=1, label="residual vector (step 4)")
        ax.set_xlabel("number of heads with c > 0.2"); ax.set_ylabel("mean accuracy over families (80 texts each)"); ax.set_title(f"Graded accuracy vs sparsity at layer {L}", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "lambda curve not evaluated", ha="center")
    fig.tight_layout(); fig.savefig(OUT / "heads_vs_accuracy.png", dpi=150); plt.close(fig)

    # ---- selected heads map ------------------------------------------------------------------------
    M = c.reshape(28, 16)
    fig, ax = plt.subplots(figsize=(6.5, 8))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    for i in sel.get("fv37_overlap_02", []):
        pass
    fv = json.load(open(ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43" / "pooled_sparse" / "selection.json"))["selected_heads"] if (ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43" / "pooled_sparse" / "selection.json").exists() else []
    for l, h, _ in fv:
        ax.add_patch(Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False, edgecolor="#1f77b4", lw=1.2))
    ax.set_xlabel("head"); ax.set_ylabel("layer (0-based block)"); ax.set_xticks(range(16)); ax.set_yticks(range(28))
    ax.set_title(f"Learned head coefficients c (layer {L}, λ = {lam:g}); {len(sel['heads_02'])} heads with c > 0.2\n"
                 f"blue outline = the 37 in-context-learning function-vector heads ({len(sel.get('fv37_overlap_02', []))} shared)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, label="coefficient")
    fig.tight_layout(); fig.savefig(OUT / "selected_heads_map.png", dpi=150); plt.close(fig)

    print(f"{'family':14s} {'base':>5s} {'sparse':>6s} {'unw':>5s} {'cf':>5s} {'resid4':>6s} {'k4':>5s} | style-only unsc judge (sparse)")
    for fam in fams:
        g = lambda a: R[(fam, a)]["accuracy"] if (fam, a) in R else float("nan")
        s = R[(fam, "sparse_w")]
        print(f"{fam:14s} {g('base'):5.2f} {g('sparse_w'):6.2f} {g('sparse_unw'):5.2f} {g('sparse_cf'):5.2f} {g('resid4'):6.2f} {g('icl_k4'):5.2f} | {s['style_only']:.2f} {s['unscorable']:.2f} {s['judge_ok']:.2f}")
    print("->", OUT)


if __name__ == "__main__":
    main()
