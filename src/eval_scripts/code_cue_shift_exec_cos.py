#!/usr/bin/env python
"""Coding cue shift re-expressed as cosine with the execution feature, from STORED activations (CPU; 2026-09-26).

Paper section 5.3 metric: cos(dz_alpha, -d_f^exec), dz_alpha = h_alpha - h_0 = steering-induced shift of the query-cue residual at
hidden state 24 (block 23 output); d_f^exec = v_nat = mean_nat - mean_alt of the cue-token activation at hidden state 24 over paired
successful training-split prompts (capture_cues.py -> steering/vectors_k3_train/<family>.npz, row 23). Steering adds -alpha*r_id, i.e.
towards the alternative convention B, so the expected sign is positive against -d_f^exec.

This is exactly the stored `cos_w` of read_causal.py (w = unit(-v_nat) at hidden state 24, computed from fp32 activations during the
run), and `cf_cos_w` is the same cosine for the real shift h_cf - h_0. The script
  1. validates: rebuilds summarize_code_read_causal_v2.py's per-family table and bootstrap (same RNG stream, seed 20260923) from the
     npz files and checks it against results/code_styles/read_causal_v2/{per_family,summary}.csv and the README headline numbers;
  2. cross-checks cos_w against a recomputation from the stored fp16 L24 activations (h_L24, h0_L24) and the execution vectors;
  3. writes exec_feature_cos_{summary,per_family}.csv and a candidate two-panel figure exec_feature_cos_by_alpha.{png,pdf}
     (left: cos with -d_f^exec by alpha; right: unchanged execution projection share) in the style of paper Figure 21.
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.paths import ARTIFACTS_ROOT, CODE_STYLES_DIR

SRC = ARTIFACTS_ROOT / "style_translation" / "qwen25_code" / "read_causal_v2"
VEC = ARTIFACTS_ROOT / "style_translation" / "qwen25_code" / "steering" / "vectors_k3_train"
OUT = CODE_STYLES_DIR / "read_causal_v2"
HS = 24                                         # hidden-state index of the readout (block 23 output)
METRICS = ("dircos", "proj_frac", "cos_w", "norm_ratio", "alt_top_steered", "margin_steered")
REFS = ("cf_cos_w", "alt_top_unsteered", "alt_top_cf", "margin_unsteered", "margin_cf")


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def per_family(pool):
    rows, check = [], []
    for fam in pool:
        z = np.load(SRC / f"{fam}.npz", allow_pickle=True)
        li = list(z["layers"]).index(HS)
        m0 = z["lp0"][:, 1] - z["lp0"][:, 0]; mcf = z["lpcf"][:, 1] - z["lpcf"][:, 0]
        # recomputation from the stored fp16 cue vectors (cross-check only)
        d_exec = np.load(VEC / f"{fam}.npz")["v_nat"][HS - 1].astype(np.float32)        # mean_nat - mean_alt = d_f^exec (A - B)
        neg = unit(-d_exec)
        h0 = z["h0_L24"].astype(np.float32); hcf = z["hcf_L24"].astype(np.float32); H = z["h_L24"].astype(np.float32)
        cf_re = unit(hcf - h0) @ neg
        for ai, arm in enumerate(z["arms"]):
            name, alpha = str(arm).split("_a")[0], float(str(arm).split("_a")[1])
            ma = z["lp"][:, ai, 1] - z["lp"][:, ai, 0]
            re = unit(H[:, ai] - h0) @ neg
            check.append(np.abs(re - z["cos_w"][:, ai, li]).max())
            rows.append(dict(family=fam, arm=name, alpha=alpha, ctrl_family=str(z["ctrl_family"]),
                             dircos=float(np.nanmean(z["dircos"][:, ai, li])), proj_frac=float(np.nanmedian(z["proj_frac"][:, ai, li])),
                             cos_w=float(np.nanmean(z["cos_w"][:, ai, li])), norm_ratio=float(np.nanmean(z["norm_ratio"][:, ai, li])),
                             cf_cos_w=float(np.nanmean(z["cf_cos_w"][:, li])),
                             alt_top_steered=float((ma > 0).mean()), alt_top_unsteered=float((m0 > 0).mean()), alt_top_cf=float((mcf > 0).mean()),
                             margin_steered=float(ma.mean()), margin_unsteered=float(m0.mean()), margin_cf=float(mcf.mean()),
                             exec_cos_fp16=float(re.mean()), cf_exec_cos_fp16=float(cf_re.mean())))
        check.append(np.abs(cf_re - z["cf_cos_w"][:, li]).max())
    return pd.DataFrame(rows), float(max(check))


def bootstrap(d):
    """Identical RNG stream and loop order to summarize_code_read_causal_v2.py (so CIs match summary.csv exactly)."""
    rng = np.random.default_rng(20260923); summ = []
    for (arm, alpha), g in d.groupby(["arm", "alpha"]):
        g = g.sort_values("family")
        for m in METRICS:
            x = g[m].to_numpy(); b = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
            summ.append(dict(arm=arm, alpha=alpha, metric=m, mean=x.mean(), lo=np.quantile(b, .025), hi=np.quantile(b, .975), n_families=len(x)))
    ref = d[(d.arm == "read") & (d.alpha == 2.0)]
    for m in REFS:
        x = ref[m].to_numpy(); b = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
        summ.append(dict(arm="reference", alpha=np.nan, metric=m, mean=x.mean(), lo=np.quantile(b, .025), hi=np.quantile(b, .975), n_families=len(x)))
    return pd.DataFrame(summ)


def counts(d, metric, alpha):
    w = d[d.alpha == alpha].pivot_table(index="family", columns="arm", values=metric)
    return dict(own_gt_both=int(((w.read > w.ctrl_family) & (w.read > w.ctrl_random)).sum()), own_ge_05=int((w.read >= .5).sum()),
                own_neg=int((w.read < 0).sum()), n=len(w))


def main():
    pool = sorted(json.load(open(CODE_STYLES_DIR / "code_pool.json"))["pool"]); assert len(pool) == 53
    d, maxdiff = per_family(pool)
    s = bootstrap(d)
    # ---- validation against the published v2 outputs
    pub_pf = pd.read_csv(OUT / "per_family.csv"); pub_s = pd.read_csv(OUT / "summary.csv")
    cols = [c for c in pub_pf.columns if c not in ("family", "arm", "ctrl_family")]
    a = d.sort_values(["family", "arm", "alpha"]).reset_index(drop=True); b = pub_pf.sort_values(["family", "arm", "alpha"]).reset_index(drop=True)
    assert (a[["family", "arm"]].values == b[["family", "arm"]].values).all() and np.allclose(a[cols], b[cols], atol=1e-6, equal_nan=True), "per_family mismatch"
    assert np.allclose(s[["mean", "lo", "hi"]].round(4), pub_s[["mean", "lo", "hi"]], atol=1.01e-4, equal_nan=True), "summary mismatch"
    g = lambda arm, m, al=2.0: s[(s.arm == arm) & (s.metric == m) & ((s.alpha == al) if arm != "reference" else True)].iloc[0]
    c_dir = counts(d, "dircos", 2.0)
    print(f"VALIDATION (dircos, alpha 2): own {g('read','dircos')['mean']:.3f} [{g('read','dircos')['lo']:.3f},{g('read','dircos')['hi']:.3f}] "
          f"other {g('ctrl_family','dircos')['mean']:.3f} random {g('ctrl_random','dircos')['mean']:.3f} | real-shift cos_w {g('reference','cf_cos_w')['mean']:.3f} | "
          f"own>both {c_dir['own_gt_both']}/53, own>=.5 {c_dir['own_ge_05']}/53")
    assert (round(g('read', 'dircos')['mean'], 3), round(g('ctrl_family', 'dircos')['mean'], 3), round(g('ctrl_random', 'dircos')['mean'], 3),
            round(g('reference', 'cf_cos_w')['mean'], 3), c_dir['own_gt_both'], c_dir['own_ge_05']) == (.746, .173, .039, .721, 52, 46)
    print(f"cross-check: max |cos_w (fp32, run) - recomputed from fp16 L24 vectors| = {maxdiff:.2e}")
    # ---- new metric outputs
    OUT.mkdir(parents=True, exist_ok=True)
    e = s[s.metric == "cos_w"].drop(columns="metric").copy()
    rf = s[(s.arm == "reference") & (s.metric == "cf_cos_w")].drop(columns="metric").assign(arm="real_shift")
    e = pd.concat([e, rf]).rename(columns={"mean": "exec_cos_mean", "lo": "ci_lo", "hi": "ci_hi"})
    cnt = []
    for al in sorted(d.alpha.unique()):
        c = counts(d, "cos_w", al); cnt.append(dict(alpha=al, own_gt_both_controls=c["own_gt_both"], own_ge_0p5=c["own_ge_05"], own_negative=c["own_neg"]))
    e = e.merge(pd.DataFrame(cnt).assign(arm="read"), on=["arm", "alpha"], how="left")   # counts refer to the own-family arm
    real = d[(d.arm == "read") & (d.alpha == 2.0)].set_index("family").cf_cos_w
    e.loc[e.arm == "real_shift", ["own_ge_0p5", "own_negative"]] = [int((real >= .5).sum()), int((real < 0).sum())]
    e.round(4).to_csv(OUT / "exec_feature_cos_summary.csv", index=False)
    pf = d[["family", "arm", "alpha", "ctrl_family", "cos_w", "cf_cos_w", "exec_cos_fp16", "cf_exec_cos_fp16", "dircos"]].rename(
        columns={"cos_w": "exec_cos", "cf_cos_w": "real_shift_exec_cos"})
    pf.round(4).to_csv(OUT / "exec_feature_cos_per_family.csv", index=False)
    print(e.round(3).to_string(index=False))
    w2 = d[d.alpha == 2.0].pivot_table(index="family", columns="arm", values="cos_w")
    print("alpha 2 lowest own exec-cos families:\n" + w2.sort_values("read").head(8).round(3).to_string())
    # ---- candidate Figure 21 (style of paper_materials/plot_coding_causality.py; left panel metric swapped)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6})
    T, O, G = '#007C91', '#D98B27', '#707A84'
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.15))
    for arm, label, color in zip(['read', 'ctrl_family', 'ctrl_random'], ['Own family', 'Other family', 'Random'], [T, O, G]):
        for ax, metric in zip(axs, ['cos_w', 'proj_frac']):
            z = s[(s.arm == arm) & (s.metric == metric)].sort_values('alpha'); xx = np.arange(len(z))
            ax.plot(xx, z['mean'], c=color, label=label, marker='o', ms=2.5, lw=1.3); ax.fill_between(xx, z.lo, z.hi, color=color, alpha=.14, lw=0)
            ax.set_xticks(xx, [f'{a:g}' for a in z.alpha]); ax.set_xlabel(r'Steering strength $\alpha$'); ax.grid(axis='y', alpha=.15)
    axs[0].set_ylabel(r'Cosine with $-d_f^{\mathrm{exec}}$'); axs[1].set_ylabel('Execution projection share'); axs[0].set_ylim(-.05, 1); axs[1].axhline(1, c=G, ls=':', lw=.7)
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False, fontsize=8); fig.subplots_adjust(left=.11, right=.99, bottom=.25, top=.80, wspace=.38)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f"exec_feature_cos_by_alpha.{ext}", dpi=300, bbox_inches='tight', pad_inches=.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
