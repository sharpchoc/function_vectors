#!/usr/bin/env python
"""k = 8 vs k = 4 for the code families (user request 2026-09-16): the k = 8 point exists only for documents with ≥ 9 opportunities, so the
fair comparison is PAIRED — k = 4 accuracy on the same documents vs k = 8 accuracy. Reads the judged rollouts (all k), writes
<results>/code/k8/k8_vs_k4.csv and k8_vs_k4.png (two panels nat / alt, families sorted by paired delta, n < 50 greyed) and prints the verdict."""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.analyze import wilson

C = {"nat": "#1f77b4", "alt": "#d62728"}


def acc(recs):
    n = len(recs); return (sum(r["style_ok"] and r["judge"]["ok"] for r in recs) / n if n else float("nan")), n


def main():
    MP = model_paths("qwen25_base"); R = MP["results"] / "code"; OUT = R / "k8"; OUT.mkdir(exist_ok=True)
    pool = json.load(open(MP["results"] / "code_pool_full.json"))["pool"]
    rows = []
    for f in pool:
        recs = [r for r in json.load(open(MP["rollouts"] / f"{f}.json")) if r.get("judge")]
        for s in ("nat", "alt"):
            k8 = [r for r in recs if r["style"] == s and r["k"] == 8]; docs8 = {r["doc_id"] for r in k8}
            k4 = [r for r in recs if r["style"] == s and r["k"] == 4]; k4p = [r for r in k4 if r["doc_id"] in docs8]
            a8, n8 = acc(k8); a4, n4 = acc(k4); a4p, n4p = acc(k4p)
            lo8, hi8 = wilson(a8, n8) if n8 else (np.nan, np.nan); lo4, hi4 = wilson(a4p, n4p) if n4p else (np.nan, np.nan)
            # paired: per-document correctness difference (docs in both)
            c4 = {r["doc_id"]: (r["style_ok"] and r["judge"]["ok"]) for r in k4p}; c8 = {r["doc_id"]: (r["style_ok"] and r["judge"]["ok"]) for r in k8}
            both = sorted(set(c4) & set(c8)); d = np.array([int(c8[x]) - int(c4[x]) for x in both]) if both else np.array([])
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            rows.append(dict(family=f, style=s, n_k8=n8, n_k4_all=n4, acc_k4_all=round(a4, 3), acc_k4_same_docs=round(a4p, 3), acc_k8=round(a8, 3),
                             ci_lo_k4=round(lo4, 3), ci_hi_k4=round(hi4, 3), ci_lo_k8=round(lo8, 3), ci_hi_k8=round(hi8, 3),
                             delta=round(float(d.mean()) if len(d) else np.nan, 3), delta_se=round(float(se), 3) if se == se else np.nan,
                             sig=bool(len(d) > 1 and abs(d.mean()) > 1.96 * se),
                             capped_k8=round(float(np.mean([r["capped"] for r in k8])), 3) if n8 else np.nan,
                             unscorable_k8=round(float(np.mean([r["decision"] is None for r in k8])), 3) if n8 else np.nan,
                             unscorable_k4_same_docs=round(float(np.mean([r["decision"] is None for r in k4p])), 3) if n4p else np.nan,
                             judge_ok_k8=round(float(np.mean([r["judge"]["ok"] for r in k8])), 3) if n8 else np.nan))
    with open(OUT / "k8_vs_k4.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(19, 7.5))
    for ax, s in zip(axes, ("nat", "alt")):
        sub = sorted([r for r in rows if r["style"] == s and r["n_k8"] > 0], key=lambda r: -r["delta"])
        y = np.arange(len(sub))
        for i, r in enumerate(sub):
            col = C[s] if r["n_k8"] >= 50 else "#aaaaaa"
            ax.plot([r["acc_k4_same_docs"], r["acc_k8"]], [i, i], color="#cccccc", lw=2, zorder=1)
            ax.plot(r["acc_k4_same_docs"], i, "o", color="#9e9e9e", ms=6, zorder=2)
            ax.errorbar(r["acc_k8"], i, xerr=[[r["acc_k8"] - r["ci_lo_k8"]], [r["ci_hi_k8"] - r["acc_k8"]]], fmt="o", color=col, ms=6, capsize=2, lw=1, zorder=3)
            ax.text(1.02, i, f"n={r['n_k8']}  Δ={r['delta']:+.2f}{'*' if r['sig'] else ''}", va="center", fontsize=6.5, color="#555555")
        ax.set_yticks(y); ax.set_yticklabels([r["family"] for r in sub], fontsize=7.5); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-1, len(sub))
        ax.set_xlabel("accuracy (convention used AND correct solution)"); ax.grid(axis="x", alpha=0.3); ax.axvline(0.5, color="#dddddd", lw=0.8)
        m4 = np.mean([r["acc_k4_same_docs"] for r in sub if r["n_k8"] >= 50]); m8 = np.mean([r["acc_k8"] for r in sub if r["n_k8"] >= 50])
        ax.set_title(f"{s} convention in context — mean over families with n ≥ 50: k = 4 {m4:.2f} → k = 8 {m8:.2f}", fontsize=10, color=C[s])
    handles = [Line2D([], [], marker="o", color="#9e9e9e", ls="", ms=6, label="k = 4 accuracy on the same documents"),
               Line2D([], [], marker="o", color="#444444", ls="", ms=6, label="k = 8 accuracy (95% CI); grey = fewer than 50 documents")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, fontsize=9.5, frameon=False)
    fig.suptitle("Code conventions on Qwen2.5-7B base: 8 in-context examples vs 4, paired on the documents that have ≥ 9 opportunities\n"
                 "row label = family; Δ = paired mean difference k8 − k4 (* = |Δ| > 1.96 SE); sorted by Δ", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=4); fig.savefig(OUT / "k8_vs_k4.png", dpi=150); plt.close(fig)
    for s in ("nat", "alt"):
        sub = [r for r in rows if r["style"] == s and r["n_k8"] >= 50]
        print(f"{s}: families n≥50 {len(sub)} | k4 (all docs) {np.mean([r['acc_k4_all'] for r in sub]):.3f} | k4 (same docs) {np.mean([r['acc_k4_same_docs'] for r in sub]):.3f} → k8 {np.mean([r['acc_k8'] for r in sub]):.3f} | "
              f"mean paired Δ {np.mean([r['delta'] for r in sub]):+.3f} | sig gain {sum(r['sig'] and r['delta'] > 0 for r in sub)} / sig loss {sum(r['sig'] and r['delta'] < 0 for r in sub)} | "
              f"capped k8 {np.mean([r['capped_k8'] for r in sub]):.2f} | unscorable k8 {np.mean([r['unscorable_k8'] for r in sub]):.2f} vs k4 same docs {np.mean([r['unscorable_k4_same_docs'] for r in sub]):.2f}")
    print("->", OUT)


if __name__ == "__main__":
    main()
