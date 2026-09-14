#!/usr/bin/env python
"""Phase 2.3 (plan 2026-09-14): coverage diagnostics for the read→write map BEFORE fitting anything.
For every family in the pool (per-prompt captures in <model>/prompt_pairs): convention direction = nat centroid − alt centroid
of the read feature (layer --read_layer) and of the write feature (layer --write_layer).
  span_centroids   fraction of the family's direction inside the span of the OTHER families' centroid directions
  span_pcs_k       fraction inside the top-k principal components of the other families' centred prompts (k = 11, 50, 200)
  nearest          the other family with the largest |cos| (read and write) — the axis clusters
  rsa              correlation between the read and write pairwise-cosine matrices over all family pairs
  participation    participation ratio of the unit read / write directions (effective dimensionality of the code)
Outputs → <model results>/read_write_map/coverage.{csv,png}."""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths


def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def span_frac(v, basis):
    Q, _ = np.linalg.qr(np.array(basis).T); return float(np.linalg.norm(Q.T @ v) / np.linalg.norm(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_base"); ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--read_layer", type=int, default=12); ap.add_argument("--write_layer", type=int, default=24)
    args = ap.parse_args()
    MP = model_paths(args.model); PP = MP["prompt_pairs"]; OUT = MP["results"] / "read_write_map"; OUT.mkdir(parents=True, exist_ok=True)
    X, Y, F, P = [], [], [], []
    for f in args.families:
        d = np.load(PP / f"{f}.npz"); X.append(d[f"read_L{args.read_layer}"].astype(np.float64)); Y.append(d[f"write_L{args.write_layer}"].astype(np.float64))
        F += [f] * len(d["pole"]); P += list(d["pole"])
    X, Y, F, P = np.vstack(X), np.vstack(Y), np.array(F), np.array(P)
    D = {f: (X[(F == f) & (P == "nat")].mean(0) - X[(F == f) & (P == "alt")].mean(0), Y[(F == f) & (P == "nat")].mean(0) - Y[(F == f) & (P == "alt")].mean(0)) for f in args.families}
    rows = []
    for f in args.families:
        others = [g for g in args.families if g != f]
        m = F != f
        pcs = {}
        for side, M, i in (("read", X, 0), ("write", Y, 1)):
            Mc = M[m] - M[m].mean(0); _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
            for k in (11, 50, 200):
                pcs[f"{side}_pcs_{k}"] = float(np.linalg.norm(Vt[:k] @ D[f][i]) / np.linalg.norm(D[f][i]))
        near_r = max(others, key=lambda g: abs(cos(D[f][0], D[g][0]))); near_w = max(others, key=lambda g: abs(cos(D[f][1], D[g][1])))
        rows.append(dict(family=f, read_span=span_frac(D[f][0], [D[g][0] for g in others]), write_span=span_frac(D[f][1], [D[g][1] for g in others]),
                         **pcs, nearest_read=near_r, cos_read=cos(D[f][0], D[near_r][0]), nearest_write=near_w, cos_write=cos(D[f][1], D[near_w][1])))
    fams = args.families; n = len(fams)
    Cr = np.array([[cos(D[a][0], D[b][0]) for b in fams] for a in fams]); Cw = np.array([[cos(D[a][1], D[b][1]) for b in fams] for a in fams])
    iu = np.triu_indices(n, 1); rsa = float(np.corrcoef(Cr[iu], Cw[iu])[0, 1])
    def pr(M):
        U = np.array([M[f] / np.linalg.norm(M[f]) for f in fams]); s = np.linalg.svd(U, compute_uv=False); return float(s.sum() ** 2 / (s ** 2).sum())
    pr_r, pr_w = pr({f: D[f][0] for f in fams}), pr({f: D[f][1] for f in fams})
    with open(OUT / "coverage.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        fh.write(f"# rsa_corr={rsa:.3f} participation_read={pr_r:.1f}/{n} participation_write={pr_w:.1f}/{n} read_layer={args.read_layer} write_layer={args.write_layer}\n")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    x = np.arange(n); w = 0.38
    ax[0].bar(x - w / 2, [r["read_span"] for r in rows], w, label=f"read L{args.read_layer}"); ax[0].bar(x + w / 2, [r["write_span"] for r in rows], w, label=f"write L{args.write_layer}")
    ax[0].axhline((max(n - 1, 1) / X.shape[1]) ** .5, color="grey", ls="--", lw=1, label="chance (random direction)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(fams, rotation=45, ha="right", fontsize=8); ax[0].set_ylim(0, 1); ax[0].set_ylabel("fraction of the convention direction inside the span of the other families")
    ax[0].set_title("Coverage by the other families' centroid directions"); ax[0].legend(fontsize=8)
    for i, (C, lab) in enumerate(((Cr, f"read L{args.read_layer}"), (Cw, f"write L{args.write_layer}")), 1):
        im = ax[i].imshow(C, vmin=-1, vmax=1, cmap="RdBu_r"); ax[i].set_xticks(x); ax[i].set_yticks(x); ax[i].set_xticklabels(fams, rotation=90, fontsize=7); ax[i].set_yticklabels(fams, fontsize=7)
        ax[i].set_title(f"pairwise cos of convention directions — {lab}"); plt.colorbar(im, ax=ax[i], fraction=0.046)
    fig.suptitle(f"{MP['label']}: coverage diagnostics for the read→write map ({n} families). RSA corr(read cos, write cos) = {rsa:.2f}; "
                 f"participation ratio read {pr_r:.1f} / write {pr_w:.1f} of {n}", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "coverage.png", dpi=150)
    print(f"{'family':14s} {'read span':>9s} {'write span':>10s} {'read pcs11/50/200':>18s} {'write pcs11/50/200':>19s}  nearest (read cos) / (write cos)")
    for r in rows:
        print(f"{r['family']:14s} {r['read_span']:9.2f} {r['write_span']:10.2f} {r['read_pcs_11']:5.2f}/{r['read_pcs_50']:.2f}/{r['read_pcs_200']:.2f} "
              f"{r['write_pcs_11']:6.2f}/{r['write_pcs_50']:.2f}/{r['write_pcs_200']:.2f}  {r['nearest_read']} ({r['cos_read']:+.2f}) / {r['nearest_write']} ({r['cos_write']:+.2f})")
    print(f"RSA corr(read cos, write cos) = {rsa:.2f}; participation ratio read {pr_r:.1f}, write {pr_w:.1f} of {n}")


if __name__ == "__main__":
    main()
