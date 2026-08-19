#!/usr/bin/env python
"""Stage-2 of the two-shot pair-diff pre-image study (Stream S). CPU-only.

For each task pair's two-shot matched-label captures (Stream K), build per-pair difference
vectors d_i = act_f1 - act_f2 at every (token role, layer), and measure how much of their
variance is explained by the 1-D FV-PRE-IMAGE-DIFFERENCE direction u from stage 1
(fit_ridge_preimages_multicell.py): with c_i = <d_i, u>,

    centered   explained = sum((c_i - cbar)^2) / sum(||d_i - dbar||^2)   (headline)
    uncentered explained = sum(c_i^2) / sum(||d_i||^2)                    (secondary)

i.e. explained = 1 - var(d - proj_u d) / var(d) for the corresponding var definition.

Directions compared per cell:
    damped        - Tikhonov-damped pre-image of fv_A - fv_B (headline)
    exact         - undamped pre-image (expected ~noise: cond(W) ~ 1e9)
    fv_diff       - unit(fv_A - fv_B) in raw activation space (no inversion; same at every layer)
    top_pc        - top principal direction of {d_i} (the 1-D upper bound; per centering + filter)
    random        - mean +/- sd over --n_random isotropic random unit vectors (floor ~ 1/4096)
    random_actcov - mean +/- sd over --n_random unit vectors sampled from the anisotropic
                    Gaussian N(0, Sigma_hat), where Sigma_hat is the empirical covariance of the
                    RAW (undiffed) two-shot activations of both functions at that (role, layer):
                    v = unit(Xc^T g), g ~ N(0, I). A task-agnostic but residual-stream-geometry-
                    aware chance level (the residual stream is very anisotropic, so this sits
                    well above the isotropic floor).

Role -> ridge-cell mapping (context-matched; two-shot layer index j == bank edit_layer j):
    demo1_prelabel -> pre_label_token icl1      demo1_label -> last_label_token icl1
    demo2_prelabel -> pre_label_token icl2      demo2_label -> last_label_token icl2
    query_final    -> pre_label_token icl3 (primary) + last_prompt_token icl10 (secondary view)

Loader sanity gate: reproduces the Stream K mean pairwise-cosine grid
(results/direction2_label_geometry/twoshot/diffcos_heatmap/meancos_grid.json) from the
freshly built diff vectors before computing anything else.

Outputs per pair (TRACKED): explained_grid.json, heatmap PNGs (view x layer) per
(direction x centering x filter), and per-view line plots.

Secondary metric (cos_grid.json, plotted by plot_twoshot_pairdiff_cos_lines.py): SIGNED
mean_i cos(d_i, x) per cell for x in {exact pre-image of fv_A - fv_B ("inv_fv_diff"),
unit(fv_A - fv_B) ("fv_diff")}, with the analytic maximum mean_dir = ||mean_i unit(d_i)||
and the same isotropic + activation-covariance random baselines (signed expectation ~0;
compare against their sd bands).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    load_function_vector,
    load_json,
    torch_load_trusted,
    write_json,
)
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR, LABEL_GEOMETRY_DIR

PAIR_DIRS = {
    "antonym_synonym": ("antonym", "synonym"),
    "next_number_digits_prev_number_digits": ("next_number_digits", "prev_number_digits"),
}
# view name -> (two-shot role, stage-1 cell dir name)
ROLE_VIEWS = [
    ("demo1_prelabel", "demo1_prelabel", "pre_label_token_icl1"),
    ("demo1_label", "demo1_label", "last_label_token_icl1"),
    ("demo2_prelabel", "demo2_prelabel", "pre_label_token_icl2"),
    ("demo2_label", "demo2_label", "last_label_token_icl2"),
    ("query_final", "query_final", "pre_label_token_icl3"),
    ("query_final@lastprompt10", "query_final", "last_prompt_token_icl10"),
]
DIRECTIONS = ["damped", "exact", "fv_diff", "top_pc", "random", "random_actcov"]
CENTERINGS = ["centered", "uncentered"]
FILTERS = ["all", "both_correct"]
# Secondary metric: SIGNED mean_i cos(d_i, x) per cell -> cos_grid.json (mean_dir = analytic max)
# inv_fv_diff_pcak16 = pre-image through the k=16 PCA ridge (fit_pca_ridge_preimages_multicell)
# inv_fv_diff_tsvdk16 = rank-16 TSVD pre-image of the full-dim maps (fit_tsvd_preimages_multicell)
COS_KEYS = ["inv_fv_diff", "inv_fv_diff_pcak16", "inv_fv_diff_tsvdk16", "fv_diff", "mean_dir",
            "random", "random_sd", "random_actcov", "random_actcov_sd"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", nargs="+", default=list(PAIR_DIRS))
    p.add_argument("--twoshot_root", type=Path, default=ARTIFACTS_ROOT / "twoshot_paired_graded")
    p.add_argument("--preimage_root", type=Path,
                   default=ARTIFACTS_ROOT / "preimage_pairdiff/train_varicl_max4_top40")
    p.add_argument("--pca_preimage_root", type=Path,
                   default=ARTIFACTS_ROOT / "preimage_pairdiff_pcak16/train_varicl_max4_top40",
                   help="PCA-k16 ridge pre-image banks (optional; cos metric only).")
    p.add_argument("--tsvd_preimage_root", type=Path,
                   default=ARTIFACTS_ROOT / "preimage_pairdiff_tsvdk16/train_varicl_max4_top40",
                   help="Rank-16 TSVD pre-image banks (optional; cos metric only).")
    p.add_argument("--output_root", type=Path, default=None,
                   help="Default: results/direction3_fv_formation/twoshot_pairdiff_fv_preimage/"
                        "<preimage_root basename>.")
    p.add_argument("--meancos_reference", type=Path,
                   default=LABEL_GEOMETRY_DIR / "twoshot/diffcos_heatmap/meancos_grid.json")
    p.add_argument("--meancos_tol", type=float, default=2e-3)
    p.add_argument("--n_random", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_pair_diffs(pair_dir, f1, f2):
    """Return {role: (keys, D [n_pairs, 28, 4096] fp32, both_correct bool [n_pairs],
    X [2*n_pairs, 28, 4096] fp32 raw activations, f1 rows then f2 rows, key-aligned)."""
    index = load_json(pair_dir / "index.json")
    roles = index["config"]["roles"]
    per_role = {r: {} for r in roles}       # key -> {function: activation [28, 4096]}
    judge = {}                              # (function, key) -> judge_top1
    for shard in index["shards"]:
        shard_path = Path(shard)
        if not shard_path.exists():
            shard_path = pair_dir / shard_path.name
        data = torch_load_trusted(shard_path, map_location="cpu")
        acts, metas = data["activations"], data["metadata"]
        for i, m in enumerate(metas):
            key = (m["label1"], m["label2"], m["query_word"])
            per_role[m["role"]].setdefault(key, {})[m["function"]] = acts[i]
            judge[(m["function"], key)] = bool(m.get("judge_top1", False))
    out = {}
    for role in roles:
        keys = sorted(k for k, v in per_role[role].items() if "f1" in v and "f2" in v)
        A1 = torch.stack([per_role[role][k]["f1"] for k in keys]).float()
        A2 = torch.stack([per_role[role][k]["f2"] for k in keys]).float()
        D = A1 - A2
        both = torch.tensor([judge[("f1", k)] and judge[("f2", k)] for k in keys])
        out[role] = (keys, D, both, torch.cat([A1, A2]))
    (_, D0, b0, _) = next(iter(out.values()))
    print(f"  {f1} - {f2}: {D0.shape[0]} pairs x {D0.shape[1]} layers "
          f"(both-judge-correct: {int(b0.sum())})")
    return out


def mean_pairwise_cos(D_layer):
    """Mean upper-triangle pairwise cosine of unit-normalized rows: (||sum u||^2 - n)/(n(n-1))."""
    U = D_layer / D_layer.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n = U.shape[0]
    s = U.sum(dim=0)
    return float((s @ s - n) / (n * (n - 1)))


def sanity_check_meancos(diffs, pair_name, reference_path, tol):
    ref = load_json(reference_path).get(pair_name)
    if ref is None:
        print(f"  [sanity] no meancos reference for {pair_name}; skipping gate")
        return
    worst = 0.0
    for ri, role in enumerate(ref["roles"]):
        _, D, _, _ = diffs[role]
        for li, layer in enumerate(ref["layers"]):
            got = mean_pairwise_cos(D[:, layer, :])
            worst = max(worst, abs(got - ref["mean_cos_grid"][ri][li]))
    if worst > tol:
        raise AssertionError(f"[sanity] meancos grid mismatch for {pair_name}: "
                             f"max |diff| = {worst:.2e} > tol {tol}")
    print(f"  [sanity] meancos grid reproduced (max |diff| = {worst:.2e}) PASS")


def explained_ratios(D, u):
    """(centered, uncentered) explained-variance fractions of D [n, dim] along unit u."""
    c = D @ u
    unctr = float((c ** 2).sum() / (D ** 2).sum())
    cc = c - c.mean()
    Dc = D - D.mean(dim=0, keepdim=True)
    ctr = float((cc ** 2).sum() / (Dc ** 2).sum())
    return {"centered": ctr, "uncentered": unctr}


def top_pc_explained(D):
    """1-D upper bounds: top singular direction (uncentered) / top PC (centered)."""
    Dc = D - D.mean(dim=0, keepdim=True)
    s_c = torch.linalg.svdvals(Dc)
    s_u = torch.linalg.svdvals(D)
    return {"centered": float(s_c[0] ** 2 / (s_c ** 2).sum()),
            "uncentered": float(s_u[0] ** 2 / (s_u ** 2).sum())}


def unit(v):
    return v / v.norm().clamp_min(1e-12)


def row_unit(D):
    return D / D.norm(dim=1, keepdim=True).clamp_min(1e-12)


def mean_cos(Dn, x):
    """mean_i cos(d_i, x) for row-normalized Dn [n, dim] and unit x (SIGNED)."""
    return float((Dn @ x).mean())


def main():
    args = parse_args()
    out_base = args.output_root or (FV_FORMATION_DIR / "preimage_analysis/twoshot_pairdiff_fv_preimage"
                                    / args.preimage_root.name)
    stage1_cfg = load_json(args.preimage_root / "run_config.json")
    fv_root = Path(stage1_cfg["fv_root"])
    rng = torch.Generator().manual_seed(args.seed)
    # Separate stream for the anisotropic draws so the isotropic "random" values reproduce
    # the original run exactly.
    rng_cov = torch.Generator().manual_seed(args.seed + 1)

    for pair_name in args.pairs:
        f1, f2 = PAIR_DIRS[pair_name]
        print(f"== {pair_name}")
        out_dir = out_base / pair_name
        out_dir.mkdir(parents=True, exist_ok=True)
        diffs = load_pair_diffs(args.twoshot_root / pair_name, f1, f2)
        sanity_check_meancos(diffs, pair_name, args.meancos_reference, args.meancos_tol)

        u_fv = unit((load_function_vector(fv_root, f1) - load_function_vector(fv_root, f2)).float())

        grid = {}       # view -> filter -> centering -> direction -> [n_layers]
        cos_grid = {}   # view -> filter -> cos key -> [n_layers]
        n_layers = next(iter(diffs.values()))[1].shape[1]
        for view, role, cell in ROLE_VIEWS:
            bank_path = (args.preimage_root / cell / "pairdiff_preimages"
                         / f"{f1}__{f2}_pairdiff_preimage_bank.pt")
            if not bank_path.exists():
                print(f"  [skip] {view}: no bank at {bank_path}")
                continue
            bank = torch_load_trusted(bank_path, map_location="cpu")["preimages_by_edit_layer"]
            pca_bank_path = (args.pca_preimage_root / cell / "pairdiff_preimages"
                             / f"{f1}__{f2}_pairdiff_preimage_bank.pt")
            pca_bank = (torch_load_trusted(pca_bank_path, map_location="cpu")["preimages_by_edit_layer"]
                        if pca_bank_path.exists() else {})
            if not pca_bank:
                print(f"  [note] {view}: no PCA-k16 bank at {pca_bank_path}; line will be null")
            tsvd_bank_path = (args.tsvd_preimage_root / cell / "pairdiff_preimages"
                              / f"{f1}__{f2}_pairdiff_preimage_bank.pt")
            tsvd_bank = (torch_load_trusted(tsvd_bank_path, map_location="cpu")["preimages_by_edit_layer"]
                         if tsvd_bank_path.exists() else {})
            if not tsvd_bank:
                print(f"  [note] {view}: no TSVD-k16 bank at {tsvd_bank_path}; line will be null")
            _, D_all, both, X_all = diffs[role]
            both2 = torch.cat([both, both])
            view_grid = {f: {c: {d: [None] * n_layers
                                 for d in DIRECTIONS + ["random_sd", "random_actcov_sd"]}
                             for c in CENTERINGS} for f in FILTERS}
            cos_view_grid = {f: {d: [None] * n_layers for d in COS_KEYS} for f in FILTERS}
            for j in range(n_layers):
                if j not in bank:
                    continue
                u_damped = unit(bank[j]["damped"].float())
                u_exact = unit(bank[j]["exact"].float())
                for filt in FILTERS:
                    D = D_all[:, j, :] if filt == "all" else D_all[both][:, j, :]
                    X = X_all[:, j, :] if filt == "all" else X_all[both2][:, j, :]
                    Xc = X - X.mean(dim=0, keepdim=True)
                    Dn = row_unit(D)
                    vals = {
                        "damped": explained_ratios(D, u_damped),
                        "exact": explained_ratios(D, u_exact),
                        "fv_diff": explained_ratios(D, u_fv),
                        "top_pc": top_pc_explained(D),
                    }
                    cg = cos_view_grid[filt]
                    cg["inv_fv_diff"][j] = mean_cos(Dn, u_exact)
                    if j in pca_bank:
                        cg["inv_fv_diff_pcak16"][j] = mean_cos(Dn, unit(pca_bank[j]["pca_k16"].float()))
                    if j in tsvd_bank:
                        cg["inv_fv_diff_tsvdk16"][j] = mean_cos(Dn, unit(tsvd_bank[j]["tsvd"].float()))
                    cg["fv_diff"][j] = mean_cos(Dn, u_fv)
                    # analytic max of mean_i cos(d_i, x) over unit x (at x = unit(mean u_i))
                    cg["mean_dir"][j] = float(Dn.mean(dim=0).norm())
                    rand = {c: [] for c in CENTERINGS}
                    rand_cov = {c: [] for c in CENTERINGS}
                    cos_rand, cos_rand_cov = [], []
                    for _ in range(args.n_random):
                        v_iso = unit(torch.randn(D.shape[1], generator=rng))
                        r = explained_ratios(D, v_iso)
                        # v = Xc^T g is an exact sample from N(0, Sigma_hat) (rank <= rows-1)
                        g = torch.randn(Xc.shape[0], generator=rng_cov)
                        v_cov = unit(Xc.T @ g)
                        rc = explained_ratios(D, v_cov)
                        cos_rand.append(mean_cos(Dn, v_iso))
                        cos_rand_cov.append(mean_cos(Dn, v_cov))
                        for c in CENTERINGS:
                            rand[c].append(r[c])
                            rand_cov[c].append(rc[c])
                    cg["random"][j] = float(np.mean(cos_rand))
                    cg["random_sd"][j] = float(np.std(cos_rand))
                    cg["random_actcov"][j] = float(np.mean(cos_rand_cov))
                    cg["random_actcov_sd"][j] = float(np.std(cos_rand_cov))
                    for c in CENTERINGS:
                        for d in ("damped", "exact", "fv_diff", "top_pc"):
                            view_grid[filt][c][d][j] = vals[d][c]
                        view_grid[filt][c]["random"][j] = float(np.mean(rand[c]))
                        view_grid[filt][c]["random_sd"][j] = float(np.std(rand[c]))
                        view_grid[filt][c]["random_actcov"][j] = float(np.mean(rand_cov[c]))
                        view_grid[filt][c]["random_actcov_sd"][j] = float(np.std(rand_cov[c]))
            grid[view] = view_grid
            cos_grid[view] = cos_view_grid
            print(f"  {view}: done (cell {cell})")

        counts = {f: int(next(iter(diffs.values()))[2].sum()) if f == "both_correct"
                  else next(iter(diffs.values()))[1].shape[0] for f in FILTERS}
        write_json(out_dir / "explained_grid.json", {
            "pair": [f1, f2], "diff_convention": "d = act_f1 - act_f2; pre-image target fv_f1 - fv_f2",
            "preimage_root": str(args.preimage_root), "fv_root": str(fv_root),
            "role_views": [{"view": v, "role": r, "cell": c} for v, r, c in ROLE_VIEWS],
            "n_pairs": counts, "n_random": args.n_random,
            "random_actcov_note": "unit(Xc^T g), g~N(0,I): direction sampled from the empirical"
                                  " covariance of the raw (undiffed) two-shot activations of"
                                  " both functions at that (role, layer), per filter;"
                                  " seed = --seed + 1",
            "layers": list(range(n_layers)),
            "layer_note": "two-shot layer index j = block output j = ridge capture layer j+1",
            "explained": grid,
        })
        write_json(out_dir / "cos_grid.json", {
            "pair": [f1, f2],
            "metric": "SIGNED mean_i cos(d_i, x); d_i = act_f1 - act_f2; positive = diffs point"
                      " toward x. inv_fv_diff = exact (undamped) pre-image of fv_f1 - fv_f2;"
                      " inv_fv_diff_pcak16 = pre-image through the k=16 PCA ridge"
                      " (fit_pca_ridge_preimages_multicell.py, same fv_root);"
                      " inv_fv_diff_tsvdk16 = rank-16 TSVD pre-image of the full-dim maps"
                      " (fit_tsvd_preimages_multicell.py);"
                      " mean_dir = ||mean_i unit(d_i)|| = analytic max over unit x"
                      " (mean_dir^2 ~ Stream K mean pairwise cosine).",
            "preimage_root": str(args.preimage_root), "fv_root": str(fv_root),
            "role_views": [{"view": v, "role": r, "cell": c} for v, r, c in ROLE_VIEWS],
            "n_pairs": counts, "n_random": args.n_random,
            "random_actcov_note": "same covariance-matched draws as explained_grid.json;"
                                  " signed expectation ~0 for both baselines - the anisotropy"
                                  " shows in the sd (band width)",
            "layers": list(range(n_layers)),
            "layer_note": "two-shot layer index j = block output j = ridge capture layer j+1",
            "mean_cos": cos_grid,
        })

        views = [v for v, _, _ in ROLE_VIEWS if v in grid]
        # --- heatmaps: one per (direction, centering, filter)
        for d in ("damped", "exact", "fv_diff", "top_pc"):
            for c in CENTERINGS:
                for filt in FILTERS:
                    M = np.array([[grid[v][filt][c][d][j] if grid[v][filt][c][d][j] is not None
                                   else np.nan for j in range(n_layers)] for v in views])
                    fig, ax = plt.subplots(figsize=(12, 3.2))
                    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0)
                    ax.set_yticks(range(len(views)), views, fontsize=8)
                    ax.set_xticks(range(0, n_layers, 2))
                    ax.set_xlabel("layer (two-shot index = edit layer)")
                    ax.set_title(f"{pair_name}: explained pair-diff variance ({d}, {c}, "
                                 f"{filt}, n={counts[filt]})", fontsize=10)
                    fig.colorbar(im, ax=ax, fraction=0.025)
                    fig.tight_layout()
                    fig.savefig(out_dir / f"heatmap_{d}_{c}_{filt}.png", dpi=150)
                    plt.close(fig)

        # --- line plots: one figure per (centering, filter); panels = views
        colors = {"damped": "tab:blue", "exact": "tab:red", "fv_diff": "tab:green",
                  "top_pc": "tab:gray", "random": "k", "random_actcov": "tab:orange"}
        for c in CENTERINGS:
            for filt in FILTERS:
                fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
                for ax, v in zip(axes.flat, views):
                    xs = list(range(n_layers))
                    for d in DIRECTIONS:
                        ys = grid[v][filt][c][d]
                        style = dict(color=colors[d], lw=1.5)
                        if d == "top_pc":
                            style.update(ls="--")
                        if d == "random":
                            style.update(ls=":", lw=1)
                        if d == "random_actcov":
                            style.update(ls="-.", lw=1)
                        ax.plot(xs, [np.nan if y is None else y for y in ys], label=d, **style)
                    ax.set_title(v, fontsize=9)
                    ax.set_yscale("log")
                    ax.grid(alpha=0.3)
                for ax in axes.flat[len(views):]:
                    ax.axis("off")
                axes.flat[0].legend(fontsize=8)
                fig.suptitle(f"{pair_name}: explained pair-diff variance vs layer "
                             f"({c}, {filt}, n={counts[filt]}) — log scale", fontsize=11)
                fig.supxlabel("layer")
                fig.tight_layout()
                fig.savefig(out_dir / f"lines_{c}_{filt}.png", dpi=150)
                plt.close(fig)

        print(f"  wrote {out_dir}")

    print("DONE")


if __name__ == "__main__":
    main()
