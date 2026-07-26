#!/usr/bin/env python
"""SANDBOX (not repo standard): singular-value spectra of the per-prompt ridge map vs canonical.

For ONE cell (default the sandbox best, icl10/pre_label_token L13) this fits the full-dim
4096->4096 ridge weight matrix exactly as the two studies do (single 20-train-task standardizer,
centered eigendecomposition ridge) under both target modes:

  * canonical  -- Y = the task's varicl_top40 FV broadcast to all 170 rows (rank(Y) <= 20)
  * perprompt  -- Y = each prompt's top-40 head-sum vector (rank(Y) up to n_rows)

and computes full SVD spectra of the four weight matrices {canonical, perprompt} x {own alpha,
swapped alpha} -- the alpha swap separates what the per-prompt targets change from what the
~10x smaller CV-chosen alpha changes -- plus the spectra of the two centered TARGET matrices
themselves (the rank bound the maps inherit).

Repro gates: each map's test-vs-stored-FV MSE at its own alpha must match the stored study
metrics (canonical combined_metrics.csv / sandbox perprompt shard CSV) before spectra are saved.

Output (grid-only PNG policy): one summary figure + spectra.npz + summary.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT  # noqa: E402
from sandbox.perprompt_fv.regress_activation_to_perprompt_headsum_ridge import (  # noqa: E402
    load_function_vector,
    load_json,
    load_perprompt_targets,
    load_task_role_pooled,
    align_targets,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="SANDBOX: weight/target SVD spectra, per-prompt vs canonical ridge.")
    p.add_argument("--icl_index", type=int, default=10)
    p.add_argument("--token_role", type=str, default="pre_label_token")
    p.add_argument("--layer", type=int, default=13)
    p.add_argument("--alpha_canonical", type=float, default=31622.776601683792,
                   help="Canonical study's CV-chosen alpha for this cell.")
    p.add_argument("--alpha_perprompt", type=float, default=3162.2776601683795,
                   help="Sandbox per-prompt study's CV-chosen alpha for this cell.")
    p.add_argument("--gate_mse_canonical", type=float, default=0.19633592156802906,
                   help="Stored canonical test MSE for this cell (repro gate; <=0 disables).")
    p.add_argument("--gate_mse_perprompt", type=float, default=0.1528394819307728,
                   help="Stored sandbox test_mse_fv for this cell (repro gate; <=0 disables).")
    p.add_argument("--task_manifest", type=Path, default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_varicl_top40")
    p.add_argument("--targets_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox/perprompt_head_acts/gptj_train_varicl_top40")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations/gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--test_tasks", nargs="+", default=[
        "landmark-country", "word_length", "capitalize_first_letter", "synonym",
        "lowercase_first_letter", "capitalize", "antonym"])
    p.add_argument("--std_eps", type=float, default=1e-6)
    p.add_argument("--output_dir", type=Path,
                   default=RESULTS_ROOT / "sandbox/perprompt_ridge_pilot/gptj_train_varicl_top40/weight_spectrum")
    return p.parse_args()


def spectrum_stats(sv):
    sv = np.asarray(sv, dtype=np.float64)
    s1 = sv[0]
    energy = sv ** 2
    cum = np.cumsum(energy) / energy.sum()
    pr = float(energy.sum() ** 2 / (energy ** 2).sum())  # participation ratio
    return {
        "top_sv": float(s1),
        "numerical_rank_1e-6": int((sv > 1e-6 * s1).sum()),
        "rank_energy_90": int(np.searchsorted(cum, 0.90) + 1),
        "rank_energy_99": int(np.searchsorted(cum, 0.99) + 1),
        "participation_ratio": pr,
        "sv20_over_sv1": float(sv[19] / s1),
        "sv21_over_sv1": float(sv[20] / s1),
        "sv100_over_sv1": float(sv[99] / s1),
    }


def main():
    args = parse_args()
    t0 = time.time()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    manifest = load_json(args.task_manifest)
    train_tasks = sorted(manifest["train_tasks"])
    test_tasks = sorted(args.test_tasks)
    all_tasks = train_tasks + test_tasks
    root = (args.query_activations_root if args.icl_index == 10
            else Path(args.icl_activations_root_template.format(icl=args.icl_index)))

    # X at the cell's layer + row-aligned per-prompt targets; broadcast FVs.
    xs, y_pp = {}, {}
    fvs = {t: load_function_vector(args.fv_root, t) for t in all_tasks}
    for task in all_tasks:
        acts, keys = load_task_role_pooled(root, task, args.splits, args.token_role, args.icl_index)
        xs[task] = acts[:, args.layer, :].float()
        y_pp[task] = align_targets(load_perprompt_targets(args.targets_root, task, args.splits), keys, task)
    print(f"loaded {len(all_tasks)} tasks in {time.time()-t0:.0f}s", flush=True)

    # Single standardizer on pooled train rows (canonical protocol).
    x_fit_raw = torch.cat([xs[t] for t in train_tasks], dim=0)
    mu = x_fit_raw.mean(dim=0)
    sd = x_fit_raw.std(dim=0).clamp_min(args.std_eps)
    xz = {t: (xs[t] - mu) / sd for t in all_tasks}
    x_fit = torch.cat([xz[t] for t in train_tasks], dim=0)

    y_fit = {
        "canonical": torch.cat([fvs[t].unsqueeze(0).expand(xz[t].shape[0], -1) for t in train_tasks], dim=0),
        "perprompt": torch.cat([y_pp[t] for t in train_tasks], dim=0),
    }

    # One eigendecomposition serves every alpha for both target modes.
    xbar = x_fit.mean(dim=0)
    xc = x_fit - xbar
    eigvals, eigvecs = torch.linalg.eigh(xc.T @ xc)
    print(f"gram eigendecomposition done ({time.time()-t0:.0f}s)", flush=True)

    def fit_w(mode, alpha):
        yb = y_fit[mode].mean(dim=0)
        c = eigvecs.T @ (xc.T @ (y_fit[mode] - yb))
        w = eigvecs @ (c / (eigvals + alpha).unsqueeze(1))
        return w, yb

    def test_mse_fv(w, yb):
        sqerr, n = 0.0, 0
        for task in test_tasks:
            pred = (xz[task] - xbar) @ w + yb
            sqerr += float(torch.sum((pred - fvs[task].unsqueeze(0)) ** 2))
            n += xz[task].shape[0]
        return sqerr / (n * w.shape[1])

    alphas = {"canonical": args.alpha_canonical, "perprompt": args.alpha_perprompt}
    gates = {"canonical": args.gate_mse_canonical, "perprompt": args.gate_mse_perprompt}
    arms = {}  # label -> (mode, alpha)
    for mode in ("canonical", "perprompt"):
        for aname, alpha in alphas.items():
            arms[f"{mode}_alpha_{'own' if aname == mode else 'swapped'}"] = (mode, alpha)

    summary = {"cell": f"icl{args.icl_index}/{args.token_role}/L{args.layer}",
               "alphas": alphas, "arms": {}, "targets": {}}
    spectra = {}

    # Repro gates first (own-alpha arms).
    for mode in ("canonical", "perprompt"):
        w, yb = fit_w(mode, alphas[mode])
        mse = test_mse_fv(w, yb)
        summary["arms"][f"{mode}_alpha_own"] = {"test_mse_fv": mse}
        if gates[mode] > 0:
            rel = abs(mse - gates[mode]) / gates[mode]
            print(f"GATE {mode}: test_mse_fv={mse:.6f} stored={gates[mode]:.6f} rel={rel:.2e}", flush=True)
            if rel > 1e-3:
                raise RuntimeError(f"REPRO GATE FAILED for {mode}: {mse} vs stored {gates[mode]}. "
                                   f"STOP -- user adjudicates.")

    for label, (mode, alpha) in arms.items():
        w, _ = fit_w(mode, alpha)
        sv = torch.linalg.svdvals(w.double()).numpy()
        spectra[f"sv_{label}"] = sv
        summary["arms"].setdefault(label, {}).update(spectrum_stats(sv))
        summary["arms"][label]["alpha"] = alpha
        print(f"svdvals {label}: rank90={summary['arms'][label]['rank_energy_90']} "
              f"pr={summary['arms'][label]['participation_ratio']:.1f} ({time.time()-t0:.0f}s)", flush=True)

    # Centered target-matrix spectra (the rank bound W inherits).
    for mode in ("canonical", "perprompt"):
        yc = (y_fit[mode] - y_fit[mode].mean(dim=0)).double()
        sv = torch.linalg.svdvals(yc).numpy()
        spectra[f"sv_targets_{mode}"] = sv
        summary["targets"][mode] = spectrum_stats(sv)
        print(f"svdvals targets {mode}: numrank={summary['targets'][mode]['numerical_rank_1e-6']}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "spectra.npz", **spectra)
    write_json(args.output_dir / "summary.json", summary)

    # Figure: the two FINAL maps (own CV-chosen alpha each), one spectrum per panel —
    # side-by-side qualitative shape comparison, not a magnitude overlay.
    panels = [
        ("canonical_alpha_own", f"canonical W  (FV-broadcast targets, α={alphas['canonical']:.3g})", "#4878a8"),
        ("perprompt_alpha_own", f"per-prompt W  (head-sum targets, α={alphas['perprompt']:.3g})", "#d62728"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, (label, title, color) in zip(axes, panels):
        sv = spectra[f"sv_{label}"]
        ax.plot(np.arange(1, len(sv) + 1), sv / sv[0], lw=1.4, color=color)
        ax.axvline(20, color="gray", lw=0.8, ls=":", label="rank 20 (# train tasks)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("singular value index")
        ax.set_ylabel("σ_i / σ_1")
        ax.set_ylim(1e-9, 2)
        ax.grid(alpha=0.25)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"SANDBOX: singular-value structure of the final ridge maps — {summary['cell']}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output_dir / "spectra_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"done in {time.time()-t0:.0f}s -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
