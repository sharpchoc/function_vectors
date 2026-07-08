#!/usr/bin/env python
"""Why does the EXACT FV-diff pre-image have ~zero mean cosine with the two-shot pair diffs?

Hypothesis (Stream R): cond(W_std) ~ 1e9, so the exact inverse dz = sum_i (u_i^T fv / s_i) v_i
is dominated by W's SMALLEST singular directions, which are unrelated to where the diff
vectors actually live. Diagnostics per (cell, layer):

  1. Energy spectra in W's right-singular basis (V of W_std^T = U S V^T): squared-coefficient
     distribution over singular index i for (a) the exact pre-image, (b) the damped pre-image,
     (c) the raw mean diff direction mapped to standardized space (d/std). If the hypothesis
     holds, (a) concentrates at the spectrum tail while (c) does not.
  2. Truncated-inverse sweep: dz_k = sum_{i<k} (u_i^T fv / s_i) v_i, dx_k = std * dz_k;
     mean_i cos(d_i, dx_k) vs k. Shows how alignment builds with well-conditioned components
     and collapses as the noise-amplified tail is added (exact = k=4096).
  3. Scalars: cond(W), |dz_exact|/|dz_damped|, cos(exact, damped) in raw space, fraction of
     exact-pre-image energy in the bottom decile of the spectrum.
  4. Direction instability: cos(bank exact, exact recomputed from the fp16-saved W). The maps
     store W in fp16 (~1e-3 relative rounding); with cond(W) ~ 1e9 that perturbation totally
     reorients the exact inverse, so this cosine ~ 0 is itself the core finding. The bank's
     fp32-derived vectors are used for all spectra/mean-cos numbers; the fp16 SVD is used only
     for the basis and the truncated sweep (top singular directions are stable under 1e-3
     perturbation, the tail is noise under either precision).

Outputs (TRACKED): results/.../twoshot_pairdiff_fv_preimage/<fv_root>/preimage_diagnostics/
  spectrum_{pair}_L{j}.png, truncation_sweep_{pair}.png, diagnostics.json
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

from src.eval_scripts.analyze_twoshot_pairdiff_fv_preimage import (
    PAIR_DIRS,
    load_pair_diffs,
    row_unit,
    unit,
)
from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    load_function_vector,
    load_json,
    torch_load_trusted,
    write_json,
)
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", nargs="+", default=list(PAIR_DIRS))
    p.add_argument("--twoshot_root", type=Path, default=ARTIFACTS_ROOT / "twoshot_paired_graded")
    p.add_argument("--preimage_root", type=Path,
                   default=ARTIFACTS_ROOT / "preimage_pairdiff/train_varicl_max4_top40")
    p.add_argument("--cell", type=str, default="pre_label_token_icl3",
                   help="Stage-1 cell dir (default: the query_final view's context-matched cell).")
    p.add_argument("--role", type=str, default="query_final",
                   help="Two-shot role whose diffs to compare against.")
    p.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 20],
                   help="Two-shot layer indices (= bank edit_layers = capture layer - 1).")
    p.add_argument("--ks", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 3686, 4096])
    p.add_argument("--output_root", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_root or (FV_FORMATION_DIR / "twoshot_pairdiff_fv_preimage"
                                   / args.preimage_root.name / "preimage_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    fv_root = Path(load_json(args.preimage_root / "run_config.json")["fv_root"])

    # SVDs are per (cell, layer) and shared by both pairs
    svds = {}
    for j in args.layers:
        m = torch_load_trusted(args.preimage_root / args.cell / "maps" / f"layer_{j + 1:02d}.pt",
                               map_location="cpu")
        w_t64 = m["w_std"].double().T
        u, svals, vh = torch.linalg.svd(w_t64)
        svds[j] = {"u": u, "svals": svals, "vh": vh, "std": m["std"].double()}
        print(f"layer {j}: SVD done, cond(W) = {float(svals[0] / svals[-1]):.3e}")

    summary = {"cell": args.cell, "role": args.role, "fv_root": str(fv_root),
               "layers": args.layers, "ks": args.ks, "per_pair": {}}
    for pair_name in args.pairs:
        f1, f2 = PAIR_DIRS[pair_name]
        print(f"== {pair_name}")
        diffs = load_pair_diffs(args.twoshot_root / pair_name, f1, f2)
        _, D_all, _, _ = diffs[args.role]
        fv = (load_function_vector(fv_root, f1) - load_function_vector(fv_root, f2)).double()
        bank = torch_load_trusted(args.preimage_root / args.cell / "pairdiff_preimages"
                                  / f"{f1}__{f2}_pairdiff_preimage_bank.pt",
                                  map_location="cpu")["preimages_by_edit_layer"]
        pair_out = {}
        fig_sw, axes_sw = plt.subplots(1, len(args.layers), figsize=(4.2 * len(args.layers), 3.6),
                                       sharey=True)
        for ax, j in zip(np.atleast_1d(axes_sw), args.layers):
            s = svds[j]
            Dn = row_unit(D_all[:, j, :].double())
            d = svds[j]["std"].shape[0]
            utf = s["u"].T @ fv
            c_exact_fp16 = utf / s["svals"]                 # V-basis coords, fp16-W exact dz

            # instability probe: exact from the fp16-saved W vs the bank's fp32-derived exact
            dx_exact_fp16 = (s["vh"].T @ c_exact_fp16) * s["std"]
            bank_exact = bank[j]["exact"].double()
            instab = float(torch.dot(unit(dx_exact_fp16), unit(bank_exact)))

            dz_exact = bank_exact / s["std"]                # bank vectors for all spectra
            c_exact = s["vh"] @ dz_exact
            dz_damped = bank[j]["damped"].double() / s["std"]
            c_damped = s["vh"] @ dz_damped
            m_std = unit(Dn.mean(dim=0)) / s["std"]         # mean diff direction, std space
            c_diff = s["vh"] @ unit(m_std)

            # energy spectra (block-summed for plotting)
            def energy(c):
                e = (c ** 2).numpy()
                return e / e.sum()
            e_exact, e_damped, e_diff = energy(c_exact), energy(c_damped), energy(c_diff)
            bottom_decile = int(0.9 * d)
            tail_frac = float(e_exact[bottom_decile:].sum())

            fig, ax2 = plt.subplots(figsize=(6.4, 4.2))
            nblk = 64
            for e, lab, col in [(e_exact, "exact pre-image", "tab:red"),
                                (e_damped, "damped pre-image", "tab:blue"),
                                (e_diff, "mean diff direction", "tab:green")]:
                blk = e[: (d // nblk) * nblk].reshape(nblk, -1).sum(axis=1)
                ax2.semilogy(np.arange(nblk) * (d // nblk), np.maximum(blk, 1e-12),
                             label=lab, color=col)
            ax2.set_xlabel("singular index of W_std^T (0 = largest s_i)")
            ax2.set_ylabel(f"energy per block of {d // nblk}")
            ax2.set_title(f"{pair_name} L{j}: energy in W's right-singular basis\n"
                          f"cond(W)={float(s['svals'][0]/s['svals'][-1]):.1e}, "
                          f"exact tail(bottom 10%)={tail_frac:.2f}", fontsize=9)
            ax2.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"spectrum_{pair_name}_L{j}.png", dpi=150)
            plt.close(fig)

            # truncated-inverse sweep (fp16 SVD; top-k directions stable under fp16 rounding)
            sweep = []
            for k in args.ks:
                dz_k = s["vh"].T[:, :k] @ c_exact_fp16[:k]
                sweep.append(float((Dn @ unit(dz_k * s["std"])).mean()))
            cos_damped = float((Dn @ unit(bank[j]["damped"].double())).mean())
            cos_fvdiff = float((Dn @ unit(fv)).mean())
            cos_exact_bank = float((Dn @ unit(bank_exact)).mean())
            ax.semilogx(args.ks, sweep, "o-", color="tab:red", ms=3,
                        label="truncated inv (top-k)")
            ax.axhline(cos_damped, color="tab:blue", ls="--", lw=1, label="damped")
            ax.axhline(cos_fvdiff, color="tab:green", ls=":", lw=1, label="fv_diff (raw)")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"L{j}", fontsize=9)
            ax.set_xlabel("k (top singular dirs kept)")
            ax.grid(alpha=0.3)

            pair_out[str(j)] = {
                "cond_W": float(s["svals"][0] / s["svals"][-1]),
                "exact_energy_bottom_decile": tail_frac,
                "damped_energy_bottom_decile": float(e_damped[bottom_decile:].sum()),
                "diffmean_energy_bottom_decile": float(e_diff[bottom_decile:].sum()),
                "dz_norm_ratio_exact_over_damped": float(dz_exact.norm() / dz_damped.norm()),
                "cos_exact_damped_raw":
                    float(torch.dot(unit(bank_exact), unit(bank[j]["damped"].double()))),
                "meancos_truncated_by_k": dict(zip(map(str, args.ks), sweep)),
                "meancos_damped": cos_damped, "meancos_fv_diff": cos_fvdiff,
                "meancos_exact_bank": cos_exact_bank,
                "exact_fp16_vs_bank_cos": instab,
            }
            print(f"  L{j}: tail_frac(exact)={tail_frac:.3f}, best truncated meancos="
                  f"{max(sweep):.3f} @k={args.ks[int(np.argmax(sweep))]}, "
                  f"exact meancos={cos_exact_bank:.4f}, damped={cos_damped:.3f}, "
                  f"fp16-instability cos={instab:.4f}")
        np.atleast_1d(axes_sw)[0].set_ylabel("mean cos(diff, x)")
        np.atleast_1d(axes_sw)[0].legend(fontsize=8)
        fig_sw.suptitle(f"{pair_name} ({args.role}): mean cos vs truncation rank of the inverse",
                        fontsize=11)
        fig_sw.tight_layout()
        fig_sw.savefig(out_dir / f"truncation_sweep_{pair_name}.png", dpi=150)
        plt.close(fig_sw)
        summary["per_pair"][pair_name] = pair_out

    write_json(out_dir / "diagnostics.json", summary)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
