#!/usr/bin/env python
"""Per-prompt read directions for the 69-task pool under the canonical 37-head set.

Glossary (write_up/task_id_im_subspaces.md, Eq. 4-5): the per-prompt read direction is
    r^j_A = M^+ v^j_A / ||M^+ v^j_A||,   M = sum_{h in H} W_O^h W_V^h,
with H = the canonical 37-head pooled sparse selection for the 69-task pool
(artifacts/sandbox/ext_steerability/prunedfail_seed43/pooled_sparse/selection.json,
DECISIONS 2026-08-16).

Conventions carried over from the adjudicated sparse23 precedent
(src/sandbox/sparse_head_selection/compute_perprompt_read_dirs_sparse23.py):
  * TWO truncation variants from one SVD:
      - 'literal': glossary Eq. 5, truncate machine-precision zeros only (rcond = eps*s1);
      - 'rank90' : repo convention, k = smallest with cum(S^2)/sum(S^2) >= 0.90.
  * Weights fp16-cast first (capture convention), then all math in fp64.
  * SVD driver='gesvd' on CUDA (gesvdj is ~1e-3-inaccurate); CPU LAPACK otherwise.

Inputs: per-prompt FVs captured by the peer session at
artifacts/69_task_run/perprompt_fvs/<task>.pt with keys
  'fv' (150,4096) fp16   per-prompt FVs (sum of W_O-projected 37-head outputs at cue token)
  'raw' (150,37,256) fp16 out_proj inputs of the selected heads
  'sel_flat', 'selection_path', 'prompt_index'

Gates (hard stop -> user adjudicates, never self-adjudicate):
  (a) per task, the stored 'fv' must match Sum_h W_O^h raw_h rebuilt with this script's
      fp16-cast out_proj slices (fp16-storage tolerance);
  (b) per task, mean of the stored 'fv' rows must match the FV rebuilt from the stored
      capture means (prunedfail_seed43/<task>/means.pt head_means) on the 37 heads;
  (c) fp64 SVD must reconstruct M to rel <= 1e-10.

Outputs: <out_dir>/<task>.pt (r_literal, r_rank90 (150,4096) fp32 unit rows; per-variant
pre-inversion norms, cos(Mr,v), cos(Mr,P_k v), task-level r_task; cos literal-vs-rank90),
M_spectrum.npz, build_summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.utils.paths import ARTIFACTS_ROOT

HEAD_DIM, N_HEADS = 256, 16


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fv_dir", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--out_dir", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_read_dirs")
    p.add_argument("--means_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43")
    p.add_argument("--selection_path", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43"
                   / "pooled_sparse" / "selection.json")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="pytorch_model.bin; default: latest gpt-j-6b snapshot in the volume HF cache")
    return p.parse_args()


def torch_load(p, **kw):
    return torch.load(p, map_location="cpu", weights_only=False, **kw)


def main():
    args = parse_args()
    ckpt = args.ckpt or sorted(Path("/workspace/.cache/huggingface/hub/"
                                    "models--EleutherAI--gpt-j-6b/snapshots").glob("*/pytorch_model.bin"))[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"device": device, "dtype": "fp16-cast weights -> fp64 math",
               "fv_source": str(args.fv_dir), "selection_path": str(args.selection_path),
               "definition": "glossary Eq.4-5 per-prompt read direction, H = canonical 37 heads"}

    sel_flat = sorted(json.load(open(args.selection_path))["selected_flat"])
    heads = [(f // N_HEADS, f % N_HEADS) for f in sel_flat]
    assert len(heads) == 37
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})

    # --- build M = sum of 37 OV circuits (fp16-cast weights, fp64 accumulation) ---
    sd = torch_load(ckpt, mmap=True)
    wo_slices = []  # in sel_flat (sorted) order == the capture's 'raw' head axis order
    M = torch.zeros(4096, 4096, dtype=torch.float64)
    for l, h in heads:
        wo = sd[f"transformer.h.{l}.attn.out_proj.weight"][:, h * HEAD_DIM:(h + 1) * HEAD_DIM].clone().half()
        wv = sd[f"transformer.h.{l}.attn.v_proj.weight"][h * HEAD_DIM:(h + 1) * HEAD_DIM, :].clone().half()
        wo_slices.append(wo)
        M += wo.double() @ wv.double()
    WO = torch.stack(wo_slices).double().to(device)  # (37, 4096, 256)

    # --- one SVD of M in fp64 ---
    Md = M.to(device)
    if device == "cuda":
        U, S, Vh = torch.linalg.svd(Md, full_matrices=False, driver="gesvd")
        summary["svd_driver"] = "gesvd"
    else:
        U, S, Vh = torch.linalg.svd(Md, full_matrices=False)
        summary["svd_driver"] = "cpu-lapack"

    # GATE (c): reconstruction.
    rec = (U @ torch.diag(S) @ Vh - Md).norm() / Md.norm()
    assert rec < 1e-10, f"GATE(c) FAILED: SVD reconstruction rel={rec:.3e} - HARD STOP, inform user"
    print(f"gate (c) OK: SVD reconstruction rel={rec:.3e} (driver={summary['svd_driver']})")

    s = S.cpu().numpy()
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    eps = np.finfo(np.float64).eps
    k_literal = int((s > eps * s[0] * max(M.shape)).sum())
    k_rank90 = int(np.searchsorted(energy, 0.90) + 1)
    variants = {"literal": k_literal, "rank90": k_rank90}
    summary["spectrum"] = {
        "sigma_max": float(s[0]), "sigma_min": float(s[-1]),
        "condition_number": float(s[0] / s[-1]),
        "k_literal": k_literal, "k_rank90": k_rank90,
        "k_energy_95": int(np.searchsorted(energy, 0.95) + 1),
        "k_energy_99": int(np.searchsorted(energy, 0.99) + 1),
    }
    print("spectrum:", json.dumps(summary["spectrum"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "M_spectrum.npz", singular_values=s,
                        heads=np.array(heads), energy=energy)

    def pinv_apply(V_batch, k):
        """rows of V_batch (N, 4096) -> M^+ v = Vh[:k]^T diag(1/S[:k]) U[:,:k]^T v."""
        coefs = V_batch @ U[:, :k]                    # (N, k)
        return (coefs / S[:k]) @ Vh[:k]               # (N, 4096)

    def proj_span_U(V_batch, k):
        return (V_batch @ U[:, :k]) @ U[:, :k].T

    tasks = sorted(p.stem for p in args.fv_dir.glob("*.pt"))
    assert len(tasks) == 69 and set(tasks) == set(group), \
        f"expected the 69 split tasks, got {len(tasks)} files"
    per_task_rows = []
    for task in tasks:
        d = torch_load(args.fv_dir / f"{task}.pt")
        assert sorted(int(x) for x in d["sel_flat"]) == sel_flat, \
            f"{task}: sel_flat mismatch vs canonical selection - HARD STOP, inform user"
        v = d["fv"].double().to(device)                       # (150, 4096)
        raw = d["raw"].double().to(device)                    # (150, 37, 256)
        assert v.shape == (150, 4096) and raw.shape == (150, 37, 256)

        # GATE (a): stored fv == sum_h W_O^h raw_h (fp16-storage tolerance).
        v_rebuilt = torch.einsum("hdk,nhk->nd", WO, raw)
        rel_a = ((v_rebuilt - v).norm(dim=1) / v.norm(dim=1)).max().item()
        assert rel_a < 2e-3, f"GATE(a) FAILED {task}: fv-vs-raw rebuild rel={rel_a:.3e} - HARD STOP, inform user"

        # GATE (b): mean of stored fv rows == FV from the stored capture means.
        hm = torch_load(args.means_root / task / "means.pt")["head_means"].double()  # (28,16,256)
        v_means = torch.stack([hm[l, h] for l, h in heads]).to(device)               # (37, 256)
        v_means = torch.einsum("hdk,hk->d", WO, v_means)
        rel_b = ((v.mean(0) - v_means).norm() / v_means.norm()).item()
        assert rel_b < 1e-3, f"GATE(b) FAILED {task}: fv-mean-vs-means.pt rel={rel_b:.3e} - HARD STOP, inform user"

        out = {"task": task, "group": group[task], "heads": [list(x) for x in heads],
               "sel_flat": sel_flat, "prompt_index": d["prompt_index"],
               "v": d["fv"].clone(),
               "config": {"definition": summary["definition"], "fv_source": str(args.fv_dir),
                          "variants": variants, "svd_driver": summary["svd_driver"]}}
        row = {"task": task, "group": group[task],
               "gate_a_rel": round(rel_a, 8), "gate_b_rel": round(rel_b, 8)}
        r_by_variant = {}
        v_task = v.mean(0, keepdim=True)
        for vname, k in variants.items():
            r_raw = pinv_apply(v, k)
            norms = r_raw.norm(dim=1)
            r = r_raw / norms[:, None]
            Mr = r @ Md.T  # rows are M @ r_j
            cos_v = torch.nn.functional.cosine_similarity(Mr, v, dim=1)
            cos_pv = torch.nn.functional.cosine_similarity(Mr, proj_span_U(v, k), dim=1)
            rt = pinv_apply(v_task, k)
            out[vname] = {"r": r.float().cpu(), "preinv_norm": norms.float().cpu(),
                          "cos_Mr_v": cos_v.float().cpu(), "cos_Mr_Pkv": cos_pv.float().cpu(),
                          "r_task": (rt / rt.norm()).float().cpu().squeeze(0)}
            r_by_variant[vname] = r
            row[f"{vname}_median_cos_Mr_v"] = round(float(cos_v.median()), 4)
            row[f"{vname}_min_cos_Mr_Pkv"] = round(float(cos_pv.min()), 6)
        agree = torch.nn.functional.cosine_similarity(
            r_by_variant["literal"], r_by_variant["rank90"], dim=1)
        out["cos_literal_vs_rank90"] = agree.float().cpu()
        row["median_cos_literal_vs_rank90"] = round(float(agree.median()), 4)
        torch.save(out, args.out_dir / f"{task}.pt")
        per_task_rows.append(row)
        print(f"{task} [{group[task]}]: gates a={rel_a:.2e} b={rel_b:.2e} | "
              f"lit cos(Mr,v) med={row['literal_median_cos_Mr_v']} | "
              f"r90 med={row['rank90_median_cos_Mr_v']} | "
              f"lit-vs-r90 med={row['median_cos_literal_vs_rank90']}", flush=True)

    summary["tasks"] = per_task_rows
    with open(args.out_dir / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {len(tasks)} task files to {args.out_dir}")


if __name__ == "__main__":
    main()
