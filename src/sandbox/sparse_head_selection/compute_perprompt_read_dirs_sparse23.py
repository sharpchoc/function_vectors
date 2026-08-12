#!/usr/bin/env python
"""SANDBOX: per-prompt read directions for the vanilla_sparse_opt23 per-prompt FVs.

Glossary (write_up/task_id_im_subspaces.md, Eq. 4-5): the per-prompt read direction is
    r^j_A = M^+ v^j_A / ||M^+ v^j_A||,   M = sum_{h in H23} W_O^h W_V^h,
the unit input direction (orthogonal to ker(M)) whose image under the summed OV circuit of
the selected heads is maximally cosine-aligned with the per-prompt FV v^j_A.

User-gated choices (2026-08-10):
  * H = the 23 sparse-optimization heads (c > 0.8; == vanilla_sparse_opt23 manifest).
  * TWO truncation variants from one SVD:
      - 'literal': glossary Eq. 5, truncate machine-precision zeros only (rcond = eps*s1);
      - 'rank90' : repo convention, k = smallest with cum(S^2)/sum(S^2) >= 0.90.
  * Weights fp16-cast first (repo/capture convention), then all math in fp64.
  * SVD driver='gesvd' on CUDA (gesvdj is ~1e-3-inaccurate); CPU LAPACK otherwise.

Gates (hard stop -> user adjudicates): out_proj slices must match the capture's stored
top40_outproj_slices.pt on overlapping heads; fp64 SVD must reconstruct M to rel <= 1e-10.

Inputs : artifacts/sandbox/sparse_head_selection/perprompt_fv_sparse23/<task>.pt
Outputs: artifacts/sandbox/sparse_head_selection/perprompt_read_dirs_sparse23/
         <task>.pt, M_spectrum.npz, build_summary.json
"""
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

SPARSE_ROOT = ARTIFACTS_ROOT / "sandbox" / "sparse_head_selection"
FV_ROOT = SPARSE_ROOT / "perprompt_fv_sparse23"
OUT_ROOT = SPARSE_ROOT / "perprompt_read_dirs_sparse23"
CAPTURE_ROOT = ARTIFACTS_ROOT / "sandbox" / "perprompt_head_acts" / "gptj_train_varicl_top40"
CKPT = sorted(Path("/workspace/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots").glob("*/pytorch_model.bin"))[-1]
HEAD_DIM, N_HEADS = 256, 16


def torch_load(p, **kw):
    return torch.load(p, map_location="cpu", weights_only=False, **kw)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"sandbox": True, "device": device, "dtype": "fp16-cast weights -> fp64 math"}

    heads23 = [tuple(int(x) for x in hh) for hh in
               json.load(open(FV_ROOT / "build_summary.json"))["heads"]]
    assert len(heads23) == 23

    # --- build M = sum of 23 OV circuits (fp16-cast weights, fp64 accumulation) ---
    sd = torch_load(CKPT, mmap=True)
    out_slices = {}
    M = torch.zeros(4096, 4096, dtype=torch.float64)
    for l, h in heads23:
        wo = sd[f"transformer.h.{l}.attn.out_proj.weight"][:, h * HEAD_DIM:(h + 1) * HEAD_DIM].clone().half()
        wv = sd[f"transformer.h.{l}.attn.v_proj.weight"][h * HEAD_DIM:(h + 1) * HEAD_DIM, :].clone().half()
        out_slices[(l, h)] = wo
        M += wo.double() @ wv.double()

    # GATE (a): out_proj slices match the capture's stored slices on overlapping heads.
    stored = torch_load(CAPTURE_ROOT / "top40_outproj_slices.pt")
    n_checked = 0
    for k, v in stored["slices"].items():
        l, h = (int(x) for x in k[1:].split("H"))
        if (l, h) in out_slices:
            rel = (v.float() - out_slices[(l, h)].float()).norm() / v.float().norm()
            assert rel < 1e-6, f"GATE(a) FAILED at ({l},{h}): rel={rel:.3e} - HARD STOP, inform user"
            n_checked += 1
    print(f"gate (a) OK: {n_checked} overlapping out_proj slices match the capture's stored slices")

    # --- one SVD of M in fp64 ---
    Md = M.to(device)
    if device == "cuda":
        U, S, Vh = torch.linalg.svd(Md, full_matrices=False, driver="gesvd")
        summary["svd_driver"] = "gesvd"
    else:
        U, S, Vh = torch.linalg.svd(Md, full_matrices=False)
        summary["svd_driver"] = "cpu-lapack"

    # GATE (b): reconstruction.
    rec = (U @ torch.diag(S) @ Vh - Md).norm() / Md.norm()
    assert rec < 1e-10, f"GATE(b) FAILED: SVD reconstruction rel={rec:.3e} - HARD STOP, inform user"
    print(f"gate (b) OK: SVD reconstruction rel={rec:.3e} (driver={summary['svd_driver']})")

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
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_ROOT / "M_spectrum.npz", singular_values=s,
                        heads=np.array(heads23), energy=energy)

    def pinv_apply(V_batch, k):
        """rows of V_batch (N, 4096) -> M^+ v = Vh[:k]^T diag(1/S[:k]) U[:,:k]^T v."""
        coefs = V_batch @ U[:, :k]                    # (N, k)
        return (coefs / S[:k]) @ Vh[:k]               # (N, 4096)

    def proj_span_U(V_batch, k):
        return (V_batch @ U[:, :k]) @ U[:, :k].T

    tasks = sorted(p.stem for p in FV_ROOT.glob("*.pt"))
    per_task_rows = []
    for task in tasks:
        d = torch_load(FV_ROOT / f"{task}.pt")
        out = {"heads": [list(x) for x in heads23],
               "config": {"sandbox": True, "definition": "glossary Eq.4-5 per-prompt read direction",
                          "fv_source": str(FV_ROOT), "variants": variants,
                          "svd_driver": summary["svd_driver"]}}
        row = {"task": task}
        r_by_variant = {}
        for vname, k in variants.items():
            vout = {}
            for split in ("train", "test"):
                v = d[split]["fvs"].double().to(device)
                r_raw = pinv_apply(v, k)
                norms = r_raw.norm(dim=1)
                r = r_raw / norms[:, None]
                Mr = r @ Md.T  # rows are M @ r_j
                cos_v = torch.nn.functional.cosine_similarity(Mr, v, dim=1)
                cos_pv = torch.nn.functional.cosine_similarity(Mr, proj_span_U(v, k), dim=1)
                vout[split] = {"r": r.float().cpu(), "preinv_norm": norms.float().cpu(),
                               "cos_Mr_v": cos_v.float().cpu(), "cos_Mr_Pkv": cos_pv.float().cpu(),
                               "query_indices": d[split]["query_indices"]}
            v_task = torch.cat([d["train"]["fvs"], d["test"]["fvs"]]).mean(0, keepdim=True).double().to(device)
            r_task = pinv_apply(v_task, k)
            vout["r_task"] = (r_task / r_task.norm()).float().cpu().squeeze(0)
            out[vname] = vout
            r_by_variant[vname] = torch.cat([vout["train"]["r"], vout["test"]["r"]])
            row[f"{vname}_median_cos_Mr_v"] = round(float(torch.cat(
                [vout['train']['cos_Mr_v'], vout['test']['cos_Mr_v']]).median()), 4)
            row[f"{vname}_min_cos_Mr_Pkv"] = round(float(torch.cat(
                [vout['train']['cos_Mr_Pkv'], vout['test']['cos_Mr_Pkv']]).min()), 6)
        agree = torch.nn.functional.cosine_similarity(
            r_by_variant["literal"], r_by_variant["rank90"], dim=1)
        out["cos_literal_vs_rank90"] = agree
        row["median_cos_literal_vs_rank90"] = round(float(agree.median()), 4)
        torch.save(out, OUT_ROOT / f"{task}.pt")
        per_task_rows.append(row)
        print(f"{task}: lit cos(Mr,v) med={row['literal_median_cos_Mr_v']} | "
              f"r90 med={row['rank90_median_cos_Mr_v']} | "
              f"lit-vs-r90 med={row['median_cos_literal_vs_rank90']}", flush=True)

    summary["tasks"] = per_task_rows
    with open(OUT_ROOT / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {len(tasks)} task files to {OUT_ROOT}")


if __name__ == "__main__":
    main()
