#!/usr/bin/env python
"""SANDBOX: 43-dim task subspace from the 72 task-specific FVs (CPU).

Basis = QR-orthonormalized [unit(mean FV); top-42 CENTERED PCs] of the 72 train tasks'
task-specific FVs (each task's own diag_headhunger c>0.8 head set, top-10 fallback,
unweighted sum of W_O-projected 10-shot head means, fp64). 42 = PCs for >=90% centered
variance. Gates: basis orthonormal; mean FV and every task FV's centered component
reconstruct within the span as expected; k=42 recomputed from the spectrum.
Output: artifacts/sandbox/ext_steerability/pca_subspace43.pt
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT

OUT = ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "pca_subspace43.pt"


def main():
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_90.json"))
    train_tasks = split["train_tasks"]
    hf = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    sd = torch.load(glob.glob(str(hf / "hub/models--EleutherAI--gpt-j-6b/snapshots/*/pytorch_model.bin"))[0],
                    map_location="cpu", weights_only=True, mmap=True)
    ar = ARTIFACTS_ROOT / "sandbox" / "ext_steerability"
    fvs, nsel = [], {}
    for t in train_tasks:
        c = torch.load(ar / t / "diag_headhunger_c.pt", weights_only=False)["c"]
        sel = torch.nonzero(c > 0.8).flatten().tolist() or \
            torch.argsort(c, descending=True)[:10].tolist()
        nsel[t] = len(sel)
        hm = torch.load(ar / t / "means.pt", weights_only=False)["head_means"].double()
        v = torch.zeros(4096, dtype=torch.float64)
        for i in sel:
            l, h = i // 16, i % 16
            w = sd[f"transformer.h.{l}.attn.out_proj.weight"].double()[:, h * 256:(h + 1) * 256]
            v += w @ hm[l, h]
        fvs.append(v)
    X = torch.stack(fvs)  # (72, 4096) fp64
    mu = X.mean(dim=0)
    Xc = (X - mu).numpy()
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    assert k90 == 42, f"k90 gate: expected 42, got {k90}"

    raw = torch.cat([(mu / mu.norm()).unsqueeze(0), torch.from_numpy(vt[:42].copy())], dim=0)  # (43, 4096)
    Q, _ = torch.linalg.qr(raw.T)  # (4096, 43), columns orthonormal, col0 = unit(mu)
    B = Q.T  # (43, 4096)
    assert (B @ B.T - torch.eye(43, dtype=torch.float64)).abs().max() < 1e-10
    assert torch.allclose(B[0], mu / mu.norm(), atol=1e-10) or \
        torch.allclose(B[0], -mu / mu.norm(), atol=1e-10)
    # span gate: mu reconstructs exactly; task FVs retain their (mean + top-42) energy
    rec = B.T @ (B @ mu)
    assert (rec - mu).norm() / mu.norm() < 1e-10
    retained = [float((B.T @ (B @ v)).norm() ** 2 / v.norm() ** 2) for v in X]
    print(f"basis (43,4096) built; FV energy retained in span: "
          f"min={min(retained):.4f} median={float(np.median(retained)):.4f}")

    torch.save({"sandbox": True, "B": B, "U42": torch.from_numpy(vt[:42].copy()), "mu_fv": mu,
                "singular_values": torch.from_numpy(s),
                "tasks": train_tasks, "n_selected": nsel, "k_pcs": 42,
                "note": "task subspace pieces: mu_fv = mean task-specific FV; U42 = raw top-42 "
                        "CENTERED PCs (orthonormal); B = QR[unit(mu); U42] (43-dim span, kept "
                        "for reference). Steering construction (user 2026-08-15): "
                        "v_A(l) = mu_fv + U42^T U42 (zbar_A(l) - mu_fv)."}, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
