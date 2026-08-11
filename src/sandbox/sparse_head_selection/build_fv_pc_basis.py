#!/usr/bin/env python
"""SANDBOX: build the 83-PC basis of the pooled per-prompt FV stack (sparse23 heads).

Stacks the 20 train tasks' fixed10 per-prompt FVs v^j_t = sum_{h in sparse23} W_O h(p^j_t)
(20 x 170 = 3400 x 4096, fp64), runs an UNCENTERED SVD (CPU LAPACK), and saves the top-83
right singular vectors U plus each task's fixed10 mean FV v_t. 83 = # PCs for >=90% of the
pooled variance (15 for 80%) as verified against the part-14c per-task pipeline.

Hard gates (mismatch = stop, user adjudicates):
  - npc80/npc90 recomputed from the spectrum must equal 15/83.
  - Every v_t must equal the mean of that task's 170 stacked rows (exact by construction).

CPU-only; W_O slices are mmap'd from the GPT-J checkpoint in the HF cache.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT

HD = 256
RESID = 4096
N_PROMPTS = 170


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture_dir", type=Path,
                   default=ARTIFACTS_ROOT / "perprompt_head_activations" / "gptj_27tasks_170prompts" / "fixed10")
    p.add_argument("--heads_path", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "sparse_head_selection" / "vanilla_sparse_opt23_heads.pt")
    p.add_argument("--split_metadata_path", type=Path,
                   default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl" / "multitask_top_aie_heads_metadata.json")
    p.add_argument("--n_pcs", type=int, default=83)
    p.add_argument("--expected_npc80", type=int, default=15)
    p.add_argument("--expected_npc90", type=int, default=83)
    p.add_argument("--output_root", type=Path, default=ARTIFACTS_ROOT / "sandbox" / "sparse_pc_selection")
    return p.parse_args()


def n_pc_for_energy(s2, frac):
    cum = np.cumsum(s2) / s2.sum()
    return int(np.searchsorted(cum, frac) + 1)


def main():
    args = parse_args()
    with open(args.split_metadata_path) as f:
        train_tasks = sorted(json.load(f)["split_metadata"]["train_tasks"])
    assert len(train_tasks) == 20

    heads = [(int(l), int(h)) for l, h, *_ in
             torch.load(args.heads_path, weights_only=False)["top_heads"]]
    assert len(heads) == 23 and len(set(heads)) == 23

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    bins = glob.glob(str(hf_home / "hub" / "models--EleutherAI--gpt-j-6b" /
                         "snapshots" / "*" / "pytorch_model.bin"))
    assert bins, "GPT-J pytorch_model.bin not found in HF cache"
    sd = torch.load(bins[0], map_location="cpu", weights_only=True, mmap=True)
    w_o = {(l, h): sd[f"transformer.h.{l}.attn.out_proj.weight"][:, h * HD:(h + 1) * HD].double()
           for l, h in heads}
    del sd

    stacks, v_means = [], {}
    for task in train_tasks:
        acts = torch.load(args.capture_dir / f"{task}.pt", weights_only=False)["activations"].double()
        assert acts.shape[0] == N_PROMPTS, f"{task}: {acts.shape[0]} prompts"
        fvs = torch.zeros(N_PROMPTS, RESID, dtype=torch.float64)
        for (l, h) in heads:
            fvs += acts[:, l, h] @ w_o[(l, h)].T
        stacks.append(fvs)
        v_means[task] = fvs.mean(dim=0)
        print(f"{task:28s} mean-FV norm {v_means[task].norm().item():7.2f}")

    X = torch.cat(stacks, dim=0).numpy()
    assert X.shape == (20 * N_PROMPTS, RESID)
    # CPU LAPACK fp64 (the CUDA gesvdj rule is moot here).
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    s2 = s ** 2
    npc80, npc90 = n_pc_for_energy(s2, 0.80), n_pc_for_energy(s2, 0.90)
    print(f"npc80={npc80} npc90={npc90}")
    if (npc80, npc90) != (args.expected_npc80, args.expected_npc90):
        raise RuntimeError(
            f"BASIS GATE FAILED: npc80/npc90 = {npc80}/{npc90}, expected "
            f"{args.expected_npc80}/{args.expected_npc90}. HARD STOP - report to user.")

    U = torch.from_numpy(vt[:args.n_pcs].copy())  # (n_pcs, 4096) fp64, rows orthonormal
    gram_err = (U @ U.T - torch.eye(args.n_pcs, dtype=torch.float64)).abs().max().item()
    assert gram_err < 1e-10, f"U rows not orthonormal (max dev {gram_err:.2e})"

    energy = {
        "pc_energy_frac": (s2[:args.n_pcs] / s2.sum()).tolist(),
        "cum_energy_at_n_pcs": float(s2[:args.n_pcs].sum() / s2.sum()),
        "npc80": npc80, "npc90": npc90,
    }
    retained = {t: float((U @ v_means[t]).norm().item() ** 2 / v_means[t].norm().item() ** 2)
                for t in train_tasks}
    print("per-task FV energy retained by the 83-PC subspace:")
    for t in train_tasks:
        print(f"  {t:28s} {retained[t]:.4f}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    out_pt = args.output_root / f"pc_basis_{args.n_pcs}.pt"
    torch.save({
        "sandbox": True,
        "note": "SANDBOX - basis over sparse23 per-prompt FVs; nothing here is repo standard.",
        "U": U,                                   # (n_pcs, 4096) fp64
        "singular_values": torch.from_numpy(s),   # full spectrum fp64
        "v_means": v_means,                       # {task: (4096,) fp64 fixed10 mean FV}
        "tasks": train_tasks,
        "heads": heads,
        "n_pcs": args.n_pcs,
    }, out_pt)
    meta = {
        "sandbox": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "capture_dir": str(args.capture_dir),
        "heads_path": str(args.heads_path),
        "split_metadata_path": str(args.split_metadata_path),
        "stack_shape": list(X.shape),
        "svd": "uncentered, numpy fp64 CPU LAPACK",
        "fv_definition": "fixed10 capture mean over the same 170 per-prompt FVs the PCA is fit on",
        "energy": energy,
        "fv_energy_retained_by_subspace": retained,
        "gates": {"npc80": npc80, "npc90": npc90,
                  "expected": [args.expected_npc80, args.expected_npc90], "passed": True},
    }
    with open(args.output_root / "pc_basis_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out_pt} and pc_basis_metadata.json")


if __name__ == "__main__":
    main()
