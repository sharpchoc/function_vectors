#!/usr/bin/env python
"""Build a task's attention_head_payload_subspace from its top-CIE heads' d_payload vectors.

For a task's top-N per-task-CIE heads (from <task>_cie_result.pt), the value-channel payload
direction of head (L, H) is

    d_payload = unit( W_V[L,H]^T @ unit(z_bar[L,H]) )

with z_bar the task-mean head activation (single cue-token position, ICL-correct prompts).
d_payload is prompt-independent (task means + weights only) and fully position-free (RoPE
never touches V). The payload subspace = top-k right singular vectors of the stacked
(N, 4096) unit d_payload matrix (UNcentered SVD, fp64 CPU).

Prints the dimensionality stats (stable rank raw/unit + pairwise cos) for both the d_payload
stack and the out_proj-projected mean outputs, and (unless --stats_only) caches
{basis (k, 4096) orthonormal, singular_values, d_payloads, heads, metadata} to
artifacts/payload_subspaces/<task>_top<N>heads_k<k>.pt.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.utils.paths import ARTIFACTS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description="Build/cache a task's attention_head_payload_subspace.")
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Multiple tasks in one invocation (single model load).")
    p.add_argument("--head_source", choices=["pertask", "pooled40"], default="pertask",
                   help="pertask: the task's own top-N per-task-CIE heads. pooled40: the "
                        "canonical pooled top-40 train-selected head list (same for every "
                        "task; artifact suffix becomes pooled40heads_k<k>).")
    p.add_argument("--pooled_heads_path", type=Path,
                   default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl" / "multitask_top_aie_heads.pt")
    p.add_argument("--n_heads", type=int, default=10, help="Top-N per-task CIE heads (pertask mode).")
    p.add_argument("--k", type=int, default=4, help="Subspace dimension (top-k SVD directions).")
    p.add_argument("--ks", nargs="+", type=int, default=None,
                   help="Multiple k values in one invocation (one SVD per task, one artifact "
                        "per k). Overrides --k.")
    p.add_argument("--stats_only", action="store_true",
                   help="Print dimensionality stats only; do not build/cache the subspace.")
    p.add_argument("--cie_weight", action="store_true",
                   help="Scale each unit d_payload row to norm 100*CIE (pooled train CIE in "
                        "pooled40 mode, per-task CIE otherwise) before the SVD, so high-CIE "
                        "heads dominate the basis. The global 100x is cosmetic (SVD directions "
                        "and energy fractions depend only on relative weights). Artifact "
                        "suffix gains a 'ciew' marker.")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--aie_root", type=Path, default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl")
    p.add_argument("--out_root", type=Path, default=ARTIFACTS_ROOT / "payload_subspaces")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def stable_rank(M):
    s = torch.linalg.svdvals(M)
    return float((s ** 2).sum() / s[0] ** 2)


def pair_stats(M):
    U = M / M.norm(dim=1, keepdim=True)
    C = U @ U.T
    iu = torch.triu_indices(len(M), len(M), offset=1)
    p = C[iu[0], iu[1]]
    return float(p.mean()), float(p.median()), float(p.min()), float(p.max())


def main():
    args = parse_args()
    tasks = list(args.tasks) if args.tasks is not None else [args.task]
    assert tasks and tasks[0] is not None, "pass --task or --tasks"

    from transformers import AutoModelForCausalLM
    print("Loading model (float32 on CPU / native on GPU)...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    model.eval()
    for task in tasks:
        process_task(task, args, model)


def process_task(task, args, model):
    ks = list(args.ks) if args.ks is not None else [args.k]
    if args.head_source == "pooled40":
        pooled = torch.load(args.pooled_heads_path, weights_only=False)["top_heads"]
        heads = [(int(l), int(h)) for l, h, _ in pooled]
        cie_scores = [float(s) for _, _, s in pooled]
        suffix_base = "pooled40heads_ciew_k" if args.cie_weight else "pooled40heads_k"
        heads_desc = "pooled top-40 train-selected heads"
    else:
        cie = torch.load(args.aie_root / task / f"{task}_cie_result.pt", weights_only=False)
        heads = [(int(l), int(h)) for l, h, _ in cie["top_heads"][: args.n_heads]]
        cie_scores = [float(s) for _, _, s in cie["top_heads"][: args.n_heads]]
        suffix_base = (f"top{args.n_heads}heads_ciew_k" if args.cie_weight
                       else f"top{args.n_heads}heads_k")
        heads_desc = f"task's top-{args.n_heads} per-task-CIE heads"
    out_paths = {k: args.out_root / f"{task}_{suffix_base}{k}.pt" for k in ks}
    if not args.stats_only and not args.overwrite:
        for p in out_paths.values():
            if p.exists():
                raise FileExistsError(f"{p} exists. Pass --overwrite to rebuild.")

    mean_acts = torch.load(
        args.aie_root / task / f"{task}_mean_head_activations_varicl.pt",
        weights_only=False,
    )
    print(f"\n{task}: {heads_desc} ({len(heads)}): "
          + " ".join(f"L{l}H{h}" for l, h in heads[:10]) + (" ..." if len(heads) > 10 else ""))

    HD = mean_acts.shape[-1]
    payload_rows, output_rows = [], []
    with torch.no_grad():
        for l, h in heads:
            attn = model.transformer.h[l].attn
            z = mean_acts[l, h].to(torch.float64)
            m_hat = z / z.norm()
            w_v = attn.v_proj.weight[h * HD:(h + 1) * HD].double()      # (256, 4096)
            w_o = attn.out_proj.weight[:, h * HD:(h + 1) * HD].double() # (4096, 256)
            d = w_v.T @ m_hat
            payload_rows.append(d / d.norm())
            output_rows.append(w_o @ z)
    D = torch.stack(payload_rows)   # (N, 4096) unit rows
    A = torch.stack(output_rows)    # (N, 4096) raw FV contributions

    print(f"\n[d_payload stack, unit rows]  stable rank {stable_rank(D):.3f}", end="")
    m, md, mn, mx = pair_stats(D)
    print(f"   pairwise cos mean {m:.4f} median {md:.4f} min {mn:.4f} max {mx:.4f}")
    print(f"[mean outputs, raw norms]     stable rank {stable_rank(A):.3f}   "
          f"(norms {', '.join(f'{float(x):.1f}' for x in A.norm(dim=1))})")
    Au = A / A.norm(dim=1, keepdim=True)
    m, md, mn, mx = pair_stats(A)
    print(f"[mean outputs, unit rows]     stable rank {stable_rank(Au):.3f}   "
          f"pairwise cos mean {m:.4f} median {md:.4f} min {mn:.4f} max {mx:.4f}")

    if args.stats_only:
        print("\n--stats_only: no subspace built.")
        return

    # UNcentered SVD of the payload stack; top-k right singular vectors = the subspace.
    # With --cie_weight, row i is scaled to norm 100*CIE_i so high-CIE heads dominate.
    if args.cie_weight:
        weights = 100.0 * torch.tensor(cie_scores, dtype=D.dtype)
        stack = D * weights[:, None]
        print(f"CIE weighting: row norms {float(weights.min()):.3f}..{float(weights.max()):.3f}")
    else:
        weights = None
        stack = D
    U_, S, Vh = torch.linalg.svd(stack, full_matrices=False)
    args.out_root.mkdir(parents=True, exist_ok=True)
    for k in ks:
        basis = Vh[:k]                                     # (k, 4096), orthonormal rows
        gram_dev = (basis @ basis.T - torch.eye(k, dtype=basis.dtype)).abs().max().item()
        assert gram_dev < 1e-10, f"basis not orthonormal: {gram_dev:.2e}"
        energy = float((S[:k] ** 2).sum() / (S ** 2).sum())
        print(f"\nsubspace k={k}: singular values "
              f"{[round(float(s), 3) for s in S[:k]]} "
              f"({energy:.1%} of stack energy); orthonormality dev {gram_dev:.1e}")
        cover = (D @ basis.T).norm(dim=1) ** 2
        print("per-head coverage ||proj||^2: "
              + "  ".join(f"L{l}H{h}:{float(c):.2f}" for (l, h), c in zip(heads, cover)))
        torch.save(
            {
                "name": "attention_head_payload_subspace",
                "task": task,
                "basis": basis,                   # (k, 4096) fp64, orthonormal rows
                "singular_values": S,             # all N singular values of the stack
                "d_payloads": D,                  # (N, 4096) unit rows, head order below
                "heads": heads,                   # [(layer, head)]
                "cie_scores": cie_scores,
                "k": k,
                "head_source": args.head_source,
                "cie_weights": None if weights is None else weights,
                "definition": "top-k right singular vectors (uncentered SVD, fp64) of stacked "
                              + ("CIE-weighted (row norm = 100*pooled-train-CIE) " if args.cie_weight
                                 else "unit ")
                              + "d_payload = unit(W_V^T @ unit(task-mean head activation)); "
                              f"heads = {heads_desc}",
                "mean_activations_path": str(args.aie_root / task /
                                             f"{task}_mean_head_activations_varicl.pt"),
                "model_name": args.model_name,
                "built": str(date.today()),
            },
            out_paths[k],
        )
        print(f"cached -> {out_paths[k]}")


if __name__ == "__main__":
    main()
