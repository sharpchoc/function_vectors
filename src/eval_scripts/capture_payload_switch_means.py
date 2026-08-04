#!/usr/bin/env python
"""Stage 1 of payload-subspace task-switch steering: mean 10-shot last-demo-label activations.

For each task (default synonym + antonym), load the cached 10-shot residual activations
(gptj_56tasks_170prompts_4tokens; token_role == last_label_token == the 10th/last
demonstration's label token), TRAIN split only, mean over prompts -> mean[29, 4096] fp32
(stack entry 0 = embedding, entry b+1 = output of transformer.h.b). Then project every task's
mean into every task's payload-subspace basis (pooled40heads_k<k>):

    coords[t -> s] = mean_t @ B_s^T        # (29, k), B_s rows orthonormal

Projection of the mean equals the mean of the projections (linear), so these ARE the
"average projections" used as steering targets by steer_payload_switch_logit.py.

Output: artifacts/payload_switch_steering/tenshot_lastlabel_means.pt
Gates: expected row count per task, finiteness, basis orthonormality.
CPU-only; no model load.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.eval_scripts.cosine_activation_to_task_fv import load_split_roles_with_meta
from src.eval_scripts.ablate_oneshot_payload_subspace_logprob import load_subspace
from src.eval_scripts.ablate_oneshot_preimage_logprob import git_commit_hash
from src.utils.paths import ARTIFACTS_ROOT

ROLE = "last_label_token"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=["synonym", "antonym"])
    p.add_argument("--activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations" / "gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train"],
                   help="Capture splits to average (default train only: keeps the dataset's "
                        "test rows out of the steering targets).")
    p.add_argument("--expected_rows", type=int, default=130,
                   help="Expected prompt rows per task (hard gate; 130 for train-only).")
    p.add_argument("--subspace_root", type=Path, default=ARTIFACTS_ROOT / "payload_subspaces")
    p.add_argument("--subspace_suffix", type=str, default="pooled40heads_k4")
    p.add_argument("--out_path", type=Path,
                   default=ARTIFACTS_ROOT / "payload_switch_steering" / "tenshot_lastlabel_means.pt")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.out_path.exists() and not args.overwrite:
        raise SystemExit(f"{args.out_path} exists; pass --overwrite to replace.")

    means = {}
    counts = {}
    for task in args.tasks:
        rows = []
        for split in args.splits:
            per_role = load_split_roles_with_meta(args.activations_root, task, split)
            if ROLE not in per_role:
                raise SystemExit(f"GATE FAIL: role {ROLE!r} missing for {task}/{split}")
            acts, _ = per_role[ROLE]  # (n, 29, 4096) fp32
            rows.append(acts)
        acts = torch.cat(rows, dim=0)
        if acts.shape[0] != args.expected_rows:
            raise SystemExit(f"GATE FAIL: {task}: {acts.shape[0]} rows, expected {args.expected_rows}")
        if not torch.isfinite(acts).all():
            raise SystemExit(f"GATE FAIL: non-finite activations for {task}")
        means[task] = acts.mean(dim=0)  # (29, 4096) fp32
        counts[task] = int(acts.shape[0])
        print(f"{task}: {counts[task]} rows -> mean {tuple(means[task].shape)}")

    bases = {}
    basis_files = {}
    for task in args.tasks:
        basis, sub = load_subspace(args.subspace_root, task, args.subspace_suffix, "cpu")
        bases[task] = basis  # (k, 4096) fp32, orthonormality asserted inside
        basis_files[task] = sub["path"]
        print(f"{task}: basis {tuple(basis.shape)} from {Path(sub['path']).name}")

    coords = {}  # "t->s": (29, k) = task t's mean in task s's basis
    for t in args.tasks:
        for s in args.tasks:
            c = means[t] @ bases[s].T
            if not torch.isfinite(c).all():
                raise SystemExit(f"GATE FAIL: non-finite coords {t}->{s}")
            coords[f"{t}->{s}"] = c

    # Diagnostic: fraction of the mean's norm captured by each subspace, per edit layer.
    for key, c in coords.items():
        t = key.split("->")[0]
        frac = (c.norm(dim=1) / means[t].norm(dim=1)).numpy()
        picks = ", ".join(f"L{b}:{frac[b + 1]:.3f}" for b in (0, 4, 8, 12, 16, 20, 24, 27))
        print(f"|coords|/|mean| {key} (edit layers): {picks}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "name": "payload_switch_tenshot_lastlabel_means",
        "tasks": list(args.tasks),
        "means": {t: means[t] for t in args.tasks},          # (29, 4096) fp32 each
        "coords": coords,                                     # "t->s" -> (29, k) fp32
        "bases": {t: bases[t] for t in args.tasks},           # (k, 4096) fp32 each
        "basis_files": basis_files,
        "counts": counts,
        "role": ROLE,
        "icl_example_index": 10,
        "splits": list(args.splits),
        "activations_root": str(args.activations_root),
        "layer_convention": "stack entry 0 = embedding; entry b+1 = output of transformer.h.b",
        "git_commit": git_commit_hash(),
        "built": date.today().isoformat(),
    }, args.out_path)
    print(f"saved -> {args.out_path}")


if __name__ == "__main__":
    main()
