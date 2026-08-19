#!/usr/bin/env python
"""SANDBOX: steering accuracy of the sparse23 FV projected onto the top-29 selected PCs.

Mirrors the sparse_head_selection loto_vs_canonical comparison protocol exactly: the same
1720 zero-shot datapoints (valid cap 100 / min 80, train top-up, seed 42), injection ONCE
at the cue token at the output of block 9, full-label teacher-forced accuracy, per-task
table over the 20 train tasks.

Arms (all evaluated in this one run, same GPU):
  - no_intervention
  - full sparse23 FV (fixed10 capture mean v_t, unprojected)
  - v_t projected onto span of the TOP-29 PCs (final c > 0.8 from train_sparse_pcs) --
    pure orthogonal projection, unweighted (c=1 on the 29, 0 elsewhere)
  - v_t projected onto all 83 PCs (c=1) as the parametrization ceiling

Advisory gate: no_intervention / full FV / proj83 accs are compared to the stored
baselines.json (computed on the same GPU model); mismatches are printed, not fatal
(cross-run fp noise rule 2026-07-16).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.sandbox.sparse_head_selection.train_sparse_heads import (
    build_task_datapoints,
    evaluate_points,
)
from src.sandbox.sparse_head_selection.train_sparse_pcs import (
    build_pc_contributions,
    consistency_check_pc,
    evaluate_points_v,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT

DEFAULT_ARTIFACT_ROOT = ARTIFACTS_ROOT / "sandbox" / "sparse_pc_selection"
DEFAULT_RESULTS_ROOT = RESULTS_ROOT / "sandbox" / "sparse_pc_selection"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis_path", type=Path, default=DEFAULT_ARTIFACT_ROOT / "pc_basis_83.pt")
    p.add_argument("--coeffs_path", type=Path, default=DEFAULT_ARTIFACT_ROOT / "coeffs_final.pt")
    p.add_argument("--baselines_path", type=Path, default=DEFAULT_ARTIFACT_ROOT / "baselines.json")
    p.add_argument("--c_high", type=float, default=0.8, help="PC-selection cut (top-29 = c > 0.8).")
    p.add_argument("--task_split_path", type=Path,
                   default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--root_data_dir", type=str, default=str(REPO_ROOT / "dataset_files"))
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--inject_layer", type=int, default=9)
    p.add_argument("--max_queries", type=int, default=100)
    p.add_argument("--min_queries", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--task_split_key", choices=["train_tasks", "test_tasks"], default="train_tasks")
    p.add_argument("--v_means_capture_dir", type=Path, default=None,
                   help="Compute task mean FVs from this fixed10 capture dir (fp64, checkpoint "
                        "W_O slices - identical recipe to build_fv_pc_basis) for tasks missing "
                        "from the basis, e.g. the held-out test tasks.")
    p.add_argument("--out_tag", type=str, default="",
                   help="Suffix for output filenames (e.g. _testtasks).")
    p.add_argument("--output_root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    p.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return p.parse_args()


def v_means_from_capture(capture_dir, heads, tasks):
    """Fixed10-capture mean FVs, built exactly like build_fv_pc_basis (fp64, mmap'd W_O)."""
    import glob as _glob
    import os as _os
    hd = 256
    hf_home = Path(_os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    bins = _glob.glob(str(hf_home / "hub" / "models--EleutherAI--gpt-j-6b" /
                          "snapshots" / "*" / "pytorch_model.bin"))
    assert bins, "GPT-J pytorch_model.bin not found in HF cache"
    sd = torch.load(bins[0], map_location="cpu", weights_only=True, mmap=True)
    w_o = {(l, h): sd[f"transformer.h.{l}.attn.out_proj.weight"][:, h * hd:(h + 1) * hd].double()
           for l, h in heads}
    del sd
    out = {}
    for task in tasks:
        acts = torch.load(capture_dir / f"{task}.pt", weights_only=False)["activations"].double()
        assert acts.shape[0] == 170, f"{task}: {acts.shape[0]} prompts"
        fvs = torch.zeros(acts.shape[0], 4096, dtype=torch.float64)
        for (l, h) in heads:
            fvs += acts[:, l, h] @ w_o[(l, h)].T
        out[task] = fvs.mean(dim=0)
    return out


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.tasks:
        tasks = list(args.tasks)
    else:
        with open(args.task_split_path) as f:
            tasks = list(json.load(f)[args.task_split_key])
    print(f"tasks ({len(tasks)}): {tasks}")

    basis = torch.load(args.basis_path, map_location="cpu", weights_only=False)
    missing = [t for t in tasks if t not in basis["v_means"]]
    if missing:
        assert args.v_means_capture_dir is not None, \
            f"tasks missing from basis ({missing}) - pass --v_means_capture_dir"
        print(f"computing v_means from capture for: {missing}")
        basis["v_means"].update(
            v_means_from_capture(args.v_means_capture_dir, basis["heads"], missing))
    c_final = torch.load(args.coeffs_path, map_location="cpu", weights_only=False)["c"]
    sel = torch.nonzero(c_final > args.c_high).flatten().tolist()
    print(f"selected PCs (c > {args.c_high}): n={len(sel)} -> {sel}")

    # Subspace containment of each task FV (basis was fit on TRAIN tasks only, so this is
    # the geometric generalization diagnostic for held-out tasks).
    U = basis["U"]
    U29 = U[sel]
    for t in tasks:
        v = basis["v_means"][t]
        e83 = float((U @ v).norm() ** 2 / v.norm() ** 2)
        e29 = float((U29 @ v).norm() ** 2 / v.norm() ** 2)
        print(f"  energy retained {t:28s} 83-PC: {e83:.4f}   top-{len(sel)}-PC: {e29:.4f}")

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    C = build_pc_contributions(basis, tasks, model.device)
    consistency_check_pc(basis, tasks, C)

    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    task_index = {t: i for i, t in enumerate(tasks)}

    print("building datapoints ...")
    points_by_task = {t: build_task_datapoints(t, args, tokenizer, model_config) for t in tasks}
    n_total = sum(len(v) for v in points_by_task.values())
    print(f"total datapoints: {n_total}")

    n_pcs = C.shape[1]
    c_sel = torch.zeros(n_pcs, device=C.device)
    c_sel[sel] = 1.0
    ones = torch.ones(n_pcs, device=C.device)
    v_full = {t: basis["v_means"][t].float().to(C.device) for t in tasks}

    rows = []
    with torch.no_grad():
        for task in tasks:
            pts = points_by_task[task]
            _, acc0 = evaluate_points(model, model_config, tokenizer, pts, C, task_index, None, args)
            _, accf = evaluate_points_v(model, model_config, tokenizer, pts, v_full, args)
            _, acc29 = evaluate_points(model, model_config, tokenizer, pts, C, task_index, c_sel, args)
            _, acc83 = evaluate_points(model, model_config, tokenizer, pts, C, task_index, ones, args)
            rows.append({"task": task, "no_intervention": acc0, "full_sparse23_fv_L9": accf,
                         f"top{len(sel)}pc_proj_L9": acc29, "proj83_c1_L9": acc83})
            print(f"{task:28s} none={acc0:.3f} fullFV={accf:.3f} "
                  f"top{len(sel)}proj={acc29:.3f} proj83={acc83:.3f}", flush=True)

    keys = [k for k in rows[0] if k != "task"]
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print("MEANS: " + "  ".join(f"{k}={v:.3f}" for k, v in means.items()))

    # Advisory reproduction check vs the stored reduce-time baselines (same GPU model);
    # only covers tasks present there (i.e. train tasks).
    with open(args.baselines_path) as f:
        stored = json.load(f)
    dev, n_checked = [], 0
    for r in rows:
        if r["task"] not in stored:
            continue
        for new_k, old_k in (("no_intervention", "no_intervention"),
                             ("full_sparse23_fv_L9", "full_fv_fixed10"),
                             ("proj83_c1_L9", "proj83_c1")):
            n_checked += 1
            d = abs(r[new_k] - stored[r["task"]][old_k]["acc"])
            if d > 1e-9:
                dev.append((r["task"], new_k, r[new_k], stored[r["task"]][old_k]["acc"]))
    print(f"advisory reproduction check vs stored baselines ({n_checked} cells): "
          f"{'EXACT' if not dev else f'{len(dev)} deviations'}")
    for d in dev:
        print("   dev:", d)

    args.results_root.mkdir(parents=True, exist_ok=True)
    out_csv = args.results_root / f"top{len(sel)}pc_projection_vs_fullfv{args.out_tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task"] + keys)
        w.writeheader()
        w.writerows(rows)
        w.writerow({"task": "MEAN", **{k: round(means[k], 4) for k in keys}})
    print(f"wrote {out_csv}")

    with open(args.output_root / f"top{len(sel)}pc_projection_eval{args.out_tag}.json", "w") as f:
        json.dump({"sandbox": True, "selected_pcs": sel, "c_high": args.c_high,
                   "inject_layer": args.inject_layer, "rows": rows, "means": means,
                   "reproduction_deviations": dev}, f, indent=2)


if __name__ == "__main__":
    main()
