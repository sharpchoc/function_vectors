#!/usr/bin/env python
"""SANDBOX: steering eval for the PCA-subspace mean-activation construction (ext tasks).

Arms per task (both from the stored per-layer cue-token activation means zbar_A(l),
28x4096, over the 150 train prompts):
  pcasub: v_A(l) = mu_fv + U42^T U42 (zbar_A(l) - mu_fv)   [user construction 2026-08-15:
          mean task-specific FV always fully included; deviations of the activation mean
          from it added along the 42 centered FV-PCs]
  rawmean: v_A(l) = zbar_A(l)                               [unprojected control]
Injected at layer l (each layer its own vector), alpha in --alphas, settings
{test_zeroshot, test_mixedtask10}, full-label acc on the paired test queries.
Writes <task>/diag_pcasub.json. Works for train AND heldout tasks.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.sandbox.isolation_upper_bound.run_task import (
    eval_points_fixed_v,
    load_records,
    record_to_point,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.paths import ARTIFACTS_ROOT

OUT_ROOT = ARTIFACTS_ROOT / "sandbox" / "ext_steerability"
SETTINGS = ["test_zeroshot", "test_mixedtask10"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--prompts_root", type=Path, default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path, default=OUT_ROOT)
    p.add_argument("--basis_path", type=Path, default=OUT_ROOT / "pca_subspace43.pt")
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    basis = torch.load(args.basis_path, map_location="cpu", weights_only=False)
    U = basis["U42"].float()          # (42, 4096)
    mu = basis["mu_fv"].float()       # (4096,)

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model = model.to(torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.set_grad_enabled(False)
    n_layers = model_config["n_layers"]
    dev = model.device
    U_d, mu_d = U.to(dev), mu.to(dev)

    for t in args.tasks:
        out = args.out_root / t / "diag_pcasub.json"
        if out.exists():
            print(f"[{t}] exists, skip", flush=True)
            continue
        zbar = torch.load(args.out_root / t / "means.pt", map_location="cpu",
                          weights_only=False)["resid_means"].float().to(dev)  # (28, 4096)
        arms = {}
        for l in range(n_layers):
            dev_vec = zbar[l] - mu_d
            arms.setdefault("pcasub", []).append(mu_d + U_d.T @ (U_d @ dev_vec))
            arms.setdefault("rawmean", []).append(zbar[l])
        results = {}
        for setting in SETTINGS:
            recs = load_records(args, t, setting)
            points = [record_to_point(r, tokenizer, model_config) for r in recs]
            baseline = eval_points_fixed_v(model, model_config, tokenizer, points, None, 9)
            entry = {"baseline": baseline, "arms": {}}
            for arm, vecs in arms.items():
                for a in args.alphas:
                    accs = [eval_points_fixed_v(model, model_config, tokenizer, points,
                                                (a * vecs[L]).float(), L)
                            for L in range(n_layers)]
                    entry["arms"][f"{arm}|a{a:g}"] = accs
            results[setting] = entry
            best = {k: max(v) for k, v in entry["arms"].items()}
            print(f"[{t}] {setting}: base={baseline:.2f} " +
                  " ".join(f"{k}={v:.2f}" for k, v in best.items()), flush=True)
        with open(out, "w") as f:
            json.dump({"task": t, "construction": basis["note"], "alphas": args.alphas,
                       "settings": results}, f, indent=1)
    print("PCASUB EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
