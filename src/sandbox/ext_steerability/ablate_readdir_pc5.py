#!/usr/bin/env python
"""Multi-direction (rank-5) bottom-up read-feature ablation.

Same protocol as ablate_readdir_labeltokens.py (n-shot truncated 150-prompt bank, edits at
EVERY demo-label token entering EVERY block, T=1 sampled exact-match, same seeds scheme),
but the ablated object is the task's top-5 UNCENTERED PCs of its per-prompt read-feature
activations (build_readdir_pc5_bases.py; PC1 ~ the single mean direction, cos >= 0.98
asserted, so this strictly extends the rank-1 run). Per user 2026-08-20:
  zero : h <- h - P_V h                (all 5 projections removed)
  mean : h <- h - P_V h + P_V m_l     (all 5 projections replaced by the cross-task
                                       grand mean's projections, per layer)
Counterfactual arms use the cf task's OWN 5-PC basis (same cf_task_pairs.json pairing).

Conditions per n_shots in {1, 6}: mean_ablation_pc5, zero_ablation_pc5,
cf_mean_ablation_pc5, cf_zero_ablation_pc5.
Output: <out_root>/pc5/n{n}shot/<task>.json  (resumable, missing conditions filled in).
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import Ablator, batches_by_len
from src.sandbox.ext_steerability.ablate_readdir_labeltokens import (
    load_model_eager, make_batch, prep_task_nshot, verify_ablation)

CONDITIONS = ("mean_ablation_pc5", "zero_ablation_pc5",
              "cf_mean_ablation_pc5", "cf_zero_ablation_pc5")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_shots", type=int, required=True, choices=(1, 6))
    p.add_argument("--conditions", type=str, default=",".join(CONDITIONS))
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--bases_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "pc5_bases.pt")
    p.add_argument("--grand_mean_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "pc50_ablation" / "grand_mean69.pt")
    p.add_argument("--pairs_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "cf_task_pairs.json")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=11000)
    p.add_argument("--batch_cap", type=int, default=16)
    p.add_argument("--out_sub", type=str, default="pc5",
                   help="output subdir under out_root (e.g. pc5_centered for the "
                        "centered-PCA bases variant)")
    p.add_argument("--task_set", choices=("train", "heldout", "all"), default="all")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    conds = [c for c in args.conditions.split(",") if c]
    assert all(c in CONDITIONS for c in conds), conds
    split = json.load(open(args.split_path))
    pool = {"train": split["train_tasks"], "heldout": split["heldout_tasks"],
            "all": split["train_tasks"] + split["heldout_tasks"]}[args.task_set]
    tasks = sorted(pool)[args.shard_idx::args.shard_n]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    pairs = json.load(open(args.pairs_path))["pairs"]
    bases = torch.load(args.bases_path, map_location="cpu", weights_only=False)["tasks"]
    for t, b in bases.items():
        # uncentered bases: PC1 must be the mean direction; centered bases carry
        # mean_frac_in_V instead and deliberately exclude the mean.
        if "cos_pc1_mean" in b:
            assert b["cos_pc1_mean"] >= 0.98, \
                f"{t}: PC1 drifted from mean ({b['cos_pc1_mean']:.3f})"
    grand = torch.load(args.grand_mean_path, map_location="cpu",
                       weights_only=False)["mean"].float().cuda()   # (28, 4096)

    model, tok = load_model_eager(args.model_dir)
    ab = Ablator(model)
    tok.padding_side = "left"
    outdir = args.out_root / args.out_sub / f"n{args.n_shots}shot"
    outdir.mkdir(parents=True, exist_ok=True)

    verified = False
    for task in tasks:
        outpath = outdir / f"{task}.json"
        res = json.load(open(outpath)) if outpath.exists() else None
        todo = [c for c in conds if res is None or c not in res["conditions"]]
        if not todo:
            print(f"{task}: all conditions present, skip", flush=True)
            continue
        items = prep_task_nshot(task, args.prompts_root, tok, args.n_shots)
        V_own = bases[task]["V"].cuda()          # (5, 4096)
        V_cf = bases[pairs[task]]["V"].cuda()
        setup = {
            "mean_ablation_pc5":    (V_own, (grand @ V_own.T) @ V_own),
            "zero_ablation_pc5":    (V_own, None),
            "cf_mean_ablation_pc5": (V_cf, (grand @ V_cf.T) @ V_cf),
            "cf_zero_ablation_pc5": (V_cf, None),
        }
        if not verified:
            # verify_ablation probes the projection onto V[0] (=~ the mean direction);
            # run it for both modes, then a full-subspace residue check via zero mode.
            verify_ablation(model, ab, tok, items, V_own,
                            (grand @ V_own.T) @ V_own, args.token_budget, args.batch_cap)
            verify_ablation(model, ab, tok, items, V_own, None,
                            args.token_budget, args.batch_cap)
            verified = True
        if res is None:
            res = {"task": task, "n_shots": args.n_shots, "cf_task": pairs[task],
                   "rank": 5, "bases_path": str(args.bases_path),
                   "n_prompts": len(items), "conditions": {},
                   "golds": [it["gold"] for it in items]}
        for cname in todo:
            V, mproj = setup[cname]
            preds = [None] * len(items)
            for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
                ids, att, mask, _ = make_batch(items, b, tok)
                ab.V, ab.mproj, ab.mask = V, mproj, mask
                max_new = min(max(items[i]["gold_len"] for i in b) + 3, 16)
                torch.manual_seed(zlib.crc32(f"{task}|{cname}|{bi}".encode()))
                with torch.no_grad():
                    gen = model.generate(input_ids=ids, attention_mask=att,
                                         do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                         max_new_tokens=max_new,
                                         pad_token_id=tok.eos_token_id)
                ab.V = ab.mproj = ab.mask = None
                for r, i in enumerate(b):
                    preds[i] = tok.decode(gen[r, ids.shape[1]:],
                                          skip_special_tokens=True).split("\n")[0].strip()
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | n{args.n_shots} | {cname}: acc={acc:.3f}", flush=True)
        with open(outpath, "w") as f:
            json.dump(res, f)
    print("eval done", flush=True)


if __name__ == "__main__":
    main()
