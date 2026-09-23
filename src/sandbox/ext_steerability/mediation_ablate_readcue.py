#!/usr/bin/env python
"""Mediation (a), reviewer T2.4: does ablating the identification direction lower execution alignment at the cue?

Clean n-shot prompts (the paper's identification-ablation bank and protocol: ablate_readdir_pc5.py with the rank-1
meanresid_top1 bases, i.e. u_hat_A, at EVERY demo-label token entering EVERY block, prefill), but instead of sampling we
read the residual stream at the final query cue at every block output and record its cosine with the task FV v_A (mean
per-prompt FV, as in the paper's alignment readouts) and with the generic FV (mean of v_A over the pool).

Conditions: none | own_mean | own_zero | cf_mean | cf_zero   (cf = the paired other-family task's u_hat, as in the paper).
Forward passes only. Output: <out_root>/<task>.pt  {cos_task, cos_gen: (n_cond, n_prompts, n_layers)}.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT  # noqa: E402
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import Ablator, batches_by_len  # noqa: E402
from src.sandbox.ext_steerability.ablate_readdir_labeltokens import (  # noqa: E402
    load_model_eager, make_batch, prep_task_nshot, verify_ablation)
from src.sandbox.ext_steerability.steer_effect_on_cue import CueReader  # noqa: E402

CONDS = ("none", "own_mean", "own_zero", "cf_mean", "cf_zero")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_shots", type=int, default=6, choices=(1, 6))
    p.add_argument("--prompts_root", type=Path, default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--bases_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA" / "meanresid_top1_bases.pt")
    p.add_argument("--grand_mean_path", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "pc50_ablation" / "grand_mean69.pt")
    p.add_argument("--pairs_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "cf_task_pairs.json")
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--out_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "mediation" / "ablate_readcue")
    p.add_argument("--split_path", type=Path, default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--model_name", default=None)
    p.add_argument("--print_layer", type=int, default=13)
    p.add_argument("--token_budget", type=int, default=11000)
    p.add_argument("--batch_cap", type=int, default=16)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    all_tasks = sorted(group)
    tasks = all_tasks[args.shard_idx::args.shard_n][:args.max_tasks]
    out = args.out_root / f"n{args.n_shots}shot"
    out.mkdir(parents=True, exist_ok=True)
    pairs = json.load(open(args.pairs_path))["pairs"]
    bases = torch.load(args.bases_path, map_location="cpu", weights_only=False)["tasks"]
    grand = torch.load(args.grand_mean_path, map_location="cpu", weights_only=False)["mean"].float().cuda()
    fvs = {t: torch.load(args.fv_root / f"{t}.pt", map_location="cpu", weights_only=False)["fv"].float().mean(0)
           for t in all_tasks}
    vg = torch.stack([fvs[t] for t in all_tasks]).mean(0).cuda()

    model, tok = load_model_eager(args.model_dir, args.model_name)
    tok.padding_side = "left"
    ab = Ablator(model)
    reader = CueReader(model)
    nL, D = reader.n_layers, reader.d
    PL = args.print_layer
    verified = False
    for task in tasks:
        op = out / f"{task}.pt"
        if op.exists():
            print(f"{task}: exists, skip", flush=True); continue
        items = prep_task_nshot(task, args.prompts_root, tok, args.n_shots)
        Vo, Vc = bases[task]["V"].float().cuda(), bases[pairs[task]]["V"].float().cuda()
        assert Vo.shape[0] == 1
        setup = {"none": (None, None), "own_mean": (Vo, (grand @ Vo.T) @ Vo), "own_zero": (Vo, None),
                 "cf_mean": (Vc, (grand @ Vc.T) @ Vc), "cf_zero": (Vc, None)}
        if not verified:
            verify_ablation(model, ab, tok, items, Vo, (grand @ Vo.T) @ Vo, args.token_budget, args.batch_cap)
            verified = True
        vt = fvs[task].cuda()
        n = len(items)
        res = {k: torch.zeros(len(CONDS), n, nL) for k in ("cos_task", "cos_gen")}
        for ci, c in enumerate(CONDS):
            V, mproj = setup[c]
            for b in batches_by_len(items, args.token_budget, args.batch_cap):
                ids, att, mask, _ = make_batch(items, b, tok)
                ab.V, ab.mproj, ab.mask = V, mproj, (mask if V is not None else None)
                reader.last_idx = torch.full((len(b),), ids.shape[1] - 1, device="cuda", dtype=torch.long)
                reader.buf = torch.zeros(len(b), nL, D, device="cuda")
                with torch.no_grad():
                    model(input_ids=ids, attention_mask=att, use_cache=False)
                acts = reader.buf
                ab.V = ab.mproj = ab.mask = None
                reader.last_idx = reader.buf = None
                ct = torch.nn.functional.cosine_similarity(acts, vt.view(1, 1, -1), dim=2).cpu()
                cg = torch.nn.functional.cosine_similarity(acts, vg.view(1, 1, -1), dim=2).cpu()
                for r, i in enumerate(b):
                    res["cos_task"][ci, i] = ct[r]; res["cos_gen"][ci, i] = cg[r]
            print(f"{task} | {c}: L{PL} cos_task={res['cos_task'][ci, :, PL].mean():.4f} "
                  f"cos_gen={res['cos_gen'][ci, :, PL].mean():.4f}", flush=True)
        res.update({"task": task, "group": group[task], "cf_task": pairs[task], "conditions": list(CONDS),
                    "n_shots": args.n_shots, "n_prompts": n, "model_name": args.model_name or "EleutherAI/gpt-j-6b",
                    "site": "final query cue, all block outputs; ablation at all demo-label tokens entering all blocks"})
        torch.save(res, op)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
