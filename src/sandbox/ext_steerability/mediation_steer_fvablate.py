#!/usr/bin/env python
"""Mediation (b), reviewer T2.5: does identification-site steering still work when the FV direction is removed at the cue?

Six-dummy-label scaffold (sixshot_dummy_steer.py): alpha * m_A(L_id) added at all six '_' slots at block L_id output
(GPT-J 6, Qwen 12). Simultaneously, the rank-1 component along the task's unit FV (ablate_fv_cue6.unit_fv: selected-head
sum of head means) is removed at the FINAL query cue at the inputs of blocks 9..27 (prefill), as in the paper's execution
ablation. Control: the same ablation with the paired other-family task's FV.

Conditions: unsteered | steer | steer_own_zero | steer_cf_zero  [+ steer_own_mean, steer_cf_mean with --with_mean].
T=1 sampled exact match, seeding crc32(task|cond|batch). Output: <out_root>/<task>.json
"""
import argparse
import json
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT  # noqa: E402
from src.sandbox.ext_steerability.steer_read_dir_1shot import load_model, batches_by_len  # noqa: E402
from src.sandbox.ext_steerability.steer_read_dir_methods import Injector  # noqa: E402
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import load_model as load_model_generic  # noqa: E402
from src.sandbox.ext_steerability.sixshot_dummy_steer import build_items_6shot  # noqa: E402
from src.sandbox.ext_steerability.ablate_fv_cue6 import FVAblator, unit_fv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", type=int, default=6, help="identification steering block (GPT-J 6, Qwen 12)")
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--with_mean", action="store_true", help="also run mean-ablation arms (needs --grand_mean_cue)")
    p.add_argument("--grand_mean_cue", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "FV_ablation" / "grand_mean_cue6.pt")
    p.add_argument("--resid_means_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means")
    p.add_argument("--prompts_root", type=Path, default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--means_root", type=Path, default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability")
    p.add_argument("--selection_path", type=Path, default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability" /
                   "prunedfail_seed43" / "pooled_sparse" / "selection.json")
    p.add_argument("--cf_pairs_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "cf_task_pairs.json")
    p.add_argument("--out_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "mediation" / "steer_fvablate")
    p.add_argument("--split_path", type=Path, default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--model_name", default=None)
    p.add_argument("--token_budget", type=int, default=12000)
    p.add_argument("--batch_cap", type=int, default=16)
    p.add_argument("--task_stride", type=int, default=1, help="pilot: every k-th task of the sorted pool")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)[::args.task_stride][args.shard_idx::args.shard_n]
    args.out_root.mkdir(parents=True, exist_ok=True)
    pairs = json.load(open(args.cf_pairs_path))["pairs"]
    sel_flat = torch.tensor(json.load(open(args.selection_path))["selected_flat"])
    if args.model_name is None:
        model, tok = load_model(args.model_dir)
    else:
        model, tok = load_model_generic(args.model_dir, args.model_name)
    tok.padding_side = "left"
    inj = Injector(model, [args.layer])
    ab = FVAblator(model)
    layers = frozenset(range(9, ab.n_layers))
    gm = (torch.load(args.grand_mean_cue, map_location="cpu", weights_only=False)["mean"].float().cuda()
          if args.with_mean else None)
    uargs = SimpleNamespace(means_root=args.means_root)
    print(f"{len(tasks)} tasks on this shard", flush=True)
    for task in tasks:
        op = args.out_root / f"{task}.json"
        if op.exists():
            print(f"{task}: exists, skip", flush=True); continue
        items = build_items_6shot(task, args.prompts_root, tok, real_labels=False)
        m = torch.load(args.resid_means_root / f"{task}.pt", map_location="cpu",
                       weights_only=False)["resid_means"][args.layer].float().cuda()
        u_own, _ = unit_fv(task, uargs, model, sel_flat)
        u_cf, _ = unit_fv(pairs[task], uargs, model, sel_flat)
        u_own, u_cf = u_own.cuda(), u_cf.cuda()
        conds = [("unsteered", None, None, None), ("steer", args.alpha * m, None, None),
                 ("steer_own_zero", args.alpha * m, u_own, None), ("steer_cf_zero", args.alpha * m, u_cf, None)]
        if gm is not None:
            conds += [("steer_own_mean", args.alpha * m, u_own, torch.outer(gm @ u_own, u_own)),
                      ("steer_cf_mean", args.alpha * m, u_cf, torch.outer(gm @ u_cf, u_cf))]
        res = {"task": task, "group": group[task], "cf_task": pairs[task], "layer": args.layer, "alpha": args.alpha,
               "cos_own_cf": round(float(u_own @ u_cf), 4), "n_prompts": len(items),
               "model_name": args.model_name or "EleutherAI/gpt-j-6b", "conditions": {}}
        for cname, vec, u, mproj in conds:
            inj.vec = vec
            ab.u, ab.mproj, ab.layers = u, mproj, (layers if u is not None else None)
            preds = [None] * len(items)
            for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
                lens = [len(items[i]["ids"]) for i in b]
                L = max(lens)
                ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
                att = torch.zeros(len(b), L, dtype=torch.long)
                imask = torch.zeros(len(b), L, dtype=torch.bool)
                cmask = torch.zeros(len(b), L, dtype=torch.bool)
                for r, i in enumerate(b):
                    off = L - lens[r]
                    ids[r, off:] = torch.tensor(items[i]["ids"]); att[r, off:] = 1
                    for p_ in items[i]["inj_idx_list"]:
                        imask[r, off + p_] = True
                    cmask[r, L - 1] = True
                inj.mask = imask.cuda()
                ab.mask = cmask.cuda() if u is not None else None
                max_new = min(max(items[i]["gold_len"] for i in b) + 3, 16)
                torch.manual_seed(zlib.crc32(f"{task}|{cname}|{bi}".encode()))
                with torch.no_grad():
                    gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), do_sample=True,
                                         temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=max_new,
                                         pad_token_id=tok.eos_token_id)
                inj.mask = None; ab.mask = None
                for r, i in enumerate(b):
                    preds[i] = tok.decode(gen[r, L:], skip_special_tokens=True).split("\n")[0].strip()
            inj.vec = None; ab.u = ab.mproj = ab.layers = None
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | {cname}: acc={acc:.3f}", flush=True)
        res["golds"] = [it["gold"] for it in items]
        json.dump(res, open(op, "w"))
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
