#!/usr/bin/env python
"""Cue->label attention at L13 under steering, 1-shot scaffolds, 69 tasks.

USER THEORY TEST 2026-08-19: does injecting the full mean make the final cue token "pay
more attention" to the label token than injecting only the task-unique direction?

Metric: post-softmax attention weight from the FINAL CUE token (last prompt token) to the
FINAL label token, averaged over the 16 heads of layer 13, over 150 prompts, per task.

Conditions (all 1-shot):
  dummy1_unsteered                      '_' label, no intervention
  real_1shot                            true label, no intervention
  dummy1_fullmean_a{0.5,1,2,4}          z += alpha * m_A(L6) at the '_' slot (Injector)
  dummy1_swap1_a{0,...,64}              h <- h - (h.v1)v1 + alpha*s1*v1 (ProjSwap)
Label token = the '_' slot for dummy scaffolds; the LAST token of the true label for real.

Prefill-only with output_attentions (requires eager attention on GPT-J 4.49).
Output: artifacts/69_task_run/raw_mean_steering/cue_attention_L13/<task>.json
  {conditions: {name: {mean_attn, per_prompt (150,)}}, ...}  (resumable per task)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
try:
    from src.sandbox.ext_steerability.ablate_readdir_labeltokens import (
        load_model_eager, prep_task_nshot)
    from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len
    from src.sandbox.ext_steerability.steer_read_dir_methods import Injector, build_items
    from src.sandbox.ext_steerability.steer_taskunique_svd import ProjSwap, signed_dirs
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ablate_readdir_labeltokens import load_model_eager, prep_task_nshot
    from ablate_pc50_labeltokens import batches_by_len
    from steer_read_dir_methods import Injector, build_items
    from steer_taskunique_svd import ProjSwap, signed_dirs

FULLMEAN_ALPHAS = (0.5, 1.0, 2.0, 4.0)
SWAP_ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0)
L_INJECT = 6
L_ATTN = 13


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resid_means_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means")
    p.add_argument("--bases_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" /
                   "meanremoved_top3_bases.pt")
    p.add_argument("--acts_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_avg10_L5-15_acts")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" /
                   "cue_attention_L13")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=4000)
    p.add_argument("--batch_cap", type=int, default=32)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)[args.shard_idx::args.shard_n]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    args.out_root.mkdir(parents=True, exist_ok=True)

    bases = torch.load(args.bases_path, map_location="cpu", weights_only=False)["tasks"]
    dirs = signed_dirs(bases, args.acts_root, split)

    model, tok = load_model_eager(args.model_dir)
    tok.padding_side = "left"
    inj = Injector(model, [L_INJECT])          # additive full-mean steering
    sw = ProjSwap(model, L_INJECT)             # projection-swap steering

    def measure(items):
        """Forward all items; return per-prompt L13 head-mean attention cue->label."""
        out = np.zeros(len(items))
        for b in batches_by_len(items, args.token_budget, args.batch_cap):
            lens = [len(items[i]["ids"]) for i in b]
            L = max(lens)
            ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros(len(b), L, dtype=torch.long)
            mask = torch.zeros(len(b), L, dtype=torch.bool)
            cols = []
            for r, i in enumerate(b):
                n = lens[r]; off = L - n
                ids[r, off:] = torch.tensor(items[i]["ids"])
                att[r, off:] = 1
                for p_ in items[i]["inj_idx_list"]:
                    mask[r, off + p_] = True
                cols.append(off + items[i]["label_idx"])
            inj.mask = sw.mask = mask.cuda()
            with torch.no_grad():
                o = model(input_ids=ids.cuda(), attention_mask=att.cuda(),
                          output_attentions=True)
            A = o.attentions[L_ATTN].float()          # (B, 16, L, L)
            for r, i in enumerate(b):
                out[i] = float(A[r, :, -1, cols[r]].mean())
            inj.mask = sw.mask = None
            del o, A
        return out

    for task in tasks:
        out_path = args.out_root / f"{task}.json"
        res = json.load(open(out_path)) if out_path.exists() else None
        if res is None:
            res = {"task": task, "group": group[task], "layer_attn": L_ATTN,
                   "layer_inject": L_INJECT, "n_prompts": 150, "conditions": {}}

        m = torch.load(args.resid_means_root / f"{task}.pt", map_location="cpu",
                       weights_only=False)["resid_means"][L_INJECT].float().cuda()
        V, s, _, _ = dirs[task]
        v1 = V[:1].cuda()
        r1 = float(s[0]) * v1[0]

        dummy_items = real_items = None

        def get_dummy():
            nonlocal dummy_items
            if dummy_items is None:
                dummy_items = build_items(task, args.prompts_root, tok)
                for it in dummy_items:
                    it["inj_idx_list"] = [it["inj_idx"]]
                    it["label_idx"] = it["inj_idx"]        # '_' is the label token
            return dummy_items

        def get_real():
            nonlocal real_items
            if real_items is None:
                real_items = prep_task_nshot(task, args.prompts_root, tok, 1)
                for it in real_items:
                    it["inj_idx_list"] = []
                    it["label_idx"] = max(it["label_pos"])  # final label token
            return real_items

        todo = {"dummy1_unsteered": ("dummy", None, None),
                "real_1shot": ("real", None, None)}
        for a in FULLMEAN_ALPHAS:
            todo[f"dummy1_fullmean_a{a}"] = ("dummy", "inj", a)
        for a in SWAP_ALPHAS:
            todo[f"dummy1_swap1_a{a}"] = ("dummy", "swap", a)

        for cname, (which, mode, a) in todo.items():
            if cname in res["conditions"]:
                continue
            items = get_dummy() if which == "dummy" else get_real()
            inj.vec = sw.V = sw.vec = None
            if mode == "inj":
                inj.vec = a * m
            elif mode == "swap":
                sw.V = v1
                sw.vec = (a * r1) if a != 0.0 else None
            vals = measure(items)
            inj.vec = sw.V = sw.vec = None
            res["conditions"][cname] = {"mean_attn": round(float(vals.mean()), 6),
                                        "per_prompt": [round(float(x), 6) for x in vals]}
            print(f"{task} | {cname}: attn={vals.mean():.4f}", flush=True)
            with open(out_path, "w") as f:
                json.dump(res, f)
    print("capture done", flush=True)


if __name__ == "__main__":
    main()
