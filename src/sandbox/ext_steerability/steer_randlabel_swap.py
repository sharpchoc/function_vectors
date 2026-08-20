#!/usr/bin/env python
"""Task-unique projection-swap steering on the 6-shot RANDOM-label scaffold (69 tasks).

HYPOTHESIS-1 TEST (2026-08-20): the shared carrier's steering advantage comes from
repairing the defective '_' base — on labels that are real words (sampled from OTHER
tasks' output pools, so they carry natural carrier content but no/wrong task identity),
the task-unique swap should close the gap to full-mean steering.

Scaffold + label sampling + injection positions are IDENTICAL to
sixshot_randomlabel_steer.py (its build_items_randomlabel: every token of all six wrong
labels; deterministic per-(task,record,slot) sampling), so prompts are byte-identical to
the peer full-mean run (random6_steer_a* / random6_unsteered) and directly comparable.

Steering: h <- h - (h.v1)v1 + alpha * s1 * v1  (ProjSwap at block-6 output, prefill only)
with v1, s1 from meanremoved_top3_bases (sign-fixed toward the task's L6 feature).
alpha grid: 0 (removal-only), 2, 4, 8, 16, 32, 64.
Baseline random6_unsteered is recomputed with the peer's exact condition name and
batching (budget 9000 / cap 12) as a bit-for-bit infra cross-check.

Readout: T=1 sampled exact match, established crc32 seeding.
Outputs: artifacts/69_task_run/raw_mean_steering/randlabel_swap/<task>.json (resumable).
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
try:
    from src.sandbox.ext_steerability.steer_read_dir_1shot import load_model, batches_by_len
    from src.sandbox.ext_steerability.sixshot_randomlabel_steer import (
        build_output_pools, build_items_randomlabel)
    from src.sandbox.ext_steerability.steer_taskunique_svd import ProjSwap, signed_dirs
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from steer_read_dir_1shot import load_model, batches_by_len
    from sixshot_randomlabel_steer import build_output_pools, build_items_randomlabel
    from steer_taskunique_svd import ProjSwap, signed_dirs

ALPHAS = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
LAYER = 6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bases_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" /
                   "meanremoved_top3_bases.pt")
    p.add_argument("--acts_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_avg10_L5-15_acts")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" /
                   "randlabel_swap")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=9000)
    p.add_argument("--batch_cap", type=int, default=12)
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
    tasks = all_tasks[args.shard_idx::args.shard_n]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    args.out_root.mkdir(parents=True, exist_ok=True)

    pools = build_output_pools(all_tasks, args.prompts_root)
    bases = torch.load(args.bases_path, map_location="cpu", weights_only=False)["tasks"]
    dirs = signed_dirs(bases, args.acts_root, split)

    model, tok = load_model(args.model_dir)
    tok.padding_side = "left"
    sw = ProjSwap(model, LAYER)

    for task in tasks:
        out_path = args.out_root / f"{task}.json"
        res = json.load(open(out_path)) if out_path.exists() else None
        V, s, c, _ = dirs[task]
        v1 = V[:1].cuda()
        r1 = float(s[0]) * v1[0]
        if res is None:
            res = {"task": task, "group": group[task], "n_prompts": 150,
                   "layer": LAYER, "alphas": list(ALPHAS),
                   "s1": round(float(s[0]), 4),
                   "natural_L6_coord1": round(float(c[0]), 3),
                   "definition": ("h <- h - (h.v1)v1 + alpha*s1*v1 at EVERY token of all "
                                  "six wrong-task labels, block-6 output; scaffold = "
                                  "sixshot_randomlabel_steer.build_items_randomlabel"),
                   "conditions": {}}
        names = ["random6_unsteered"] + [f"random6_swap1_a{a}" for a in ALPHAS]
        todo = [cn for cn in names if cn not in res["conditions"]]
        if not todo:
            print(f"{task}: all conditions present, skip", flush=True)
            continue
        items = build_items_randomlabel(task, args.prompts_root, tok, pools)

        for cname in todo:
            if cname == "random6_unsteered":
                V_used, vec = None, None
            else:
                a = float(cname.rsplit("_a", 1)[1])
                V_used, vec = v1, (a * r1) if a != 0.0 else None
            sw.V, sw.vec = V_used, vec
            preds = [None] * len(items)
            for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
                lens = [len(items[i]["ids"]) for i in b]
                L = max(lens)
                ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
                att = torch.zeros(len(b), L, dtype=torch.long)
                mask = torch.zeros(len(b), L, dtype=torch.bool)
                for r, i in enumerate(b):
                    n = lens[r]; off = L - n
                    ids[r, off:] = torch.tensor(items[i]["ids"])
                    att[r, off:] = 1
                    for p_ in items[i]["inj_idx_list"]:
                        mask[r, off + p_] = True
                sw.mask = mask.cuda()
                max_new = min(max(items[i]["gold_len"] for i in b) + 3, 16)
                torch.manual_seed(zlib.crc32(f"{task}|{cname}|{bi}".encode()))
                with torch.no_grad():
                    gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                         do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                         max_new_tokens=max_new,
                                         pad_token_id=tok.eos_token_id)
                sw.mask = None
                for r, i in enumerate(b):
                    preds[i] = tok.decode(gen[r, ids.shape[1]:],
                                          skip_special_tokens=True).split("\n")[0].strip()
            sw.V = sw.vec = None
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | {cname}: acc={acc:.3f}", flush=True)
            with open(out_path, "w") as f:
                json.dump(res, f)
    print("eval done", flush=True)


if __name__ == "__main__":
    main()
