#!/usr/bin/env python
"""FIXED-vector layer sweep: (L6+L7 mean + top-1 dir) steering on 1-shot dummy prompts.

USER REQUEST 2026-08-31: the steering vector is the FIXED per-task
    w_A = 0.5*(m_A(L6)+m_A(L7)) + <base, v1>*v1     (build_l67_plus_top1_vectors.py)
injected additively at the ' _' dummy target slot at ONE layer at a time, swept over all
layers 0..27 (unlike sweep_raw_mean_layers.py the vector does NOT change with the
injection layer). alpha in {0.5, 1, 2, 4} x the vector's own norm. 1-shot dummy scaffold,
T=1 sampled exact match on the task's 150 fixed prompts. No shared-mean control.

Outputs: artifacts/69_task_run/l67top1_steering/<task>.json  (same schema as
raw_mean_steering: conditions L{l}_a{a} + baseline)
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
    from src.sandbox.ext_steerability.steer_read_dir_methods import Injector, build_items
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from steer_read_dir_1shot import load_model, batches_by_len
    from steer_read_dir_methods import Injector, build_items

ALPHAS = (0.5, 1.0, 2.0, 4.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vectors_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation"
                   / "l67_plus_top1_vectors.pt")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "l67top1_steering")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--layers", type=int, nargs="+", default=list(range(28)))
    p.add_argument("--token_budget", type=int, default=24000)
    p.add_argument("--batch_cap", type=int, default=48)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)[args.shard_idx::args.shard_n]
    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} tasks on this shard, layers {args.layers[0]}..{args.layers[-1]}",
          flush=True)

    vecs = torch.load(args.vectors_path, map_location="cpu", weights_only=False)["tasks"]
    model, tok = load_model(args.model_dir)
    tok.padding_side = "left"
    injectors = {l: Injector(model, [l]) for l in args.layers}

    for task in tasks:
        out_path = args.out_root / f"{task}.json"
        if out_path.exists():
            print(f"{task}: exists, skip", flush=True)
            continue
        items = build_items(task, args.prompts_root, tok)
        w = vecs[task]["vec"].cuda()
        res = {"task": task, "group": group[task], "n_prompts": len(items),
               "vec_norm": vecs[task]["vec_norm"], "n_A": vecs[task].get("n_A", vecs[task].get("u_norm")),
               "site": f"fixed vector from {args.vectors_path.name}, injected additively at the swept layer",
               "alphas": list(ALPHAS), "conditions": {}}

        def run(cname, vec, layer):
            inj = injectors[layer]
            inj.vec = None if vec is None else vec
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
                    mask[r, off + items[i]["inj_idx"]] = True
                inj.mask = mask.cuda()
                max_new = min(max(items[i]["gold_len"] for i in b) + 3, 16)
                torch.manual_seed(zlib.crc32(f"{task}|{cname}|{bi}".encode()))
                with torch.no_grad():
                    gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                         do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                         max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
                inj.mask = None
                for r, i in enumerate(b):
                    preds[i] = tok.decode(gen[r, L:], skip_special_tokens=True).split("\n")[0].strip()
            inj.vec = None
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | {cname}: acc={acc:.3f}", flush=True)

        run("baseline", None, args.layers[0])
        for l in args.layers:
            for a in ALPHAS:
                run(f"L{l}_a{a}", a * w, l)
        res["golds"] = [it["gold"] for it in items]
        with open(out_path, "w") as f:
            json.dump(res, f)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
