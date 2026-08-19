#!/usr/bin/env python
"""Evaluate label-slot steering vectors on the 1-shot dummy-label scaffold (all 69 tasks).

Conditions per task (T=1 sampled exact match, 150 prompts, injection at the ' _' token):
  baseline                      no injection
  headsum_L<sel>@L<inj>         sum of the sparse-selected heads' label-token mean outputs
                                (selection fit at L<sel>), injected at L<inj>; the cross
                                terms separate "which heads" from "which site"
  meandiff@L<inj>               task mean residual at the label token MINUS the 55-train-task
                                mean at that layer (the mean-activation-difference baseline)
  rawmean@L<inj>                task mean residual at the label token, un-differenced
Each steered condition sweeps alpha in {0.5, 1, 2, 4} x the vector's own natural magnitude.

Outputs: artifacts/69_task_run/read_vector_head_selection/eval/<task>.json
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
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from steer_read_dir_1shot import load_model, batches_by_len
    from steer_read_dir_methods import Injector, build_items

ALPHAS = (0.5, 1.0, 2.0, 4.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sel_L7", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "read_vector_head_selection"
                   / "pooled_sparse" / "selection.json")
    p.add_argument("--sel_L3", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "read_vector_head_selection_L3"
                   / "pooled_sparse" / "selection.json")
    p.add_argument("--head_means_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_head_means")
    p.add_argument("--resid_means_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "read_vector_head_selection" / "eval")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--layers", type=int, nargs="+", default=[3, 7])
    p.add_argument("--token_budget", type=int, default=24000)
    p.add_argument("--batch_cap", type=int, default=48)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def wo_slices(model, n_heads=16, head_dim=256):
    """(28, 16, 4096, 256) W_O slices as fp32 on GPU is large; keep per-layer views."""
    return [model.transformer.h[l].attn.out_proj.weight.detach().float()
            for l in range(len(model.transformer.h))]


def head_sum_vector(head_means, flat_heads, wos, n_heads=16, head_dim=256):
    """sum_h W_O^h m_A[h] over the selected flat head indices."""
    v = torch.zeros(4096, dtype=torch.float32, device=wos[0].device)
    for f in flat_heads:
        l, h = f // n_heads, f % n_heads
        w = wos[l].view(4096, n_heads, head_dim)[:, h]        # (4096, 256)
        v += w @ head_means[l, h].to(w.device).float()
    return v


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)[args.shard_idx::args.shard_n]
    args.out_root.mkdir(parents=True, exist_ok=True)

    sels = {}
    for tag, pth in (("L7", args.sel_L7), ("L3", args.sel_L3)):
        if pth.exists():
            sels[tag] = json.load(open(pth))["selected_flat"]
            print(f"selection {tag}: {len(sels[tag])} heads", flush=True)
        else:
            print(f"selection {tag}: MISSING ({pth}) - skipping those conditions", flush=True)

    # mean-difference reference: mean over the 55 TRAIN tasks only (no heldout leakage)
    ref = torch.stack([torch.load(args.resid_means_root / f"{t}.pt", map_location="cpu",
                                  weights_only=False)["resid_means"]
                       for t in split["train_tasks"]]).mean(0)          # (28, 4096)

    model, tok = load_model(args.model_dir)
    tok.padding_side = "left"
    wos = wo_slices(model)
    injectors = {l: Injector(model, [l]) for l in args.layers}

    for task in tasks:
        out_path = args.out_root / f"{task}.json"
        if out_path.exists():
            print(f"{task}: exists, skip", flush=True)
            continue
        items = build_items(task, args.prompts_root, tok)
        hm = torch.load(args.head_means_root / f"{task}.pt", map_location="cpu",
                        weights_only=False)["head_means"]
        rm = torch.load(args.resid_means_root / f"{task}.pt", map_location="cpu",
                        weights_only=False)["resid_means"]
        vecs = {}
        for tag, flat in sels.items():
            vecs[f"headsum_{tag}"] = head_sum_vector(hm, flat, wos)
        for l in args.layers:
            vecs[f"meandiff_L{l}"] = (rm[l] - ref[l]).cuda()
            vecs[f"rawmean_L{l}"] = rm[l].clone().cuda()

        res = {"task": task, "group": group[task], "n_prompts": len(items),
               "vector_norms": {k: float(v.norm()) for k, v in vecs.items()},
               "n_heads": {k: len(v) for k, v in sels.items()},
               "conditions": {}}

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
        for tag in sels:
            for inj_l in args.layers:
                base = vecs[f"headsum_{tag}"]
                for a in ALPHAS:
                    run(f"headsum_{tag}sel@L{inj_l}_a{a}", a * base, inj_l)
        for l in args.layers:
            for kind in ("meandiff", "rawmean"):
                base = vecs[f"{kind}_L{l}"]
                for a in ALPHAS:
                    run(f"{kind}@L{l}_a{a}", a * base, l)
        res["golds"] = [it["gold"] for it in items]
        with open(out_path, "w") as f:
            json.dump(res, f)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
