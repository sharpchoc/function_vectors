#!/usr/bin/env python
"""Steering test for per-task READ directions on a content-free 1-shot scaffold (GPU).

Prompt per query: "Q: Input\nA: Output\n\nQ: {query}\nA:" — the literal words Input/Output
give the ICL format but no task content. At the " Output" token position, inject
    h <- h + alpha * v_task
into the residual stream at the output of block --layer (default 7), where v_task is the
task's NATURAL-magnitude per-task read direction = mean over the 150 per-prompt natural
read dirs = r_task * r_task_norm from the read_dir_sweep tree (linearity). Families:
dot_perhead and cosine_perhead. alpha swept over {0.5, 1, 2, 4}; baseline = same scaffold,
no injection. Readout: T=1 sampled exact match on the task's 150 train-prompt queries
(same protocol/seeding as the ablation evals).

Task picked at random from the 55 train tasks (seeded) unless --task is given.

--scaffold_mode:
  const              "Q: Input\nA: Output\n\nQ: {q}\nA:", inject at ' Output' (original)
  sampled_underscore "Q: {demo_input}\nA: _\n\nQ: {q}\nA:" with demo_input = the record's
                     first demo's input (in-distribution, never the query); inject at ' _'.
Outputs: <out_root>/<task>[__<mode>].json (accs + preds) — summarize/plot separately.
"""
import argparse
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

# local bootstrap for in-repo runs; a PYTHONPATH-supplied repo also works (staged copies)
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
try:
    from src.sandbox.ext_steerability.ablate_pc50_labeltokens import load_model, batches_by_len
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ablate_pc50_labeltokens import load_model, batches_by_len

FAMILIES = ("dot_perhead", "cosine_perhead")
ALPHAS = (0.5, 1.0, 2.0, 4.0)
SCAFFOLD = "Q: Input\nA: Output\n\n"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--scaffold_mode", choices=("const", "sampled_underscore"), default="const")
    p.add_argument("--layers", type=str, default="7",
                   help="injection layers: single '7' or inclusive range '7-20'")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--sweep_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "read_dir_sweep")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "read_dir_steering_1shot")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--task_seed", type=int, default=43)
    return p.parse_args()


class Injector:
    """Forward hook on one or more blocks: add vec to each block's OUTPUT hidden state at
    masked positions (prefill only)."""

    def __init__(self, model, layers):
        self.vec = None    # (D,) fp32 cuda (already alpha-scaled)
        self.mask = None   # (B, L) bool cuda
        self.h = [model.transformer.h[l].register_forward_hook(self._hook) for l in layers]

    def _hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self.vec is None or self.mask is None or hs.shape[1] != self.mask.shape[1]:
            return None
        hs = hs.clone()
        hs[self.mask] = (hs[self.mask].float() + self.vec).to(hs.dtype)
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    task = args.task or random.Random(args.task_seed).choice(sorted(split["train_tasks"]))
    print(f"task: {task}", flush=True)

    model, tok = load_model(args.model_dir)
    tok.padding_side = "left"

    anchor_str = " Output" if args.scaffold_mode == "const" else " _"
    anchor_id = tok(anchor_str).input_ids
    assert len(anchor_id) == 1
    anchor_id = anchor_id[0]

    recs = json.load(open(args.prompts_root / task / "train_prompts.json"))
    assert len(recs) == 150
    items = []
    for rec in recs:
        q = str(rec["query"]["input"])
        gold = rec["query"]["output"]
        gold = str(gold[0] if isinstance(gold, list) else gold).strip()
        if args.scaffold_mode == "const":
            prompt = f"{SCAFFOLD}Q: {q}\nA:"
            pre = None
        else:
            demo_inp = str(rec["demos"][0]["input"])   # in-distribution, never the query
            assert demo_inp != q
            pre = f"Q: {demo_inp}\nA:"
            prompt = f"{pre} _\n\nQ: {q}\nA:"
        ids = tok(prompt).input_ids
        # the scaffold's "\n\n" retokenizes in context (628 -> 198,198): anchor on the
        # injection token directly, then sanity-check its position.
        inj_idx = ids.index(anchor_id)
        if args.scaffold_mode == "const":
            assert inj_idx == 6, f"' Output' not at expected scaffold position: {inj_idx}"
        else:
            assert inj_idx == len(tok(pre).input_ids), \
                f"' _' not directly after the demo cue: {inj_idx}"
        items.append({"ids": ids, "inj_idx": inj_idx, "gold": gold,
                      "gold_len": len(tok(" " + gold).input_ids)})

    vecs = {}
    for fam in FAMILIES:
        d = torch.load(args.sweep_root / fam / f"{task}.pt", map_location="cpu", weights_only=False)
        v = d["r_task"].float() * d["r_task_norm"]   # natural-magnitude per-task read dir
        vecs[fam] = v.cuda()
        print(f"{fam}: |v_task| = {v.norm():.2f}", flush=True)

    if "-" in args.layers:
        lo, hi = (int(x) for x in args.layers.split("-"))
        layers = list(range(lo, hi + 1))
    else:
        layers = [int(args.layers)]
    print(f"injection layers: {layers}", flush=True)
    inj = Injector(model, layers)
    conds = [("baseline", None, 0.0)]
    conds += [(f"{fam}_a{a}", fam, a) for fam in FAMILIES for a in ALPHAS]
    res = {"task": task, "layers": layers, "scaffold_mode": args.scaffold_mode,
           "n_prompts": len(items),
           "v_norms": {f: float(vecs[f].norm()) for f in FAMILIES}, "conditions": {}}
    for cname, fam, a in conds:
        inj.vec = None if fam is None else a * vecs[fam]
        preds = [None] * len(items)
        for bi, b in enumerate(batches_by_len(items, 24000, 48)):
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
        acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
        res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
        print(f"{task} | {cname}: acc={acc:.3f}", flush=True)
    res["golds"] = [it["gold"] for it in items]
    args.out_root.mkdir(parents=True, exist_ok=True)
    stem = task if args.scaffold_mode == "const" else f"{task}__{args.scaffold_mode}"
    if len(layers) > 1 or layers != [7]:
        stem += f"__L{args.layers}"
    with open(args.out_root / f"{stem}.json", "w") as f:
        json.dump(res, f)
    print(f"wrote {args.out_root / (stem + '.json')}", flush=True)


if __name__ == "__main__":
    main()
