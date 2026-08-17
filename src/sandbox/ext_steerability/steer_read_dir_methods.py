#!/usr/bin/env python
"""Read-direction steering sweep: definition method (S1) x injection layer (S2).

Levers per write_up/read_direction_eval_levers.md, with the user's 2026-08-17 decisions:
  S1  every read-direction definition we have = the 4 energy-90 sweep brackets
      (cosine_M, dot_M, cosine_perhead, dot_perhead), each aggregated to a per-task vector
      by AVERAGING OVER THE TASK'S 150 PER-PROMPT read directions:
        unit    = normalise(mean of the 150 UNIT per-prompt dirs)
        natural = mean of the 150 NATURAL per-prompt dirs (carries the construction's scale)
      Measured empirically: cos(unit, natural) = 1.0000 for every bracket/task — the two
      averaging conventions give the SAME direction, so the normalisation lever is purely a
      dose choice. Injected vector is therefore alpha * ||natural|| * unit, with alpha in
      {0.5, 1, 2, 4} (user decision 2026-08-17); alpha = 1 reproduces the natural-magnitude
      condition of the earlier single-vector run.
  S2  injection layer(s): each single layer L3..L15, plus the bands L3-L15 and L7-L11
  S3  1-shot dummy-output scaffold "Q: {in-distribution input}\nA: _\n\nQ: {query}\nA:",
      injection at the ' _' token (the demo label slot)
  S4  single-direction additive steering, z <- z + alpha * r

Readout: temperature-1 sampled exact match on the task's 150 fixed prompts (same protocol
and seeding as the earlier steering runs).

Work unit = (task, layer_config); the unsteered baseline is layer-independent and is
computed once per task (by the unit holding the first layer config). Shard over the
flattened unit list with --shard_idx/--shard_n. Skip-if-exists makes reruns resumable.

Outputs (artifacts/69_task_run/read_dir_method_steering/):
  <task>__L<cfg>.json    accs + preds per condition for that layer config
  <task>__baseline.json  unsteered scaffold accuracy + preds
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
    from src.sandbox.ext_steerability.steer_read_dir_1shot import load_model, batches_by_len
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from steer_read_dir_1shot import load_model, batches_by_len

BRACKETS = ("cosine_M", "dot_M", "cosine_perhead", "dot_perhead")
ALPHAS = (0.5, 1.0, 2.0, 4.0)
LAYER_CONFIGS = [str(l) for l in range(3, 16)] + ["3-15", "7-11"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_tasks", type=int, default=20)
    p.add_argument("--task_seed", type=int, default=43)
    p.add_argument("--layer_configs", type=str, default=",".join(LAYER_CONFIGS))
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--sweep_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "read_dir_sweep")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "read_dir_method_steering")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=24000)
    p.add_argument("--batch_cap", type=int, default=48)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def parse_cfg(cfg):
    if "-" in cfg:
        lo, hi = (int(x) for x in cfg.split("-"))
        return list(range(lo, hi + 1))
    return [int(cfg)]


class Injector:
    """Add vec to the OUTPUT hidden state of each hooked block at masked positions
    (prefill only). Hooks are removable so layer configs can be swapped in one process."""

    def __init__(self, model, layers):
        self.vec = None
        self.mask = None
        self.layers = list(layers)
        self.handles = [model.transformer.h[l].register_forward_hook(self._hook)
                        for l in self.layers]

    def _hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self.vec is None or self.mask is None or hs.shape[1] != self.mask.shape[1]:
            return None
        hs = hs.clone()
        hs[self.mask] = (hs[self.mask].float() + self.vec).to(hs.dtype)
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def build_items(task, prompts_root, tok):
    """The S3 scaffold: real in-distribution demo input, dummy '_' label, real query."""
    anchor_id = tok(" _").input_ids
    assert len(anchor_id) == 1
    anchor_id = anchor_id[0]
    recs = json.load(open(prompts_root / task / "train_prompts.json"))
    assert len(recs) == 150
    items = []
    for rec in recs:
        q = str(rec["query"]["input"])
        gold = rec["query"]["output"]
        gold = str(gold[0] if isinstance(gold, list) else gold).strip()
        demo_inp = str(rec["demos"][0]["input"])   # in-distribution, never the query
        assert demo_inp != q
        pre = f"Q: {demo_inp}\nA:"
        ids = tok(f"{pre} _\n\nQ: {q}\nA:").input_ids
        inj_idx = ids.index(anchor_id)
        assert inj_idx == len(tok(pre).input_ids), \
            f"{task}: ' _' not directly after the demo cue ({inj_idx})"
        items.append({"ids": ids, "inj_idx": inj_idx, "gold": gold,
                      "gold_len": len(tok(" " + gold).input_ids)})
    return items


def build_vectors(task, sweep_root):
    """Per bracket: unit = normalise(mean of unit rows); natural = mean of natural rows."""
    out = {}
    for b in BRACKETS:
        d = torch.load(sweep_root / b / f"{task}.pt", map_location="cpu", weights_only=False)
        r = d["r"].double()                                   # (150, 4096) unit rows
        assert r.shape == (150, 4096)
        assert float((r.norm(dim=1) - 1).abs().max()) < 1e-5, f"{task}/{b}: rows not unit"
        u = r.mean(0)
        u = u / u.norm()
        nat = d["r_task"].double() * float(d["r_task_norm"])  # mean of natural rows
        out[b] = {"unit": u.float(), "natural": nat.float(),
                  "cos_unit_natural": float(torch.nn.functional.cosine_similarity(
                      u, nat.double(), dim=0)),
                  "natural_norm": float(nat.norm())}
    return out


def run_conditions(model, tok, inj, items, conds, task, tag, budget, cap):
    """conds = list of (name, vec_or_None). Returns {name: {acc, preds}}."""
    res = {}
    for cname, vec in conds:
        inj.vec = None if vec is None else vec.cuda()
        preds = [None] * len(items)
        for bi, b in enumerate(batches_by_len(items, budget, cap)):
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
            torch.manual_seed(zlib.crc32(f"{task}|{tag}|{cname}|{bi}".encode()))
            with torch.no_grad():
                gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                     do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                     max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
            inj.mask = None
            for r, i in enumerate(b):
                preds[i] = tok.decode(gen[r, L:], skip_special_tokens=True).split("\n")[0].strip()
        acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
        res[cname] = {"acc": round(acc, 4), "preds": preds}
        print(f"{task} | {tag} | {cname}: acc={acc:.3f}", flush=True)
    return res


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    tasks = sorted(random.Random(args.task_seed).sample(sorted(split["train_tasks"]),
                                                        args.n_tasks))
    cfgs = args.layer_configs.split(",")
    units = [(t, c) for t in tasks for c in cfgs][args.shard_idx::args.shard_n]
    print(f"{len(tasks)} tasks x {len(cfgs)} layer configs -> {len(units)} units on this "
          f"shard ({args.shard_idx}/{args.shard_n})", flush=True)
    print("tasks:", tasks, flush=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args.model_dir)
    tok.padding_side = "left"
    item_cache, vec_cache = {}, {}

    for task, cfg in units:
        out_path = args.out_root / f"{task}__L{cfg}.json"
        base_path = args.out_root / f"{task}__baseline.json"
        need_base = cfg == cfgs[0] and not base_path.exists()
        if out_path.exists() and not need_base:
            print(f"{task} L{cfg}: exists, skip", flush=True)
            continue
        if task not in item_cache:
            item_cache[task] = build_items(task, args.prompts_root, tok)
            vec_cache[task] = build_vectors(task, args.sweep_root)
        items, vecs = item_cache[task], vec_cache[task]

        layers = parse_cfg(cfg)
        inj = Injector(model, layers)
        try:
            if need_base:
                base = run_conditions(model, tok, inj, items, [("baseline", None)],
                                      task, "base", args.token_budget, args.batch_cap)
                with open(base_path, "w") as f:
                    json.dump({"task": task, "scaffold": "sampled_input_underscore",
                               "n_prompts": len(items), "conditions": base,
                               "golds": [it["gold"] for it in items]}, f)
            if not out_path.exists():
                conds = []
                for b in BRACKETS:
                    for a in ALPHAS:
                        # alpha is a multiple of the bracket's own natural magnitude
                        conds.append((f"{b}__a{a}",
                                      (a * vecs[b]["natural_norm"]) * vecs[b]["unit"]))
                res = run_conditions(model, tok, inj, items, conds, task, f"L{cfg}",
                                     args.token_budget, args.batch_cap)
                with open(out_path, "w") as f:
                    json.dump({"task": task, "layer_config": cfg, "layers": layers,
                               "scaffold": "sampled_input_underscore",
                               "n_prompts": len(items),
                               "vector_meta": {b: {k: vecs[b][k] for k in
                                                   ("cos_unit_natural", "natural_norm")}
                                               for b in BRACKETS},
                               "conditions": res,
                               "golds": [it["gold"] for it in items]}, f)
        finally:
            inj.remove()
        assert not inj.handles
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
