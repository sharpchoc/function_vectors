#!/usr/bin/env python
"""Head-level MEAN ABLATION of the FV heads at the final cue token of 6-shot prompts.

Companion to ablate_fv_cue6.py (which removes the rank-1 FV *direction* from the residual
stream). Here the selected FV heads themselves are intervened on: at the FINAL QUERY CUE
token (prefill only) each selected head's output z^{l,h} (its slice of the attention
out-projection input, the quantity whose per-task mean defines the FV) is replaced by the
GRAND MEAN of that head's cue-token output over ALL pool tasks (equal-task-weighted mean of
the stored per-task 10-shot head means, <means_root>/<task>/means.pt; the evaluated task is
included). W_O is linear, so this equals replacing the head's residual contribution by its
cross-task mean contribution. Everything else (other heads, other positions, MLPs) is intact.

Conditions per task (T=1 sampled exact match, 150 six-shot prompts, crc32 seeding as in
sixshot_dummy_steer / ablate_fv_cue6):
  real6_baseline        unablated 6-shot
  fvheads_mean          selected FV heads (GPT-J 37 / Qwen2.5-7B-Instruct 140) mean-ablated
  rand{k}_mean, k<K     control: K fixed-seed random sets of the SAME SIZE drawn from the
                        NON-selected heads, each mean-ablated with the same grand means
Outputs: <out_root>/eval/<task>.json, <out_root>/grand_head_means.pt, <out_root>/random_sets.json
Sharding: --shard_idx/--shard_n over the sorted pool tasks.
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
from src.utils.model_utils import get_attn_out_proj
try:
    from src.sandbox.ext_steerability.ablate_pc50_labeltokens import (
        load_model, batches_by_len, model_dims)
    from src.sandbox.ext_steerability.sixshot_dummy_steer import build_items_6shot
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ablate_pc50_labeltokens import load_model, batches_by_len, model_dims
    from sixshot_dummy_steer import build_items_6shot


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--means_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability")
    p.add_argument("--selection_path", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability" /
                   "prunedfail_seed43" / "pooled_sparse" / "selection.json")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "FV_head_mean_ablation")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--model_name", default=None,
                   help="HF id for a non-GPT-J model (Qwen2.5 port, bf16); default GPT-J-6B")
    p.add_argument("--n_random", type=int, default=3, help="random matched-size control sets")
    p.add_argument("--token_budget", type=int, default=24000)
    p.add_argument("--batch_cap", type=int, default=48)
    p.add_argument("--max_tasks", type=int, default=None, help="cap tasks (smoke tests)")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    p.add_argument("--reverse", action="store_true",
                   help="process the shard's tasks in reverse order (second worker on the "
                        "same shard; per-task outputs are skipped when present)")
    return p.parse_args()


class HeadMeanAblator:
    """forward_pre_hook on every layer's attention out-projection: at masked positions,
    overwrite the selected heads' slices of its input with fixed per-head values.
    Active only when the sequence length matches the armed mask (prefill)."""

    def __init__(self, model):
        self.n_layers, self.d, self.n_heads = model_dims(model)
        self.hd = self.d // self.n_heads
        self.sets = None       # {layer: (cols LongTensor, vals fp32)} or None (no-op)
        self.mask = None       # (B, L) bool cuda
        self.handles = [get_attn_out_proj(model, l).register_forward_pre_hook(self._make(l))
                        for l in range(self.n_layers)]

    def arm(self, flat_heads, grand):
        """flat_heads: iterable of l*H+h; grand: (n_layers, H, hd) fp32."""
        by_layer = {}
        for f in sorted(int(x) for x in flat_heads):
            by_layer.setdefault(f // self.n_heads, []).append(f % self.n_heads)
        self.sets = {}
        for l, hs in by_layer.items():
            cols = torch.cat([torch.arange(h * self.hd, (h + 1) * self.hd) for h in hs])
            vals = torch.cat([grand[l, h].float() for h in hs])
            self.sets[l] = (cols.cuda(), vals.cuda())

    def _make(self, l):
        def hook(module, args):
            x = args[0]
            if self.sets is None or l not in self.sets or self.mask is None \
                    or x.shape[1] != self.mask.shape[1]:
                return None
            cols, vals = self.sets[l]
            x = x.clone()
            sel = x[self.mask]                       # (n_masked, resid)
            sel[:, cols] = vals.to(x.dtype)
            x[self.mask] = sel
            return (x,) + tuple(args[1:])
        return hook


def grand_head_means(args, tasks_all):
    """Equal-task-weighted mean of the stored per-task head means over all pool tasks."""
    out = args.out_root / "grand_head_means.pt"
    if out.exists():
        d = torch.load(out, map_location="cpu", weights_only=False)
        assert d["tasks"] == tasks_all, "grand_head_means.pt was built from a different pool"
        return d["mean"]
    per = []
    for t in tasks_all:
        m = torch.load(args.means_root / t / "means.pt", map_location="cpu", weights_only=False)
        assert m["n_prompts"] >= 40, f"{t}: {m['n_prompts']} capture prompts"
        per.append(m["head_means"].double())
    gm = torch.stack(per).mean(dim=0).float()        # (n_layers, H, hd)
    args.out_root.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp%d" % args.shard_idx)
    torch.save({"mean": gm, "tasks": tasks_all, "n_tasks": len(tasks_all),
                "definition": "equal-task-weighted grand mean over all pool tasks of the "
                              "per-task cue-token head means (<means_root>/<task>/means.pt, "
                              "10-shot capture prompts, out-projection input split by head)"},
               tmp)
    tmp.replace(out)
    return gm


def random_sets(args, sel_flat, n_total):
    """K fixed-seed random head sets of the FV's size, drawn from the NON-selected heads."""
    out = args.out_root / "random_sets.json"
    if out.exists():
        d = json.load(open(out))
        assert d["selected_flat"] == sorted(int(x) for x in sel_flat)
        return [list(s) for s in d["sets"]]
    pool = np.array(sorted(set(range(n_total)) - set(int(x) for x in sel_flat)))
    sets = []
    for k in range(args.n_random):
        rng = np.random.RandomState(zlib.crc32(f"fvheads_random|{k}".encode()) % (2 ** 32))
        sets.append(sorted(int(x) for x in rng.choice(pool, size=len(sel_flat), replace=False)))
    args.out_root.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp%d" % args.shard_idx)
    json.dump({"selected_flat": sorted(int(x) for x in sel_flat), "n_total_heads": n_total,
               "seed_rule": "RandomState(crc32('fvheads_random|k'))", "sets": sets}, open(tmp, "w"))
    tmp.replace(out)
    return sets


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks_all = sorted(group)
    tasks = tasks_all[args.shard_idx::args.shard_n]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    if args.reverse:
        tasks = tasks[::-1]
    print(f"{len(tasks)} tasks on this shard", flush=True)

    sel = json.load(open(args.selection_path))
    sel_flat = [int(x) for x in sel["selected_flat"]]
    assert len(sel_flat) == sel["n_selected"]
    gm = grand_head_means(args, tasks_all)
    n_layers, n_heads, hd = gm.shape
    rsets = random_sets(args, sel_flat, n_layers * n_heads)

    model, tok = load_model(args.model_dir, args.model_name)
    ml, md, mh = model_dims(model)
    assert (ml, mh, md // mh) == (n_layers, n_heads, hd), "means.pt shape != model dims"
    tok.padding_side = "left"
    ab = HeadMeanAblator(model)
    outdir = args.out_root / "eval"
    outdir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        out_path = outdir / f"{task}.json"
        if out_path.exists():
            print(f"{task}: exists, skip", flush=True)
            continue
        items = build_items_6shot(task, args.prompts_root, tok, real_labels=True)
        res = {"task": task, "group": group[task], "n_prompts": len(items), "n_shots": 6,
               "n_fv_heads": len(sel_flat), "n_random_sets": len(rsets),
               "definition": "final cue token, prefill: selected heads' out-proj input slices "
                             "<- grand mean (equal-task mean of per-task 10-shot head means, "
                             f"{len(tasks_all)} tasks); random controls = same-size sets of "
                             "non-selected heads",
               "selection_path": str(args.selection_path),
               "model_name": args.model_name or "EleutherAI/gpt-j-6b",
               "conditions": {}}
        conds = [("real6_baseline", None), ("fvheads_mean", sel_flat)] + \
                [(f"rand{k}_mean", s) for k, s in enumerate(rsets)]
        for cname, heads in conds:
            if heads is None:
                ab.sets = None
            else:
                ab.arm(heads, gm)
            preds = [None] * len(items)
            for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
                lens = [len(items[i]["ids"]) for i in b]
                L = max(lens)
                ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
                att = torch.zeros(len(b), L, dtype=torch.long)
                mask = torch.zeros(len(b), L, dtype=torch.bool)
                for r, i in enumerate(b):   # LEFT padding for generation
                    n = lens[r]; off = L - n
                    ids[r, off:] = torch.tensor(items[i]["ids"])
                    att[r, off:] = 1
                    mask[r, L - 1] = True   # final cue token
                ab.mask = mask.cuda()
                max_new = min(max(items[i]["gold_len"] for i in b) + 3, 16)
                torch.manual_seed(zlib.crc32(f"{task}|{cname}|{bi}".encode()))
                with torch.no_grad():
                    gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                         do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                         max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
                ab.mask = None
                for r, i in enumerate(b):
                    preds[i] = tok.decode(gen[r, L:], skip_special_tokens=True).split("\n")[0].strip()
            ab.sets = None
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | {cname}: acc={acc:.3f}", flush=True)
        res["golds"] = [it["gold"] for it in items]
        with open(out_path, "w") as f:
            json.dump(res, f)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
