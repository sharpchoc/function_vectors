#!/usr/bin/env python
"""FV-direction ablation at the final cue token of 6-shot prompts (69 tasks).

Necessity counterpart of the 37-head FV steering result: remove the single task-FV
direction u_A = v_A/||v_A|| (v_A = sum of the pooled-selection 37 head means, as in
eval_ext.py) from the residual stream entering the transformer blocks, at the FINAL QUERY
CUE token only, of the fixed 150 six-shot prompts per task (first 6 demos of each
train_prompts.json record, true labels). Two ops x two layer clamps x {own, counterfactual}
FV = 8 conditions per task:
  zero : h <- h - (h.u)u
  mean : h <- h - (h.u)u + (m_l.u)u   (m_l = equal-task-weighted grand mean over all 69
                                       tasks of cue-token block-l inputs, --stage combine)
  layer clamps: L9to27 (blocks 9..27) and L0to27 (blocks 0..27), prefill only.
Counterfactual task per task: cf_task_pairs.json (different semantic family, peer study).
Readout: T=1 sampled exact match, established crc32 seeding (sixshot_dummy_steer protocol).

Stages:
  --stage means   : per-task cue-token block-input means -> <out>/cue_means/<task>.pt
  --stage combine : equal-task-weighted grand mean       -> <out>/grand_mean_cue6.pt
  --stage eval    : requires grand mean + cf pairs; writes <out>/eval/<task>.json

Sharding: --shard_idx/--shard_n over the sorted 69 tasks.

--n_shots N (default 6, the original study): same prompt format with the first N demos of
each record. For N != 6 every output is suffixed so the 6-shot artifacts are untouched:
cue_means_{N}shot/, grand_mean_cue{N}.pt, eval_{N}shot/, baseline condition real{N}_baseline.
--layer_cfgs restricts the layer clamps (default: both).
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
    from src.sandbox.ext_steerability.ablate_pc50_labeltokens import (
        load_model, batches_by_len, N_LAYERS, D)
    from src.sandbox.ext_steerability.sixshot_dummy_steer import build_items_6shot
    from src.sandbox.isolation_upper_bound.run_task import build_contributions_single
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ablate_pc50_labeltokens import load_model, batches_by_len, N_LAYERS, D
    from sixshot_dummy_steer import build_items_6shot
    from run_task import build_contributions_single

MODEL_CFG = {"n_layers": N_LAYERS, "n_heads": 16, "resid_dim": D}
LAYER_CFGS = {"L9to27": tuple(range(9, N_LAYERS)), "L0to27": tuple(range(N_LAYERS))}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("means", "combine", "eval"))
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--means_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability")
    p.add_argument("--selection_path", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability" /
                   "prunedfail_seed43" / "pooled_sparse" / "selection.json")
    p.add_argument("--cf_pairs_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" /
                   "cf_task_pairs.json")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "FV_ablation")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=24000)
    p.add_argument("--batch_cap", type=int, default=48)
    p.add_argument("--with_baseline", action="store_true",
                   help="also run an unablated 'real6_baseline' condition (seed-exact "
                        "reproduction of sixshot_dummy_steer's real6_baseline; smoke checks)")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    p.add_argument("--n_shots", type=int, default=6,
                   help="demos per prompt (first N of each record); 6 = original study")
    p.add_argument("--layer_cfgs", nargs="+", default=list(LAYER_CFGS),
                   choices=list(LAYER_CFGS), help="layer clamps to run")
    return p.parse_args()


def build_items_nshot(task, prompts_root, tok, n_shots):
    """True-label N-shot prompts in the sixshot_dummy_steer format (first N demos of each
    of the 150 records). Identical to build_items_6shot(real_labels=True) at N=6."""
    recs = json.load(open(prompts_root / task / "train_prompts.json"))
    assert len(recs) == 150
    items = []
    for rec in recs:
        q = str(rec["query"]["input"])
        gold = rec["query"]["output"]
        gold = str(gold[0] if isinstance(gold, list) else gold).strip()
        demos = rec["demos"][:n_shots]
        assert len(demos) == n_shots
        parts = []
        for d in demos:
            di = str(d["input"])
            assert di != q
            parts.append(f"Q: {di}\nA: {str(d['output']).strip()}\n\n")
        prompt = "".join(parts) + f"Q: {q}\nA:"
        items.append({"ids": tok(prompt).input_ids, "inj_idx_list": [], "gold": gold,
                      "gold_len": len(tok(" " + gold).input_ids)})
    return items


def build_items(task, args, tok):
    if args.n_shots == 6:
        return build_items_6shot(task, args.prompts_root, tok, real_labels=True)
    return build_items_nshot(task, args.prompts_root, tok, args.n_shots)


def sfx(args, six, other):
    """6-shot keeps the original artifact names; other shot counts get suffixed names."""
    return six if args.n_shots == 6 else other.format(n=args.n_shots)


class FVAblator:
    """forward_pre_hook on every block; removes the rank-1 component along a unit vector
    at masked positions, for blocks in the armed layer set. Active only when the sequence
    length matches the armed mask (prefill). Also doubles as the cue-mean capturer."""

    def __init__(self, model):
        self.u = None          # (D,) fp32 cuda, unit norm
        self.mproj = None      # (N_LAYERS, D) fp32 cuda ((m_l.u)u rows) or None (zero op)
        self.layers = None     # frozenset of block indices
        self.mask = None       # (B, L) bool cuda
        self.capture = None    # dict(sums (N_LAYERS, D) fp64, count int) when capturing
        self.handles = [model.transformer.h[l].register_forward_pre_hook(
            self._make(l), with_kwargs=True) for l in range(N_LAYERS)]

    def _make(self, l):
        def hook(module, args, kwargs):
            in_args = bool(args)
            h = args[0] if in_args else kwargs["hidden_states"]
            if self.mask is None or h.shape[1] != self.mask.shape[1]:
                return None
            if self.capture is not None:
                sel = h[self.mask].double()
                self.capture["sums"][l] += sel.sum(dim=0).cpu()
                if l == 0:
                    self.capture["count"] += sel.shape[0]
                return None
            if self.u is None or l not in self.layers:
                return None
            h32 = h[self.mask].float()
            h32 = h32 - torch.outer(h32 @ self.u, self.u)
            if self.mproj is not None:
                h32 = h32 + self.mproj[l]
            h = h.clone()
            h[self.mask] = h32.to(h.dtype)
            if in_args:
                return (h,) + args[1:], kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = h
            return args, kwargs
        return hook


def unit_fv(task, args, model, sel_flat):
    means = torch.load(args.means_root / task / "means.pt", map_location="cpu",
                       weights_only=False)
    C = build_contributions_single(means["head_means"], model, MODEL_CFG)
    v = C[sel_flat.to(C.device)].sum(dim=0).float()
    return v / v.norm(), float(v.norm())


def run_means(args, model, tok, tasks):
    outdir = args.out_root / sfx(args, "cue_means", "cue_means_{n}shot")
    outdir.mkdir(parents=True, exist_ok=True)
    ab = FVAblator(model)
    for task in tasks:
        out = outdir / f"{task}.pt"
        if out.exists():
            print(f"means {task}: exists, skip", flush=True)
            continue
        items = build_items(task, args, tok)
        cap = {"sums": torch.zeros(N_LAYERS, D, dtype=torch.float64), "count": 0}
        ab.capture = cap
        for b in batches_by_len(items, args.token_budget, args.batch_cap):
            lens = [len(items[i]["ids"]) for i in b]
            L = max(lens)
            ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros(len(b), L, dtype=torch.long)
            mask = torch.zeros(len(b), L, dtype=torch.bool)
            for r, i in enumerate(b):   # right padding (no generation here)
                n = lens[r]
                ids[r, :n] = torch.tensor(items[i]["ids"])
                att[r, :n] = 1
                mask[r, n - 1] = True   # final cue token
            ab.mask = mask.cuda()
            with torch.no_grad():
                model(input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False)
            ab.mask = None
        ab.capture = None
        assert cap["count"] == len(items)
        torch.save({"mean": (cap["sums"] / cap["count"]).float(),
                    "n_prompts": cap["count"]}, out)
        print(f"means {task}: {cap['count']} cue tokens", flush=True)


def run_combine(args, tasks_all):
    per = []
    for t in tasks_all:
        d = torch.load(args.out_root / sfx(args, "cue_means", "cue_means_{n}shot") / f"{t}.pt",
                       map_location="cpu", weights_only=False)
        assert d["n_prompts"] == 150, f"{t}: {d['n_prompts']} prompts"
        per.append(d["mean"].double())
    gm = torch.stack(per).mean(dim=0).float()
    out = args.out_root / f"grand_mean_cue{args.n_shots}.pt"
    torch.save({"mean": gm, "n_tasks": len(tasks_all), "n_shots": args.n_shots,
                "definition": "equal-task-weighted grand mean of block-input residuals at "
                              f"the final cue token of the 150 {args.n_shots}-shot prompts, "
                              "all 69 tasks (train+heldout)"}, out)
    print(f"combined {len(tasks_all)} tasks -> {out}")


def run_eval(args, model, tok, tasks, group):
    cf = json.load(open(args.cf_pairs_path))
    pairs, families = cf["pairs"], cf["families"]
    sel = json.load(open(args.selection_path))
    sel_flat = torch.tensor(sel["selected_flat"])
    gm_path = args.out_root / f"grand_mean_cue{args.n_shots}.pt"
    gm = torch.load(gm_path, map_location="cpu",
                    weights_only=False)["mean"].float().cuda()   # (N_LAYERS, D)
    outdir = args.out_root / sfx(args, "eval", "eval_{n}shot")
    outdir.mkdir(parents=True, exist_ok=True)
    ab = FVAblator(model)
    tok.padding_side = "left"
    for task in tasks:
        out_path = outdir / f"{task}.json"
        if out_path.exists():
            print(f"{task}: exists, skip", flush=True)
            continue
        items = build_items(task, args, tok)
        u_own, norm_own = unit_fv(task, args, model, sel_flat)
        u_cf, norm_cf = unit_fv(pairs[task], args, model, sel_flat)
        u_own, u_cf = u_own.cuda(), u_cf.cuda()
        res = {"task": task, "group": group[task], "cf_task": pairs[task],
               "family": families[task], "cf_family": families[pairs[task]],
               "cos_own_cf": round(float(u_own @ u_cf), 4),
               "norm_fv_own": round(norm_own, 2), "norm_fv_cf": round(norm_cf, 2),
               "n_prompts": len(items), "n_shots": args.n_shots,
               "definition": "remove rank-1 FV component at final cue token, prefill, "
                             "blocks in layer clamp; mean op adds back grand-mean proj",
               "selection_path": str(args.selection_path),
               "grand_mean": f"{gm_path.name} (equal-task-weighted, 69 tasks)",
               "conditions": {}}

        conds = []
        if args.with_baseline:
            conds.append((f"real{args.n_shots}_baseline", None, None, None))
        for cfg in args.layer_cfgs:
            layers = LAYER_CFGS[cfg]
            for who, u in (("own", u_own), ("cf", u_cf)):
                mproj = torch.outer(gm @ u, u)           # (N_LAYERS, D)
                conds.append((f"{who}_zero_{cfg}", u, None, layers))
                conds.append((f"{who}_mean_{cfg}", u, mproj, layers))

        for cname, u, mproj, layers in conds:
            ab.u, ab.mproj = u, mproj
            ab.layers = None if layers is None else frozenset(layers)
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
                                         max_new_tokens=max_new,
                                         pad_token_id=tok.eos_token_id)
                ab.mask = None
                for r, i in enumerate(b):
                    preds[i] = tok.decode(gen[r, L:], skip_special_tokens=True).split("\n")[0].strip()
            ab.u = ab.mproj = ab.layers = None
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | {cname}: acc={acc:.3f}", flush=True)
        res["golds"] = [it["gold"] for it in items]
        with open(out_path, "w") as f:
            json.dump(res, f)
    print("shard done", flush=True)


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks_all = sorted(group)
    tasks = tasks_all[args.shard_idx::args.shard_n]
    if args.stage == "combine":
        run_combine(args, tasks_all)
        return
    model, tok = load_model(args.model_dir)
    if args.stage == "means":
        run_means(args, model, tok, tasks)
    else:
        run_eval(args, model, tok, tasks, group)


if __name__ == "__main__":
    main()
