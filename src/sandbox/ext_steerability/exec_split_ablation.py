#!/usr/bin/env python
"""Execution-side carrier / task-unique split ablation at the query cue (Qwen fairness test, 2026-09-23).

Each task FV v_A (mean per-prompt FV) is split with a carrier estimated from TRAINING tasks only:
    c = mean_{A in train} v_A,  c_hat = c/|c|,   u_A = v_A - (v_A . c_hat) c_hat,   u_hat_A = u_A/|u_A|.
On real six-shot prompts (the paper's execution-ablation bank), a subspace Q is ablated at the FINAL query cue at the
inputs of blocks 9..27 (prefill), exactly as the paper's FV ablation but rank-k:
    zero: h <- h - (h Q^T) Q          mean: h <- h - (h Q^T) Q + (g_l Q^T) Q   (g_l = grand mean cue activation, block l)
Conditions: baseline | uniq_own_{mean,zero} | uniq_cf_mean | carrier_{mean,zero} | span2_own_mean | span2_cf_mean
(cf = the paired other-family task, as in the paper). T=1 sampled exact match, seeding crc32(task|cond|batch).
Output: <out_root>/<task>.json
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
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT  # noqa: E402
from src.utils.model_utils import get_decoder_block  # noqa: E402
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import load_model, batches_by_len, model_dims  # noqa: E402
from src.sandbox.ext_steerability.sixshot_dummy_steer import build_items_6shot  # noqa: E402

CONDS = ("baseline", "uniq_own_mean", "uniq_own_zero", "uniq_cf_mean", "carrier_mean", "carrier_zero",
         "span2_own_mean", "span2_cf_mean")


class SubspaceAblator:
    def __init__(self, model, layers):
        self.n_layers, self.d, _ = model_dims(model)
        self.layers = frozenset(layers)
        self.Q = None; self.mproj = None; self.mask = None
        for l in range(self.n_layers):
            get_decoder_block(model, l).register_forward_pre_hook(self._make(l), with_kwargs=True)

    def _make(self, l):
        def hook(module, args, kwargs):
            in_args = bool(args)
            h = args[0] if in_args else kwargs["hidden_states"]
            if self.Q is None or self.mask is None or l not in self.layers or h.shape[1] != self.mask.shape[1]:
                return None
            h32 = h[self.mask].float()
            h32 = h32 - (h32 @ self.Q.T) @ self.Q
            if self.mproj is not None:
                h32 = h32 + self.mproj[l]
            h = h.clone(); h[self.mask] = h32.to(h.dtype)
            if in_args:
                return (h,) + args[1:], kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = h
            return args, kwargs
        return hook


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts_root", type=Path, default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--grand_mean_cue", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "FV_ablation" / "grand_mean_cue6.pt")
    p.add_argument("--cf_pairs_path", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "cf_task_pairs.json")
    p.add_argument("--split_path", type=Path, default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--out_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "exec_split_ablation")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--model_name", default=None)
    p.add_argument("--token_budget", type=int, default=12000)
    p.add_argument("--batch_cap", type=int, default=16)
    p.add_argument("--task_stride", type=int, default=1)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    all_tasks = sorted(group)
    tasks = all_tasks[::args.task_stride][args.shard_idx::args.shard_n]
    args.out_root.mkdir(parents=True, exist_ok=True)
    pairs = json.load(open(args.cf_pairs_path))["pairs"]
    V = {t: torch.load(args.fv_root / f"{t}.pt", map_location="cpu", weights_only=False)["fv"].float().mean(0)
         for t in all_tasks}
    c = torch.stack([V[t] for t in split["train_tasks"]]).mean(0)
    ch = (c / c.norm()).cuda()

    def uniq(t):
        v = V[t].cuda(); u = v - (v @ ch) * ch
        return u / u.norm()

    model, tok = load_model(args.model_dir, args.model_name)
    tok.padding_side = "left"
    ab = SubspaceAblator(model, range(9, model_dims(model)[0]))
    g = torch.load(args.grand_mean_cue, map_location="cpu", weights_only=False)["mean"].float().cuda()   # (n_layers, D)
    print(f"{len(tasks)} tasks on this shard", flush=True)
    for task in tasks:
        op = args.out_root / f"{task}.json"
        if op.exists():
            print(f"{task}: exists, skip", flush=True); continue
        items = build_items_6shot(task, args.prompts_root, tok, real_labels=True)
        uo, uc = uniq(task), uniq(pairs[task])

        def basis(*vecs):
            return torch.linalg.qr(torch.stack(vecs, 1))[0].T.contiguous()     # (k, D) orthonormal

        spec = {"baseline": (None, False), "uniq_own_mean": (basis(uo), True), "uniq_own_zero": (basis(uo), False),
                "uniq_cf_mean": (basis(uc), True), "carrier_mean": (basis(ch), True), "carrier_zero": (basis(ch), False),
                "span2_own_mean": (basis(ch, uo), True), "span2_cf_mean": (basis(ch, uc), True)}
        res = {"task": task, "group": group[task], "cf_task": pairs[task], "n_prompts": len(items),
               "cos_uniq_own_cf": round(float(uo @ uc), 4), "cos_fv_carrier": round(float(V[task].cuda() @ ch / V[task].norm()), 4),
               "model_name": args.model_name or "EleutherAI/gpt-j-6b", "conditions": {}}
        for cname in CONDS:
            Q, mean = spec[cname]
            ab.Q = Q; ab.mproj = ((g @ Q.T) @ Q) if (Q is not None and mean) else None
            preds = [None] * len(items)
            for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
                lens = [len(items[i]["ids"]) for i in b]; L = max(lens)
                ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
                att = torch.zeros(len(b), L, dtype=torch.long); mask = torch.zeros(len(b), L, dtype=torch.bool)
                for r, i in enumerate(b):
                    off = L - lens[r]; ids[r, off:] = torch.tensor(items[i]["ids"]); att[r, off:] = 1; mask[r, L - 1] = True
                ab.mask = mask.cuda()
                max_new = min(max(items[i]["gold_len"] for i in b) + 3, 16)
                torch.manual_seed(zlib.crc32(f"{task}|{cname}|{bi}".encode()))
                with torch.no_grad():
                    gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), do_sample=True, temperature=1.0,
                                         top_k=0, top_p=1.0, max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
                ab.mask = None
                for r, i in enumerate(b):
                    preds[i] = tok.decode(gen[r, L:], skip_special_tokens=True).split("\n")[0].strip()
            ab.Q = ab.mproj = None
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | {cname}: acc={acc:.3f}", flush=True)
        res["golds"] = [it["gold"] for it in items]
        json.dump(res, open(op, "w"))
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
