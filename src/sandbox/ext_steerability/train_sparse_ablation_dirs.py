#!/usr/bin/env python
"""Sparse ablation-direction optimization over the 139 dot_perhead-unit PCs (GPU).

Goal (user experiment 2026-08-16): find the smallest subset of the 139 pooled-90% centered
PC directions of the dot_perhead__unit read dirs whose MEAN-ABLATION at demo-label tokens
(all 28 block inputs, all 10-shot prompts) stops GPT-J from learning the task in context.

Gate: c in [0,1]^139, per direction j the label-token activation is moved c_j of the way
to the grand mean's projection:
    h <- h + sum_j c_j ((m_l - h) . v_j) v_j
(c_j = 1: direction fully mean-ablated -> IMPORTANT/selected; c_j = 0: untouched.)
Grand mean m_l = per-layer mean of demo-label-token block inputs over ALL 69 tasks
(user decision; grand_mean69.pt).

Loss (teacher-forced, fp32): + mean_batch[ log p(full gold label | ablated prompt) ]
+ lambda * ||c||_1  — minimizing pushes the model AWAY from the gold label with few
ablated directions. Adam on raw c, clamp to [0,1] after each step (Hu-style projected
update, repo convention); token-length-aware batches; gradient checkpointing; early
stopping on val loss.

Stages:
  --stage train : one lambda per invocation -> <out>/train/lambda_<l>.pt
                  (c vector, history, n_selected at 0.5)
  --stage eval  : hard mean-ablation (full replacement) of each lambda's selected set
                  (c > 0.5), T=1 sampled accuracy on the 150 prompts/task (same protocol
                  and seeds as the PC50 eval), plus a baseline condition; sharded over the
                  69 tasks -> <out>/eval/<task>.json

Fit data: all 69 tasks x (--n_train_prompts train + --n_val_prompts val) prompts each,
disjoint, seeded.
"""
import argparse
import json
import re
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
    from src.sandbox.ext_steerability.ablate_pc50_labeltokens import (  # noqa: E402
        load_model, prep_task, batches_by_len, N_LAYERS, D)
except ModuleNotFoundError:  # staged copy outside the repo tree: import the sibling file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ablate_pc50_labeltokens import (  # noqa: E402
        load_model, prep_task, batches_by_len, N_LAYERS, D)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("train", "eval"))
    p.add_argument("--lam", type=float, default=None, help="L1 weight (train stage)")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--pc_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "read_dir_sweep" / "dot_perhead_unit_pc139.pt")
    p.add_argument("--grand_mean_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "pc50_ablation" / "grand_mean69.pt")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "sparse_ablation_dirs")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--n_train_prompts", type=int, default=25)
    p.add_argument("--n_val_prompts", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--init_c", type=float, default=0.1)
    p.add_argument("--max_epochs", type=int, default=6)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--token_budget", type=int, default=9000)
    p.add_argument("--batch_cap", type=int, default=8)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--select_thresh", type=float, default=0.5)
    # eval stage
    p.add_argument("--lambdas", type=str, default=None,
                   help="comma list of lambda values whose train outputs to evaluate")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


class SoftMeanAblator:
    """Pre-hook on every block: h <- h + sum_j c_j ((m_l - h).v_j) v_j at masked positions.
    Pure function of (h, c, mask, means) -> safe under gradient-checkpoint recompute."""

    def __init__(self, model, V, means):
        self.V = V              # (K, D) fp32 cuda
        self.means_proj = means @ V.T   # (N_LAYERS, K) fp32 cuda
        self.c = None           # (K,) fp32 cuda (requires_grad in training)
        self.mask = None
        self.handles = [model.transformer.h[l].register_forward_pre_hook(
            self._make(l), with_kwargs=True) for l in range(N_LAYERS)]

    def _make(self, l):
        def hook(module, args, kwargs):
            in_args = bool(args)
            h = args[0] if in_args else kwargs["hidden_states"]
            if self.mask is None or self.c is None or h.shape[1] != self.mask.shape[1]:
                return None
            h32 = h[self.mask].float()                       # (N, D)
            coef = h32 @ self.V.T                            # (N, K)
            delta = (self.c * (self.means_proj[l] - coef)) @ self.V   # (N, D)
            hm = h[self.mask] + delta.to(h.dtype)
            h = h.clone()
            h[self.mask] = hm
            if in_args:
                return (h,) + args[1:], kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = h
            return args, kwargs
        return hook


def split_prompts(task, n_train, n_val, seed):
    rng = np.random.RandomState(seed + zlib.crc32(task.encode()) % 100000)
    perm = rng.permutation(150)
    return perm[:n_train].tolist(), perm[n_train:n_train + n_val].tolist()


def build_tf_batch(items, idxs, tok):
    """Teacher-forced batch: prompt + gold ids, right padding. Returns tensors + per-sample
    (gold_start, gold_len) and the demo-label ablation mask."""
    seqs, golds = [], []
    for i in idxs:
        it = items[i]
        gid = tok(" " + it["gold"]).input_ids
        seqs.append(it["ids"] + gid)
        golds.append((len(it["ids"]), len(gid)))
    L = max(len(s) for s in seqs)
    B = len(seqs)
    ids = torch.full((B, L), tok.eos_token_id, dtype=torch.long)
    att = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L, dtype=torch.bool)
    tgt = torch.full((B, L), -100, dtype=torch.long)
    for r, (i, s, (gs, gl)) in enumerate(zip(idxs, seqs, golds)):
        ids[r, :len(s)] = torch.tensor(s)
        att[r, :len(s)] = 1
        mask[r, items[i]["label_pos"]] = True
        tgt[r, gs:gs + gl] = torch.tensor(s[gs:gs + gl])
    return ids, att, mask, tgt


def tf_logp(model, ids, att, tgt):
    """Sum over gold tokens of log p(token), per sample -> (B,) fp32."""
    out = model(input_ids=ids, attention_mask=att, use_cache=False)
    logits = out.logits[:, :-1].float()
    tgt_s = tgt[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    sel = tgt_s.clamp(min=0)
    tok_lp = logp.gather(-1, sel.unsqueeze(-1)).squeeze(-1)
    tok_lp = tok_lp * (tgt_s != -100)
    return tok_lp.sum(dim=1)


def run_train(args, model, tok, tasks):
    assert args.lam is not None
    pcs = torch.load(args.pc_path, map_location="cpu", weights_only=False)
    V = pcs["V"].float().cuda()
    K = V.shape[0]
    gm = torch.load(args.grand_mean_path, map_location="cpu", weights_only=False)["mean"].float().cuda()
    # checkpointing is a no-op in eval mode, and reentrant checkpointing would drop grads
    # to the closure-captured c; GPT-J has all dropout probs 0.0, so train mode is safe.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    for prm in model.parameters():
        prm.requires_grad_(False)
    ab = SoftMeanAblator(model, V, gm)

    items_by_task, train_pool, val_pool = {}, [], []
    for t in tasks:
        items = prep_task(t, args.prompts_root, tok)
        items_by_task[t] = items
        tr, va = split_prompts(t, args.n_train_prompts, args.n_val_prompts, args.seed)
        train_pool += [(t, i) for i in tr]
        val_pool += [(t, i) for i in va]
    print(f"train pool {len(train_pool)} | val pool {len(val_pool)} | K={K}", flush=True)

    c = torch.full((K,), args.init_c, dtype=torch.float32, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([c], lr=args.lr)
    rng = np.random.RandomState(args.seed)

    def batches(pool, shuffle):
        order = rng.permutation(len(pool)) if shuffle else np.arange(len(pool))
        flat = [pool[i] for i in order]
        # group by rough length via batches_by_len on synthetic items
        proxy = [{"ids": items_by_task[t][i]["ids"]} for t, i in flat]
        for b in batches_by_len(proxy, args.token_budget, args.batch_cap):
            yield [flat[j] for j in b]

    def eval_val():
        tot, n = 0.0, 0
        with torch.no_grad():
            for group in batches(val_pool, shuffle=False):
                fake_items = [items_by_task[t][i] for t, i in group]
                ids, att, mask, tgt = build_tf_batch(fake_items, list(range(len(fake_items))), tok)
                ab.mask = mask.cuda()
                lp = tf_logp(model, ids.cuda(), att.cuda(), tgt.cuda())
                ab.mask = None
                tot += lp.sum().item()
                n += len(group)
        return tot / n

    history, best = [], (None, float("inf"), -1)
    for epoch in range(args.max_epochs):
        ep_lp, ep_n = 0.0, 0
        for group in batches(train_pool, shuffle=True):
            fake_items = [items_by_task[t][i] for t, i in group]
            ids, att, mask, tgt = build_tf_batch(fake_items, list(range(len(fake_items))), tok)
            ab.mask = mask.cuda()
            ab.c = c
            lp = tf_logp(model, ids.cuda(), att.cuda(), tgt.cuda())
            loss = lp.mean() + args.lam * c.abs().sum()
            opt.zero_grad()
            loss.backward()
            assert c.grad is not None and float(c.grad.abs().sum()) > 0, \
                "c received no gradient - checkpointing/hook wiring broken, HARD STOP"
            opt.step()
            with torch.no_grad():
                c.clamp_(0.0, 1.0)
            ab.mask = None
            ep_lp += lp.sum().item()
            ep_n += len(group)
        ab.c = c.detach()
        val_lp = eval_val()
        val_loss = val_lp + args.lam * float(c.detach().abs().sum())
        n_sel = int((c.detach() > args.select_thresh).sum())
        history.append({"epoch": epoch, "train_logp": ep_lp / ep_n, "val_logp": val_lp,
                        "val_loss": val_loss, "n_selected": n_sel,
                        "c_max": float(c.detach().max()), "c_sum": float(c.detach().sum())})
        print(f"lam={args.lam} epoch {epoch}: train_logp={ep_lp/ep_n:.3f} val_logp={val_lp:.3f} "
              f"val_loss={val_loss:.3f} n_sel={n_sel} c_max={history[-1]['c_max']:.3f}", flush=True)
        if val_loss < best[1] - 1e-4:
            best = (c.detach().clone().cpu(), val_loss, epoch)
        elif epoch - best[2] >= args.patience:
            print("early stop", flush=True)
            break
        ab.c = c  # re-arm grad-enabled gate for next epoch

    c_best = best[0]
    sel = (c_best > args.select_thresh).nonzero().squeeze(-1).tolist()
    out = {"lam": args.lam, "c": c_best, "best_epoch": best[2], "history": history,
           "selected": sel, "n_selected": len(sel), "select_thresh": args.select_thresh,
           "config": vars(args) | {"pc_path": str(args.pc_path)}}
    for k, v in list(out["config"].items()):
        if isinstance(v, Path):
            out["config"][k] = str(v)
    d = args.out_root / "train"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(out, d / f"lambda_{args.lam}.pt")
    print(f"lam={args.lam}: best_epoch={best[2]} n_selected={len(sel)} -> {d}", flush=True)


def run_eval(args, model, tok, tasks):
    pcs = torch.load(args.pc_path, map_location="cpu", weights_only=False)
    V_all = pcs["V"].float().cuda()
    gm = torch.load(args.grand_mean_path, map_location="cpu", weights_only=False)["mean"].float().cuda()
    lams = [float(x) for x in args.lambdas.split(",")]
    conds = [("baseline", None)]
    for lam in lams:
        tr = torch.load(args.out_root / "train" / f"lambda_{lam}.pt",
                        map_location="cpu", weights_only=False)
        idx = torch.tensor(tr["selected"], dtype=torch.long)
        conds.append((f"lam{lam}_n{len(idx)}", V_all[idx.cuda()] if len(idx) else None))
    ab = SoftMeanAblator(model, V_all, gm)  # placeholder; per-condition we swap V
    tok.padding_side = "left"
    outdir = args.out_root / "eval"
    outdir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        items = prep_task(task, args.prompts_root, tok)
        res = {"task": task, "n_prompts": len(items), "conditions": {}}
        for cname, Vsel in conds:
            if cname == "baseline" or Vsel is None:
                ab.c = None
            else:
                ab.V = Vsel
                ab.means_proj = gm @ Vsel.T
                ab.c = torch.ones(Vsel.shape[0], device="cuda")   # full mean ablation
            preds = [None] * len(items)
            for bi, b in enumerate(batches_by_len(items, args.token_budget, 16)):
                lens = [len(items[i]["ids"]) for i in b]
                L = max(lens)
                ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
                att = torch.zeros(len(b), L, dtype=torch.long)
                mask = torch.zeros(len(b), L, dtype=torch.bool)
                for r, i in enumerate(b):
                    n = lens[r]; off = L - n
                    ids[r, off:] = torch.tensor(items[i]["ids"])
                    att[r, off:] = 1
                    mask[r, [off + p for p in items[i]["label_pos"]]] = True
                ab.mask = mask.cuda()
                max_new = min(max(items[i]["gold_len"] for i in b) + 3, 16)
                torch.manual_seed(zlib.crc32(f"{task}|{'baseline' if ab.c is None else cname}|{bi}".encode()))
                with torch.no_grad():
                    gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                         do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                         max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
                ab.mask = None
                for r, i in enumerate(b):
                    preds[i] = tok.decode(gen[r, L:], skip_special_tokens=True).split("\n")[0].strip()
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | {cname}: acc={acc:.3f}", flush=True)
        res["golds"] = [it["gold"] for it in items]
        with open(outdir / f"{task}.json", "w") as f:
            json.dump(res, f)
    print("eval done", flush=True)


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    all_tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    model, tok = load_model(args.model_dir)
    if args.stage == "train":
        run_train(args, model, tok, all_tasks)
    else:
        run_eval(args, model, tok, all_tasks[args.shard_idx::args.shard_n])


if __name__ == "__main__":
    main()
