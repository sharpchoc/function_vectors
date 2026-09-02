#!/usr/bin/env python
"""Sparse head selection for style-property steering (per property, NO train/test split —
user decision 2026-09-01).

Candidate vector: v(c) = sum_h c_h * (W_O^h @ head_diff[h]), c in [0,1]^448, where
head_diff = per-head mean out_proj input at evidence tokens, alt − nat
(capture_head_means.py), lifted by build_contributions_single.

Objective (differentiable optimizer internals only — the reported readout stays sampled
adherence): teacher-forced −log p(property-consistent ALT continuation | nat-twin prefix
through the cue token), with v(c) injected additively at the prefix's prior
evidence positions at one block, + lam * ||c||_1.

Selection: c > 0.8, top-10 fallback (repo convention). Head-sum steering vector =
unweighted sum of selected heads' contributions (the FV convention).

Output: artifacts/style_properties/sparse_heads/<prop>.npz
  {c, selected (flat idx l*16+h), v_headsum [4096], inject_block, history}
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
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
from src.sandbox.isolation_upper_bound.run_task import build_contributions_single
from src.sandbox.ext_styleprops.steer_adherence import build_items
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len

HEAD_DIR = ARTIFACTS_ROOT / "style_properties" / "head_means"
SWEEP_DIR = ARTIFACTS_ROOT / "style_properties" / "steering" / "sweep"
OUT_DIR = ARTIFACTS_ROOT / "style_properties" / "sparse_heads"
POOL_PATH = REPO_ROOT / "task_splits" / "style_properties_pool.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--props", nargs="*", default=None)
    p.add_argument("--model_name", default="EleutherAI/gpt-j-6b")
    p.add_argument("--lam", type=float, default=0.002)
    p.add_argument("--lam_warmup", type=int, default=2,
                   help="epochs with lam=0 so weak steering gradients aren't crushed "
                        "before c finds signal (ampersand collapsed to 0 gates otherwise)")
    p.add_argument("--train_alpha", type=float, default=4.0,
                   help="fixed dose multiplier on v(c) during training; the sweep shows "
                        "steering typically ignites at alpha 2-16")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--init_c", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--max_items", type=int, default=600)
    p.add_argument("--c_high", type=float, default=0.8)
    p.add_argument("--docs", type=int, default=120)
    p.add_argument("--site", choices=("evid", "cue"), default="evid",
                   help="cue = FV analog: head means at the cue token, v(c) injected at the cue")
    return p.parse_args()


def best_sweep_layer(name, site="evid"):
    """Capture layer of the best mean-diff sweep condition for this site (fallback 10)."""
    path = (SWEEP_DIR if site == "evid" else SWEEP_DIR.parent / "sweep_cue") / f"{name}.json"
    if not path.exists():
        return 10
    conds = json.load(open(path))["conditions"]
    valid = [c for c in conds if ("diff_" in c) and not np.isnan(conds[c]["adherence_tgt"])]
    if not valid:
        return 10
    best = max(valid, key=lambda c: conds[c]["adherence_tgt"])
    return int(best.split("_L")[1].split("_")[0])


def nll_batch(model, batch, v, inject_block, device, site="evid"):
    """Teacher-forced −log p(alt continuation) per item; v [4096] fp32 differentiable,
    injected at each item's evidence positions at inject_block's output."""
    seqs = [b["ids"] + b["cont_ids"] for b in batch]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(batch), L), 50256, dtype=torch.long)
    att = torch.zeros(len(batch), L, dtype=torch.long)
    add_mask = torch.zeros(len(batch), L, dtype=torch.bool)
    for i, (b, s) in enumerate(zip(batch, seqs)):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        for p_ in (b["evid_pos"] if site == "evid" else [b["cue_pos"]]):
            add_mask[i, p_] = True
    ids, att, add_mask = ids.to(device), att.to(device), add_mask.to(device)

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        add = torch.zeros_like(hidden, dtype=torch.float32)
        add[add_mask] = v
        hidden = hidden + add.to(hidden.dtype)
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

    h = model.transformer.h[inject_block].register_forward_hook(hook)
    try:
        logits = model(input_ids=ids, attention_mask=att).logits
    finally:
        h.remove()
    nll = 0.0
    for i, b in enumerate(batch):
        p0 = len(b["ids"])
        # slice BEFORE the fp32 cast: a full-batch fp32 logits copy OOMs when the
        # injection block is early (deep backprop already holds ~10 GB of activations)
        lp = torch.log_softmax(logits[i, p0 - 1:p0 - 1 + len(b["cont_ids"])].float(), dim=-1)
        tgt = torch.tensor(b["cont_ids"], device=device)
        nll = nll - lp.gather(1, tgt[:, None]).sum()
    return nll / len(batch)


def main():
    args = parse_args()
    props = args.props or sorted(json.load(open(POOL_PATH))["pass"])
    head_dir = HEAD_DIR if args.site == "evid" else HEAD_DIR.parent / "head_means_cue"
    out_dir = OUT_DIR if args.site == "evid" else OUT_DIR.parent / "sparse_heads_cue"
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok, mc = load_gpt_model_and_tokenizer(args.model_name)
    model.requires_grad_(False)
    device = model.device

    for name in props:
        out = out_dir / f"{name}.npz"
        if out.exists():
            print(f"{name}: exists, skip", flush=True)
            continue
        hd = torch.load(head_dir / f"{name}.pt", weights_only=False)
        C = build_contributions_single(hd["head_diff"], model, mc)   # [448, 4096] fp32
        cap_layer = best_sweep_layer(name, args.site)
        inject_block = cap_layer - 1
        items = build_items(name, tok, "nat", args.docs, 8, require_evid=(args.site == "evid"))
        for it in items:
            # cap the loss to the property-committing region: long continuations
            # (all_caps = a whole uppercase sentence) dilute the objective with
            # unpredictable content tokens
            it["cont_ids"] = tok(it["exp"]["alt"]).input_ids[:8]
        items = [it for it in items if it["cont_ids"]][:args.max_items]
        print(f"{name}: {len(items)} items, inject block {inject_block} "
              f"(capture layer {cap_layer})", flush=True)

        c = torch.full((448,), args.init_c, device=device, dtype=torch.float32,
                       requires_grad=True)
        opt = torch.optim.AdamW([c], lr=args.lr)
        rng = np.random.RandomState(zlib.crc32(name.encode()) % 2**31)
        history = []
        # length-bucketed batches; budget shrinks with backprop depth (early inject
        # block = activations held for nearly the whole stack; GPT-J attention runs
        # fp32 internally, so long sequences dominate memory)
        if inject_block < 8:
            budget, cap = 1500, 4
        elif inject_block < 16:
            budget, cap = 2400, 6
        else:
            budget, cap = 3500, 8
        buckets = batches_by_len(items, budget, cap)
        first = True
        for epoch in range(args.epochs):
            order = rng.permutation(len(buckets))
            ep_nll, nb = 0.0, 0
            for bidx in order:
                batch = [items[i] for i in buckets[bidx]]
                opt.zero_grad(set_to_none=True)
                v = args.train_alpha * torch.einsum("h,hd->d", c, C)
                lam = 0.0 if epoch < args.lam_warmup else args.lam
                loss = nll_batch(model, batch, v, inject_block, device, args.site) \
                    + lam * c.abs().sum()
                loss.backward()
                if first:
                    assert c.grad is not None and torch.isfinite(c.grad).all() \
                        and c.grad.abs().sum() > 0, "no gradient reached c"
                    first = False
                opt.step()
                with torch.no_grad():
                    c.clamp_(0.0, 1.0)
                ep_nll += float(loss.detach())
                nb += 1
            n_act = int((c.detach() > args.c_high).sum())
            history.append({"epoch": epoch, "nll": ep_nll / max(nb, 1),
                            "n_active": n_act, "l1": float(c.detach().sum())})
            print(f"  {name} epoch {epoch}: loss={history[-1]['nll']:.4f} "
                  f"active={n_act} l1={history[-1]['l1']:.1f}", flush=True)

        cd = c.detach()
        sel = torch.nonzero(cd > args.c_high).flatten()
        if len(sel) == 0:
            sel = torch.topk(cd, 10).indices
        v_headsum = C[sel].sum(0)
        np.savez(out, c=cd.cpu().numpy(), selected=sel.cpu().numpy(),
                 v_headsum=v_headsum.cpu().numpy(), inject_block=inject_block,
                 cap_layer=cap_layer, lam=args.lam, train_alpha=args.train_alpha,
                 history=json.dumps(history))
        print(f"{name}: {len(sel)} heads selected, |v|={float(v_headsum.norm()):.1f}",
              flush=True)
        torch.cuda.empty_cache()
    print("sparse opt done", flush=True)


if __name__ == "__main__":
    main()
