#!/usr/bin/env python
"""Bottom-up read-feature ablation baselines (69-task pool).

Does removing the single bottom-up read direction at the "read locations" stop GPT-J
learning the task in context? For n-shot prompts (n in {1, 6}; demos truncated from the
fixed 150 clean 10-shot train prompts per task), we edit EVERY demo-label token (all
tokens of each demo output; query cue untouched) at the residual stream entering EVERY
block (layers 0-27), then sample and score with the locked T=1 protocol.

Direction (user decision 2026-08-19): the FIXED unit-normed L6 raw label-token mean of
the task (label_resid_means/<task>.pt, row 6), projected out at all 28 layers.

Conditions (per n_shots):
  attnmask          : no residual edit; instead the final cue token AND every generated
                      token are blocked (pre-softmax -inf, all layers/heads) from
                      attending to the demo-label positions.
  mean_ablation     : h <- h - (h.d)d + (m_l.d)d   (m_l = grand mean of label-token
                      block-l inputs over ALL 69 tasks, pc50_ablation/grand_mean69.pt;
                      captured on 10-shot prompts — positions carry no additive
                      positional component under rotary, so reused for n-shot)
  zero_ablation     : h <- h - (h.d)d
  cf_mean_ablation  : mean_ablation but with d of an a-priori clearly-different task
                      (cf_task_pairs.json, LLM semantic-family pairing)
  cf_zero_ablation  : zero_ablation with the counterfactual direction

Metric: accuracy of temperature-1 sampled responses (one sample per prompt, stripped
first line == gold), deterministic seeds crc32(f"{task}|{cond}|{batch}") — same
protocol as ablate_pc50_labeltokens.py, so the unablated n-shot / 0-shot baselines from
steering_results/sixshot_dummy/per_task_acc.csv are directly comparable (prompt-string
equality with that script's f-string construction is asserted per prompt).

Output: <out_root>/n{n}shot/<task>.json  (accuracies + sampled predictions; resumable —
missing conditions are filled into an existing file).
Sharding: --shard_idx/--shard_n over the sorted 69 tasks.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# local bootstrap for in-repo runs; a PYTHONPATH-supplied repo also works (staged copies)
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
from src.utils.prompt_utils import get_token_meta_labels, word_pairs_to_prompt_data
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import (
    Ablator, LABEL_RE, batches_by_len, load_model)

N_LAYERS, D = 28, 4096
CONDITIONS = ("attnmask", "mean_ablation", "zero_ablation",
              "cf_mean_ablation", "cf_zero_ablation")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_shots", type=int, required=True, choices=(1, 6))
    p.add_argument("--conditions", type=str, default=",".join(CONDITIONS))
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--readdir_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means")
    p.add_argument("--grand_mean_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "pc50_ablation" / "grand_mean69.pt")
    p.add_argument("--pairs_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "cf_task_pairs.json")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=24000)
    p.add_argument("--batch_cap", type=int, default=48)
    p.add_argument("--task_set", choices=("train", "heldout", "all"), default="all")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=None, help="cap tasks (smoke tests)")
    return p.parse_args()


# ------------------- attention knockout (transformers 5.13 GPTJAttention._attn) -------------------
# GPTJAttention.forward always calls self._attn(query, key, value, attention_mask) — both at
# prefill (q_len == k_len) and at cached decode steps (q_len == 1) — so one patch covers the
# final cue token AND every generated token. `_ko` on each attn module holds k_idx [B, K]
# (absolute key positions of the demo-label tokens, left-pad offsets included; ragged rows
# padded by repeating the first index, masking the same entry twice is harmless).
def _attn_with_knockout(self, query, key, value, attention_mask=None):
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    attn_weights = attn_weights / self.scale_attn
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    k_idx = getattr(self, "_ko", None)
    if k_idx is not None:
        B, H, QL, _ = attn_weights.shape
        if QL == 1:                      # cached decode step: the single generated token
            q = torch.zeros(B, dtype=torch.long, device=attn_weights.device)
        else:                            # prefill: only the final cue row (left padding)
            q = torch.full((B,), QL - 1, dtype=torch.long, device=attn_weights.device)
        rows = torch.arange(B, device=attn_weights.device)[:, None]
        heads = torch.arange(H, device=attn_weights.device)[None, :]
        mn = torch.finfo(attn_weights.dtype).min
        for j in range(k_idx.shape[1]):
            attn_weights[rows, heads, q[:, None], k_idx[:, j][:, None]] = mn

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.to(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


def install_knockout_patch(model):
    from transformers.models.gptj.modeling_gptj import GPTJAttention
    GPTJAttention._attn = _attn_with_knockout
    for blk in model.transformer.h:
        blk.attn._ko = None


def set_knockout(model, k_idx):
    for blk in model.transformer.h:
        blk.attn._ko = k_idx


def clear_knockout(model):
    for blk in model.transformer.h:
        blk.attn._ko = None
# --------------------------------------------------------------------------------------------------


def prep_task_nshot(task, prompts_root, tok, n_shots):
    """Tokenize the 150 clean prompts truncated to n_shots demos; return items with ids,
    demo-label token positions, gold string, gold token length. Asserts the prompt string
    is byte-identical to the sixshot_dummy_steer.py f-string construction (baseline reuse)."""
    recs = json.load(open(prompts_root / task / "train_prompts.json"))
    assert len(recs) == 150
    out = []
    for rec in recs:
        demos = rec["demos"][:n_shots]
        assert len(demos) == n_shots
        wp = {"input": [str(d["input"]) for d in demos],
              "output": [str(d["output"]) for d in demos]}
        qo = rec["query"]["output"]
        qo = [str(x) for x in qo] if isinstance(qo, list) else str(qo)
        q = {"input": str(rec["query"]["input"]), "output": qo}
        pd_ = word_pairs_to_prompt_data(wp, query_target_pair=q, prepend_bos_token=False,
                                        shuffle_labels=False)
        # NOTE: no query= override — passing the raw (unspaced) query, as some older
        # scripts do, yields "Q:{query}" while the demos read "Q: {input}"; the stored
        # query_target already carries the prepend_space, matching the f-string prompts
        # the reused sixshot_dummy baselines were sampled on.
        token_labels, prompt_string = get_token_meta_labels(pd_, tok, prepend_bos=False)
        fstr = "".join(f"Q: {str(d['input'])}\nA: {str(d['output']).strip()}\n\n"
                       for d in demos) + f"Q: {q['input']}\nA:"
        assert prompt_string == fstr, \
            f"{task}: prompt differs from the f-string construction\n{prompt_string!r}\n{fstr!r}"
        ids = tok(prompt_string).input_ids
        assert len(ids) == len(token_labels)
        pos = [int(i) for i, _, lab in token_labels if LABEL_RE.match(lab)]
        assert len(pos) >= n_shots, f"{task}: only {len(pos)} label tokens found"
        gold = q["output"][0] if isinstance(q["output"], list) else q["output"]
        gold = str(gold).strip()
        out.append({"ids": ids, "label_pos": pos, "gold": gold,
                    "gold_len": len(tok(" " + gold).input_ids)})
    return out


def make_batch(items, b, tok):
    """Left-padded batch tensors + label mask + padded label-index matrix."""
    lens = [len(items[i]["ids"]) for i in b]
    L = max(lens)
    ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
    att = torch.zeros(len(b), L, dtype=torch.long)
    mask = torch.zeros(len(b), L, dtype=torch.bool)
    K = max(len(items[i]["label_pos"]) for i in b)
    kidx = torch.zeros(len(b), K, dtype=torch.long)
    for r, i in enumerate(b):
        n = lens[r]; off = L - n
        ids[r, off:] = torch.tensor(items[i]["ids"])
        att[r, off:] = 1
        pos = [off + p for p in items[i]["label_pos"]]
        mask[r, pos] = True
        kidx[r] = torch.tensor((pos + [pos[0]] * K)[:K], dtype=torch.long)
    return ids.cuda(), att.cuda(), mask.cuda(), kidx.cuda()


def load_unit_readdir(readdir_root, task):
    rm = torch.load(readdir_root / f"{task}.pt", map_location="cpu",
                    weights_only=False)["resid_means"][6].float()
    d = rm / rm.norm()
    assert abs(float(d @ d) - 1.0) < 1e-5
    return d.unsqueeze(0).cuda()          # (1, D) for the Ablator's V slot


# ---------------------------------- one-off correctness checks ----------------------------------
def verify_attn(model, tok, items, budget, cap):
    b = batches_by_len(items, budget, cap)[0][:4]
    ids, att, mask, kidx = make_batch(items, b, tok)
    set_knockout(model, kidx)
    out = model(input_ids=ids, attention_mask=att, output_attentions=True, use_cache=True)
    aw = out.attentions[12].float()                          # (B, H, L, L)
    rows = torch.arange(len(b), device=ids.device)
    cue = torch.full((len(b),), ids.shape[1] - 1, device=ids.device)
    cut = aw[rows[:, None], :, cue[:, None], kidx]           # weights on knocked keys
    rowsum = aw[rows, :, -1, :].sum(-1)
    assert float(cut.abs().max()) < 1e-6, f"prefill knockout leak: {float(cut.abs().max())}"
    assert float((rowsum - 1).abs().max()) < 1e-3, "prefill row not renormalized"
    # one manual cached decode step: the generated token must also be blocked
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    att2 = torch.cat([att, torch.ones(len(b), 1, dtype=att.dtype, device=att.device)], dim=1)
    pos_ids = att.sum(1, keepdim=True)                       # next rotary position per row
    out2 = model(input_ids=nxt, attention_mask=att2, past_key_values=out.past_key_values,
                 position_ids=pos_ids, output_attentions=True)
    aw2 = out2.attentions[12].float()                        # (B, H, 1, L+1)
    cut2 = aw2[rows[:, None], :, 0, kidx]
    rowsum2 = aw2[rows, :, 0, :].sum(-1)
    assert float(cut2.abs().max()) < 1e-6, f"decode-step knockout leak: {float(cut2.abs().max())}"
    assert float((rowsum2 - 1).abs().max()) < 1e-3, "decode row not renormalized"
    clear_knockout(model)
    print(f"verify_attn OK: prefill cut {float(cut.abs().max()):.1e}, "
          f"decode cut {float(cut2.abs().max()):.1e}", flush=True)


def verify_ablation(model, ab, tok, items, V, mproj, budget, cap, layer=10):
    """Temp pre-hook registered AFTER the Ablator's sees the already-edited block input;
    projection onto d must equal the mean coefficient (mean mode) or 0 (zero mode)."""
    got = {}

    def probe(module, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        if ab.mask is not None and h.shape[1] == ab.mask.shape[1]:
            got["proj"] = (h[ab.mask].float() @ V[0])
        return None

    handle = model.transformer.h[layer].register_forward_pre_hook(probe, with_kwargs=True)
    b = batches_by_len(items, budget, cap)[0][:4]
    ids, att, mask, _ = make_batch(items, b, tok)
    ab.V, ab.mproj, ab.mask = V, mproj, mask
    with torch.no_grad():
        model(input_ids=ids, attention_mask=att, use_cache=False)
    ab.V = ab.mproj = ab.mask = None
    handle.remove()
    proj = got["proj"]
    want = 0.0 if mproj is None else float(mproj[layer] @ V[0])
    dev = float((proj - want).abs().max())
    assert dev < 0.05, f"ablation residue at L{layer}: max |proj-{want:.3f}| = {dev:.4f}"
    print(f"verify_ablation OK (L{layer}, {'mean' if mproj is not None else 'zero'}): "
          f"target {want:.3f}, max dev {dev:.1e}", flush=True)
# ------------------------------------------------------------------------------------------------


def main():
    args = parse_args()
    conds = [c for c in args.conditions.split(",") if c]
    assert all(c in CONDITIONS for c in conds), conds
    split = json.load(open(args.split_path))
    pool = {"train": split["train_tasks"], "heldout": split["heldout_tasks"],
            "all": split["train_tasks"] + split["heldout_tasks"]}[args.task_set]
    tasks = sorted(pool)[args.shard_idx::args.shard_n]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    pairs = json.load(open(args.pairs_path))["pairs"]
    grand = torch.load(args.grand_mean_path, map_location="cpu",
                       weights_only=False)["mean"].float().cuda()   # (N_LAYERS, D)

    model, tok = load_model(args.model_dir)
    install_knockout_patch(model)
    ab = Ablator(model)
    tok.padding_side = "left"
    outdir = args.out_root / f"n{args.n_shots}shot"
    outdir.mkdir(parents=True, exist_ok=True)

    verified = False
    for task in tasks:
        outpath = outdir / f"{task}.json"
        res = json.load(open(outpath)) if outpath.exists() else None
        todo = [c for c in conds if res is None or c not in res["conditions"]]
        if not todo:
            print(f"{task}: all conditions present, skip", flush=True)
            continue
        items = prep_task_nshot(task, args.prompts_root, tok, args.n_shots)
        d_own = load_unit_readdir(args.readdir_root, task)
        d_cf = load_unit_readdir(args.readdir_root, pairs[task])
        setup = {
            "attnmask":         (None, None, True),
            "mean_ablation":    (d_own, (grand @ d_own.T) @ d_own, False),
            "zero_ablation":    (d_own, None, False),
            "cf_mean_ablation": (d_cf, (grand @ d_cf.T) @ d_cf, False),
            "cf_zero_ablation": (d_cf, None, False),
        }
        if not verified:
            verify_attn(model, tok, items, args.token_budget, args.batch_cap)
            verify_ablation(model, ab, tok, items, d_own,
                            (grand @ d_own.T) @ d_own, args.token_budget, args.batch_cap)
            verify_ablation(model, ab, tok, items, d_own, None,
                            args.token_budget, args.batch_cap)
            verified = True
        if res is None:
            res = {"task": task, "n_shots": args.n_shots, "cf_task": pairs[task],
                   "n_prompts": len(items), "conditions": {},
                   "golds": [it["gold"] for it in items]}
        for cname in todo:
            V, mproj, use_ko = setup[cname]
            preds = [None] * len(items)
            for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
                ids, att, mask, kidx = make_batch(items, b, tok)
                if use_ko:
                    set_knockout(model, kidx)
                else:
                    ab.V, ab.mproj, ab.mask = V, mproj, mask
                max_new = min(max(items[i]["gold_len"] for i in b) + 3, 16)
                torch.manual_seed(zlib.crc32(f"{task}|{cname}|{bi}".encode()))
                with torch.no_grad():
                    gen = model.generate(input_ids=ids, attention_mask=att,
                                         do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                         max_new_tokens=max_new,
                                         pad_token_id=tok.eos_token_id)
                clear_knockout(model)
                ab.V = ab.mproj = ab.mask = None
                for r, i in enumerate(b):
                    new = gen[r, ids.shape[1]:]
                    preds[i] = tok.decode(new, skip_special_tokens=True).split("\n")[0].strip()
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | n{args.n_shots} | {cname}: acc={acc:.3f}", flush=True)
        with open(outpath, "w") as f:
            json.dump(res, f)
    print("eval done", flush=True)


if __name__ == "__main__":
    main()
