#!/usr/bin/env python
"""PC50 label-token ablation eval — causal comparison of read-direction definitions.

For each sweep bracket (cosine_M, dot_M, cosine_perhead, dot_perhead), take its top-50
UNCENTERED pooled-PC subspace (compute_sweep_pc50.py) and ablate that subspace from the
residual stream entering EVERY transformer block (layers 0-27), at every DEMO-LABEL token
position of the fixed 150 clean 10-shot train prompts per task (all tokens of each of the
10 demo outputs; the final query cue is NOT touched). Two ablation modes per bracket:
  zero : h <- h - P h
  mean : h <- h - P h + P m_l   (m_l = grand mean of label-token block-l inputs over ALL
                                 55 train tasks, from --stage means)
Metric: accuracy of temperature-1 sampled responses (one sample per prompt, exact match of
the stripped first line against the gold label), vs the unablated baseline sampled with the
same seeds. 9 conditions total per task.

Stages:
  --stage means : forward-only pass per task; save per-task label-token sums/counts of
                  block inputs -> <out>/label_means/<task>.pt (combine externally).
  --stage eval  : requires --grand_mean_path; runs baseline + 8 ablation conditions,
                  writes <out>/eval/<task>.json (accuracies + sampled predictions).

Sharding: --shard_idx/--shard_n over the sorted 55 train tasks.
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
from src.utils.prompt_utils import get_token_meta_labels, word_pairs_to_prompt_data
from src.utils.model_utils import get_decoder_block

BRACKETS = ("cosine_M", "dot_M", "cosine_perhead", "dot_perhead")
LABEL_RE = re.compile(r"^demonstration_\d+_label_token$")
N_LAYERS, D = 28, 4096          # GPT-J constants (kept for legacy imports); use model_dims()
MIN_PROMPTS = 40                # per-task prompt count gate (150 for GPT-J; 52 for two Qwen tasks)


def model_dims(model):
    """(n_layers, d_model, n_heads) from the HF config (GPT-J or Qwen2-style names)."""
    c = model.config
    n_layers = getattr(c, "num_hidden_layers", None) or getattr(c, "n_layer")
    d = getattr(c, "hidden_size", None) or getattr(c, "n_embd")
    n_heads = getattr(c, "num_attention_heads", None) or getattr(c, "n_head")
    return int(n_layers), int(d), int(n_heads)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("means", "combine", "eval"))
    p.add_argument("--model_name", default=None,
                   help="HF id for non-GPT-J models (e.g. Qwen/Qwen2.5-7B-Instruct, bf16); "
                        "default: the GPT-J-6B snapshot (fp16)")
    p.add_argument("--grand_mean_name", default="grand_mean.pt",
                   help="--stage combine: output file name under --out_root")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--pc_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "read_dir_sweep" / "pc50_uncentered.pt")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "pc50_ablation")
    p.add_argument("--grand_mean_path", type=Path, default=None)
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=11000)
    p.add_argument("--batch_cap", type=int, default=16)
    p.add_argument("--task_set", choices=("train", "heldout", "all"), default="train")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def load_model(model_dir, model_name=None, attn_implementation=None):
    """GPT-J-6B fp16 from the local snapshot (default), or any HF causal LM by id in bf16
    (Qwen2.5 port). `attn_implementation` is passed through when given."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    extra = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
    if model_name is None or "gpt-j" in str(model_name).lower():
        md = model_dir or sorted(Path("/workspace/.cache/huggingface/hub/"
                                      "models--EleutherAI--gpt-j-6b/snapshots").glob("*"))[-1]
        tok = AutoTokenizer.from_pretrained(md)
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.float16,
                                                     **extra).cuda().eval()
        return model, tok
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                 **extra).cuda().eval()
    return model, tok


def prep_task(task, prompts_root, tok):
    """Tokenize the clean 10-shot prompts (150 per task; >= MIN_PROMPTS); return list of dicts
    with ids, label-token positions, gold string, gold token length."""
    recs = json.load(open(prompts_root / task / "train_prompts.json"))
    assert len(recs) >= MIN_PROMPTS, f"{task}: only {len(recs)} prompts"
    out = []
    for rec in recs:
        # str-cast everything: number tasks store ints, which break tokenize_labels;
        # f-string prompt building renders ints and their str() identically.
        wp = {"input": [str(d["input"]) for d in rec["demos"]],
              "output": [str(d["output"]) for d in rec["demos"]]}
        qo = rec["query"]["output"]
        qo = [str(x) for x in qo] if isinstance(qo, list) else str(qo)
        q = {"input": str(rec["query"]["input"]), "output": qo}
        pd_ = word_pairs_to_prompt_data(wp, query_target_pair=q, prepend_bos_token=False,
                                        shuffle_labels=False)
        token_labels, prompt_string = get_token_meta_labels(pd_, tok, query=q["input"],
                                                            prepend_bos=False)
        ids = tok(prompt_string).input_ids
        assert len(ids) == len(token_labels)
        pos = [int(i) for i, _, lab in token_labels if LABEL_RE.match(lab)]
        assert len(pos) >= 10, f"{task}: only {len(pos)} label tokens found"
        gold = q["output"][0] if isinstance(q["output"], list) else q["output"]
        gold = str(gold).strip()   # number tasks store labels as ints
        gold_len = len(tok(" " + gold).input_ids)
        out.append({"ids": ids, "label_pos": pos, "gold": gold, "gold_len": gold_len})
    return out


def batches_by_len(items, budget, cap):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    bs, cur, cur_max = [], [], 0
    for i in order:
        L = len(items[i]["ids"])
        m = max(cur_max, L)
        if cur and (len(cur) + 1) * m > budget or len(cur) >= cap:
            bs.append(cur); cur, cur_max = [], 0
            m = L
        cur.append(i); cur_max = m
    if cur:
        bs.append(cur)
    return bs


class Ablator:
    """forward_pre_hook on every block; projects the subspace out of label positions.
    Active only when the sequence length matches the armed mask (prefill)."""

    def __init__(self, model):
        self.n_layers, self.d, _ = model_dims(model)
        self.V = None          # (k, D) fp32 cuda
        self.mproj = None      # (n_layers, D) fp32 cuda or None (zero mode)
        self.mask = None       # (B, L) bool cuda
        self.capture = None    # dict(sums (n_layers, D) fp64, count int) when capturing
        self.handles = [get_decoder_block(model, l).register_forward_pre_hook(
            self._make(l), with_kwargs=True) for l in range(self.n_layers)]

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
            if self.V is None:
                return None
            h32 = h[self.mask].float()
            h32 = h32 - (h32 @ self.V.T) @ self.V
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


def run_means(args, model, tok, tasks):
    outdir = args.out_root / "label_means"
    outdir.mkdir(parents=True, exist_ok=True)
    ab = Ablator(model)
    for task in tasks:
        if (outdir / f"{task}.pt").exists():
            print(f"means {task}: exists, skip", flush=True)
            continue
        items = prep_task(task, args.prompts_root, tok)
        cap = {"sums": torch.zeros(ab.n_layers, ab.d, dtype=torch.float64), "count": 0}
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
                mask[r, items[i]["label_pos"]] = True
            ab.mask = mask.cuda()
            with torch.no_grad():
                model(input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False)
            ab.mask = None
        ab.capture = None
        torch.save(cap, outdir / f"{task}.pt")
        print(f"means {task}: {cap['count']} label tokens", flush=True)


def run_eval(args, model, tok, tasks):
    pcs = torch.load(args.pc_path, map_location="cpu", weights_only=False)["brackets"]
    gm = torch.load(args.grand_mean_path, map_location="cpu", weights_only=False)
    grand = gm["mean"].float().cuda()             # (N_LAYERS, D)
    conds = [("baseline", None, None)]
    for br in BRACKETS:
        V = pcs[br]["V"].float().cuda()
        mproj = (grand @ V.T) @ V                  # (N_LAYERS, D)
        conds.append((f"{br}__zero", V, None))
        conds.append((f"{br}__mean", V, mproj))
    outdir = args.out_root / "eval"
    outdir.mkdir(parents=True, exist_ok=True)
    ab = Ablator(model)
    tok.padding_side = "left"
    for task in tasks:
        items = prep_task(task, args.prompts_root, tok)
        res = {"task": task, "n_prompts": len(items), "conditions": {}}
        for cname, V, mproj in conds:
            ab.V, ab.mproj = V, mproj
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
                    mask[r, [off + p for p in items[i]["label_pos"]]] = True
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
                    new = gen[r, L:]
                    preds[i] = tok.decode(new, skip_special_tokens=True).split("\n")[0].strip()
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | {cname}: acc={acc:.3f}", flush=True)
        res["golds"] = [it["gold"] for it in items]
        with open(outdir / f"{task}.json", "w") as f:
            json.dump(res, f)
    print("eval done", flush=True)


def run_combine(args, tasks_all):
    """Token-weighted grand mean of demo-label-token block inputs over --task_set (the
    GPT-J grand_mean69.pt was combined the same way, inline). No model needed."""
    sums, count, n_tasks = None, 0, 0
    for t in tasks_all:
        d = torch.load(args.out_root / "label_means" / f"{t}.pt", map_location="cpu",
                       weights_only=False)
        sums = d["sums"].double() if sums is None else sums + d["sums"].double()
        count += int(d["count"]); n_tasks += 1
    assert count > 0
    out = args.out_root / args.grand_mean_name
    torch.save({"mean": (sums / count).float(), "n_label_tokens": count, "n_tasks": n_tasks,
                "task_set": args.task_set,
                "definition": "token-weighted grand mean of block-input residuals at every "
                              f"demo-label token of the clean 10-shot prompts, {n_tasks} tasks "
                              f"({args.task_set})"}, out)
    print(f"combined {n_tasks} tasks, {count} label tokens -> {out}")


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    pool = {"train": split["train_tasks"], "heldout": split["heldout_tasks"],
            "all": split["train_tasks"] + split["heldout_tasks"]}[args.task_set]
    tasks = sorted(pool)[args.shard_idx::args.shard_n]
    if args.stage == "combine":
        run_combine(args, sorted(pool))
        return
    model, tok = load_model(args.model_dir, args.model_name)
    if args.stage == "means":
        run_means(args, model, tok, tasks)
    else:
        assert args.grand_mean_path is not None
        run_eval(args, model, tok, tasks)


if __name__ == "__main__":
    main()
