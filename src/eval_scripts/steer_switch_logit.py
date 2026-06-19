#!/usr/bin/env python
"""Logit-readout task-switch steering with a clean TRAIN/TEST split (GPT-J-6B).

For single-token-gold pairs (antonym<->synonym, next_number_digits<->prev_number_digits) we steer a
source-task 1-shot prompt toward the target task and read, in ONE forward pass, the contrast
    logit(target_gold) - logit(source_gold)   at the final query position.
alpha=0 (clean, no injection) is the flat baseline (expected negative: the source-task prompt favors
the source answer); injecting sign*alpha*Delta_site(L) should push it up. No sampling, no judge.

CLEAN SPLIT (no leakage in either the ICL example or the final query):
  - LABEL pool  = shared single-token OUTPUT words  -> 100 reserved as test (ICL-example labels), rest train.
  - QUERY pool  = shared INPUT words w/ single-token gold under BOTH tasks -> 100 reserved as test
                  (final queries), rest train.
  - Delta is derived from n_train=100 TRAIN prompt PAIRS (label in train_out x query in train_in),
    reading act(f1)-act(f2) at the demo label token (-> Delta_label) and the query final token
    (-> Delta_final). Same n_train for both task pairs -> comparable.
  - Eval = n_test=100 TEST prompt pairs (test label <-> test query, random 1-to-1). Their label/query
    tokens never appear in the train pairs that build Delta.

Delta is derived here by our own forward passes (NOT the contaminated full-pool capture).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))

from utils.prompt_utils import get_token_meta_labels  # noqa: E402
from steer_label_to_query import (  # noqa: E402
    build_prompt_data, extract_positions, build_input_to_outputs, build_output_to_inputs,
    is_single_space_token, load_task_json, get_answer_id,
)
from steer_switch_judge import DIRECTIONS  # noqa: E402
from utils.paths import LABEL_GEOMETRY_DIR  # noqa: E402

LOGIT_DIRECTIONS = [
    "synonym_to_antonym", "antonym_to_synonym",
    "prev_number_digits_to_next_number_digits", "next_number_digits_to_prev_number_digits",
]
SITES = ["label", "final"]
# function-task name -> f1/f2 slot, per pair (Delta = act(f1) - act(f2))
PAIR_FUNCS = {
    "antonym_synonym": ("antonym", "synonym"),
    "next_number_digits_prev_number_digits": ("next_number_digits", "prev_number_digits"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Logit-readout task-switch steering, clean train/test split.")
    p.add_argument("--directions", nargs="+", default=LOGIT_DIRECTIONS, choices=LOGIT_DIRECTIONS)
    p.add_argument("--sites", nargs="+", default=SITES, choices=SITES)
    p.add_argument("--layers", type=int, nargs="+",
                   default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--n_test", type=int, default=100, help="Test prompt pairs (label & query held out).")
    p.add_argument("--n_train", type=int, default=100, help="Train prompt pairs used to derive Delta.")
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--output_root", type=Path, default=LABEL_GEOMETRY_DIR / "oneshot_switch_logit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def first_single_tok(words, tok):
    return next((w for w in words if is_single_space_token(tok, w)), None)


def build_pools(f1_task, f2_task, source_task, target_task, root, tok):
    """Return (label_pool, query_pool) where each query carries its single-token source & target gold,
    and each label carries the f1/f2 demo inputs that produce it."""
    rf1, rf2 = load_task_json(root, f1_task), load_task_json(root, f2_task)
    o2i_f1, o2i_f2 = build_output_to_inputs(rf1), build_output_to_inputs(rf2)
    i2o_src = build_input_to_outputs(load_task_json(root, source_task))
    i2o_tgt = build_input_to_outputs(load_task_json(root, target_task))
    o2i_src = build_output_to_inputs(load_task_json(root, source_task))

    # LABEL pool: shared single-token outputs; need an f1-input and f2-input that produce the label,
    # plus a source-input that produces it (for the source-task test demo).
    label_pool = []
    for w in sorted(set(o2i_f1) & set(o2i_f2)):
        if not is_single_space_token(tok, w):
            continue
        label_pool.append({"label": w, "in_f1": o2i_f1[w][0], "in_f2": o2i_f2[w][0],
                           "in_src": o2i_src[w][0]})
    # QUERY pool: shared inputs with single-token gold under BOTH source and target.
    query_pool = []
    for q in sorted(set(i2o_src) & set(i2o_tgt)):
        sg = first_single_tok(i2o_src[q], tok)
        tg = first_single_tok(i2o_tgt[q], tok)
        if sg is not None and tg is not None:
            query_pool.append({"q": q, "src_gold": sg, "tgt_gold": tg})
    return label_pool, query_pool


def read_label_final_acts(model, tok, hooknames, prompts, label_pos, n_layers, bs, device):
    """Residual-stream activations at the demo-label token and the final token, all layers.
    Returns (label_acts, final_acts), each [N, n_layers, hidden] float32 on CPU."""
    from baukit import TraceDict
    N = len(prompts)
    lab = torch.zeros(N, n_layers, model.config.n_embd)
    fin = torch.zeros(N, n_layers, model.config.n_embd)
    for s in range(0, N, bs):
        enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True).to(device)
        B, ilen = enc.input_ids.shape
        pad = ilen - enc.attention_mask.sum(1)
        lidx = (pad + torch.tensor(label_pos[s:s + bs], device=device)).long()
        rows = torch.arange(B, device=device)
        with TraceDict(model, layers=hooknames, retain_output=True) as td:
            model(**enc)
        for L in range(n_layers):
            o = td[hooknames[L]].output
            o = o[0] if isinstance(o, tuple) else o
            lab[s:s + B, L] = o[rows, lidx].float().cpu()
            fin[s:s + B, L] = o[:, ilen - 1].float().cpu()
    return lab, fin


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
    from baukit import TraceDict
    set_seed(args.seed)
    print("Loading GPT-J ...")
    model, tok, mc = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()
    device = model.device
    dtype = next(model.parameters()).dtype
    hooknames = mc["layer_hook_names"]
    n_layers = mc["n_layers"]
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    args.output_root.mkdir(parents=True, exist_ok=True)
    meta = {}

    for direction in args.directions:
        cfg = DIRECTIONS[direction]
        pair, sign, source, target = cfg["pair"], cfg["sign"], cfg["source"], cfg["target"]
        f1_task, f2_task = PAIR_FUNCS[pair]
        label_pool, query_pool = build_pools(f1_task, f2_task, source, target, args.root_data_dir, tok)

        rng = np.random.default_rng(args.seed)
        lp = label_pool[:]; qp = query_pool[:]
        rng.shuffle(lp); rng.shuffle(qp)
        test_labels, train_labels = lp[:args.n_test], lp[args.n_test:]
        test_queries, train_queries = qp[:args.n_test], qp[args.n_test:]
        assert len(test_labels) == args.n_test and len(test_queries) == args.n_test, "pool too small for n_test"

        # ---- TRAIN: n_train prompt PAIRS (label in train_out x query in train_in) -> Delta ----
        ridx = np.random.default_rng(args.seed + 1)
        tl = [train_labels[i] for i in ridx.integers(0, len(train_labels), args.n_train)]
        tq = [train_queries[i] for i in ridx.integers(0, len(train_queries), args.n_train)]
        f1_prompts, f2_prompts, f1_lab, f2_lab = [], [], [], []
        for lab, qy in zip(tl, tq):
            for slot, plist, lidxlist in (("in_f1", f1_prompts, f1_lab), ("in_f2", f2_prompts, f2_lab)):
                pd = build_prompt_data(lab[slot], lab["label"], qy["q"], lab["label"])
                tk, ps = get_token_meta_labels(pd, tok, query=qy["q"], prepend_bos=mc["prepend_bos"])
                li, _ = extract_positions(tk)
                plist.append(ps); lidxlist.append(li)
        a1l, a1f = read_label_final_acts(model, tok, hooknames, f1_prompts, f1_lab, n_layers, args.batch_size, device)
        a2l, a2f = read_label_final_acts(model, tok, hooknames, f2_prompts, f2_lab, n_layers, args.batch_size, device)
        delta_label = (sign * (a1l - a2l).mean(0)).to(device)   # [n_layers, hidden]
        delta_final = (sign * (a1f - a2f).mean(0)).to(device)

        meta[direction] = {
            "pair": pair, "source": source, "target": target, "sign": sign,
            "label_pool": len(label_pool), "query_pool": len(query_pool),
            "n_train_pairs": args.n_train, "n_test_pairs": args.n_test,
            "train_labels_distinct": len(train_labels), "train_queries_distinct": len(train_queries),
            "label_norms": {int(L): float(torch.linalg.norm(delta_label[L])) for L in args.layers},
            "final_norms": {int(L): float(torch.linalg.norm(delta_final[L])) for L in args.layers}}
        print(f"\n=== {direction} (source={source}->target={target}, sign={sign:+.0f}) | "
              f"label_pool={len(label_pool)} query_pool={len(query_pool)} | "
              f"train pairs={args.n_train} test pairs={args.n_test} ===")

        # ---- TEST: n_test source-task prompts (test label <-> test query, random 1-to-1) ----
        prompts, label_idx, tgt_ids, src_ids = [], [], [], []
        for lab, qy in zip(test_labels, test_queries):
            pd = build_prompt_data(lab["in_src"], lab["label"], qy["q"], lab["label"])
            tk, ps = get_token_meta_labels(pd, tok, query=qy["q"], prepend_bos=mc["prepend_bos"])
            li, _ = extract_positions(tk)
            prompts.append(ps); label_idx.append(li)
            tgt_ids.append(get_answer_id(ps, " " + qy["tgt_gold"], tok)[0])
            src_ids.append(get_answer_id(ps, " " + qy["src_gold"], tok)[0])

        def contrast(lg, base):
            return np.array([float(lg[r, tgt_ids[base + r]] - lg[r, src_ids[base + r]])
                             for r in range(lg.shape[0])])

        def run(site, L, alpha):
            """Per-query logit(target)-logit(source) at final position; alpha=0 => no hook (clean)."""
            vec = None if alpha == 0 else (alpha * (delta_label if site == "label" else delta_final)[L]).to(dtype)
            out = np.zeros(len(prompts))
            for s in range(0, len(prompts), args.batch_size):
                enc = tok(prompts[s:s + args.batch_size], return_tensors="pt", padding=True).to(device)
                B, ilen = enc.input_ids.shape
                if vec is None:
                    lg = model(**enc).logits[:, ilen - 1, :]
                else:
                    pad = ilen - enc.attention_mask.sum(1)
                    idx = (torch.full((B,), ilen - 1, device=device) if site == "final"
                           else pad + torch.tensor(label_idx[s:s + B], device=device)).long()
                    rows = torch.arange(B, device=device)

                    def hook(output, layer_name):
                        if isinstance(output, tuple) and int(layer_name.split(".")[2]) == L:
                            output[0][rows, idx] += vec
                        return output
                    with TraceDict(model, layers=hooknames, edit_output=hook):
                        lg = model(**enc).logits[:, ilen - 1, :]
                out[s:s + B] = contrast(lg, s)
            return out

        def stat(a):
            a = np.asarray(a)
            sem = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")
            return dict(mean_logit_diff=float(a.mean()), ci95=(1.96 * sem if sem == sem else None), n=len(a))

        baseline = stat(run("final", 0, 0))  # clean, site/layer-independent
        conditions = {}
        for site in args.sites:
            for L in args.layers:
                for alpha in args.alphas:
                    conditions[f"{site}|{L}|{alpha}"] = dict(site=site, layer=int(L), alpha=float(alpha),
                                                             **stat(run(site, int(L), float(alpha))))
            print(f"  {site}: done {len(args.layers)} layers x {len(args.alphas)} alphas")

        out = {"direction": direction, "metric": "logit(target_gold) - logit(source_gold)",
               "baseline_alpha0": baseline, "conditions": conditions}
        out_dir = args.output_root / direction
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "logit_diff.json").write_text(json.dumps(out, indent=2))
        print(f"  wrote {out_dir/'logit_diff.json'}  (baseline α0 = {baseline['mean_logit_diff']:.3f})")

    (args.output_root / "delta_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {args.output_root/'delta_meta.json'}")


if __name__ == "__main__":
    main()
