"""
Capture + grade paired 1-shot ICL activations, CORRECTED query sampling.

Supersedes capture_oneshot_paired.py for the accuracy-aware runs. Two differences:

  1. QUERY is drawn from the SHARED INPUT space (words that are a valid input under
     BOTH tasks, so a gold antonym AND gold synonym is defined) instead of the shared
     output space. The demo label `w` is still drawn from the shared OUTPUT space (so
     the demo label token is identical across f1/f2 -- the paired-difference design).
  2. Each prompt is GRADED against the dataset gold and the grade is stored in EVERY
     captured row's metadata, so activations can be filtered to top-1/2/3-correct.

Per shared output word w:
    demo_in_f1 ~ inputs that map to w under f1;  demo_in_f2 ~ inputs that map to w under f2
    q ~ shared-input pool \ {w, demo_in_f1, demo_in_f2}      (same q for f1 and f2)
    f1:  Q: <demo_in_f1>\nA: w\n\nQ: q\nA:      gold = f1[q]
    f2:  Q: <demo_in_f2>\nA: w\n\nQ: q\nA:      gold = f2[q]

Activations captured at two roles per prompt (same as capture_oneshot_paired.py):
    source = demo label token (last_label_token, icl_example_index == 1)
    target = final query token (last_prompt_token, icl_example_index is None)

Grading (one extra forward + a short greedy generate per prompt):
    gold_first_tok_rank : rank of gold answer's FIRST token in next-token logits (0 = top-1)
    top1/top2/top3      : gold_first_tok_rank < k   (the standard first-token top-k)
    exact_match_full    : greedy generation of len(gold) tokens == gold token ids
    model_top1_token, model_greedy_full : the model's actual output (for inspection)

Outputs artifacts/oneshot_paired_graded/<pair>/:
    shard_*.pt (+ index.json)  -- activations [rows, 28, 4096] fp32 + rich per-row metadata
    grading.json               -- per-prompt grades (function, w, q, gold, rank, ...)
    scores.json                -- top-1/2/3 summary per task
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt
from utils.paths import ARTIFACTS_ROOT

from extract_residual_stream_activations import (
    flush_shard,
    get_residual_stack,
    make_token_record,
    selected_token_records,
)

TASK_PAIRS = {
    "antonym_synonym": ("antonym", "synonym"),
    "next_number_prev_number": ("next_number", "prev_number"),
    "next_number_digits_prev_number_digits": ("next_number_digits", "prev_number_digits"),
}


def stable_seed(*parts):
    d = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(d[:8], "little") % (2 ** 32)


def stable_rng(*parts):
    return np.random.default_rng(stable_seed(*parts))


def parse_args():
    p = argparse.ArgumentParser(description="Capture+grade paired 1-shot activations with shared-input query.")
    p.add_argument("--task_pair", choices=sorted(TASK_PAIRS), default="antonym_synonym")
    p.add_argument("--max_words", type=int, default=None)
    p.add_argument("--store_dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--save_path_root", type=str, default=str(ARTIFACTS_ROOT / "oneshot_paired_graded"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--shard_size", type=int, default=100)
    p.add_argument("--allow_multitoken_label", action="store_true",
                   help="Allow multi-token demo-label words; capture the LAST label token (identical across "
                        "f1/f2). Needed for number words (mostly multi-token). Default keeps single-token labels.")
    p.add_argument("--gen_cap", type=int, default=8, help="max new tokens to greedily generate for full exact-match.")
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    return p.parse_args()


def load_task(root, t):
    recs = json.load(open(Path(root) / "abstractive" / f"{t}.json"))
    o2i, i2o = defaultdict(set), {}
    for r in recs:
        inp, out = str(r["input"]).strip(), str(r["output"]).strip()
        o2i[out].add(inp)
        i2o[inp] = out
    return {k: sorted(v) for k, v in o2i.items()}, i2o


def build_prompt_string(demo_input, demo_output, query_input, args):
    pd = word_pairs_to_prompt_data(
        {"input": [demo_input], "output": [demo_output]},
        query_target_pair={"input": query_input, "output": query_input},  # query output unused (ends at "A:")
        prepend_bos_token=False,
        prefixes=args.prefixes,
        separators=args.separators,
        prepend_space=True,
    )
    return pd, create_prompt(pd)


def extract_source_target_records(token_labels):
    records = selected_token_records(token_labels)
    source = target = None
    for rec in records:
        if rec["token_role"] == "last_label_token" and rec["icl_example_index"] == 1:
            source = rec
        elif rec["token_role"] == "last_prompt_token" and rec["icl_example_index"] is None:
            target = rec
    if source is None or target is None:
        raise ValueError("Could not derive both source and target records.")
    return {"source": source, "target": target}


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    store_dtype = torch.float16 if args.store_dtype == "float16" else torch.float32
    f1, f2 = TASK_PAIRS[args.task_pair]
    print(f"task_pair={args.task_pair} -> f1={f1}, f2={f2}")

    o2i_f1, i2o_f1 = load_task(args.root_data_dir, f1)
    o2i_f2, i2o_f2 = load_task(args.root_data_dir, f2)

    print("Loading model...")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device, revision=args.revision)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def single(w):
        return len(tokenizer(" " + w).input_ids) == 1

    # demo label pool = shared OUTPUT; query pool = shared INPUT (gold under both).
    shared_out = sorted(set(o2i_f1) & set(o2i_f2))
    label_words = list(shared_out) if args.allow_multitoken_label else [w for w in shared_out if single(w)]
    shared_in = sorted(set(i2o_f1) & set(i2o_f2))
    query_pool = list(shared_in)
    print(f"label words (shared output, single-tok): {len(label_words)}")
    print(f"query pool (shared input, gold under both): {len(query_pool)}")
    if args.max_words is not None:
        label_words = label_words[: args.max_words]

    def gold_first_tok(answer):
        return tokenizer(" " + answer).input_ids[0]

    output_dir = Path(args.save_path_root) / args.task_pair
    if output_dir.exists():
        for old in output_dir.glob("shard_*.pt"):
            old.unlink()
        for fn in ("index.json", "grading.json", "scores.json"):
            if (output_dir / fn).exists():
                (output_dir / fn).unlink()

    config = {
        "task_pair": args.task_pair, "function_tasks": {"f1": f1, "f2": f2},
        "model_name": args.model_name, "model_config": model_config, "seed": args.seed,
        "store_dtype": str(store_dtype), "prefixes": args.prefixes, "separators": args.separators,
        "n_shots": 1, "roles": ["source", "target"],
        "query_space": "shared_input", "label_space": "shared_output",
        "allow_multitoken_label": args.allow_multitoken_label,
        "n_label_words": len(label_words), "n_query_pool": len(query_pool),
        "grading": "first-token rank (top-k = rank<k) + full greedy exact match; gold = dataset output for query",
    }

    shard_acts, shard_meta, shard_paths = [], [], []
    shard_index = words_in_shard = n_done = 0
    grading_rows = []
    funcs = (("f1", f1, o2i_f1, i2o_f1), ("f2", f2, o2i_f2, i2o_f2))

    for w in label_words:
        rng = stable_rng(args.seed, args.task_pair, w)
        d1 = str(rng.choice(o2i_f1[w]))
        d2 = str(rng.choice(o2i_f2[w]))
        forbidden = {w, d1, d2}
        cand = [q for q in query_pool if q not in forbidden]
        if not cand:
            continue
        q = str(rng.choice(cand))
        # source role = last_label_token -> compare against the LAST token of " "+w (== first token if single-token).
        expected_src_id = tokenizer(" " + w).input_ids[-1]
        demo_in = {"f1": d1, "f2": d2}

        for tag, task, _o2i, i2o in funcs:
            gold = i2o[q]
            gold_ids = tokenizer(" " + gold).input_ids
            pd, prompt = build_prompt_string(demo_in[tag], w, q, args)
            residual_stack, token_labels, prompt_string = get_residual_stack(
                pd, model, model_config, tokenizer, include_embeddings=False
            )
            roles = extract_source_target_records(token_labels)

            # verify source token id == ' '+w first token (paired-design invariant)
            full_ids = tokenizer(prompt_string).input_ids
            assert full_ids[roles["source"]["token_position"]] == expected_src_id, f"src mismatch w={w!r} {tag}"

            # --- grade: one forward for logits; first-token rank top-k ---
            inp = tokenizer(prompt_string, return_tensors="pt").to(model.device)
            logits = model(**inp).logits[0, -1, :]
            gtok = gold_ids[0]
            rank = int((logits > logits[gtok]).sum().item())
            model_top1 = tokenizer.decode([int(logits.argmax())])

            grade = {
                "task_pair": args.task_pair, "function": tag, "function_task": task,
                "output_word": w, "query": q, "demo_input": demo_in[tag],
                "gold": gold, "gold_n_tokens": len(gold_ids),
                "gold_first_tok_rank": rank,
                "top1": rank < 1, "top2": rank < 2, "top3": rank < 3,
                "model_top1_token": model_top1,
            }
            grading_rows.append(grade)

            for role in ("source", "target"):
                rec = roles[role]
                pos = rec["token_position"]
                act = residual_stack[:, pos, :].cpu().to(store_dtype)
                shard_acts.append(act)
                md = dict(make_token_record(rec["token_role"], rec["icl_example_index"],
                                            (rec["token_position"], rec["token_text"], rec["token_label"])))
                md.update({"task_pair": args.task_pair, "function": tag, "function_task": task,
                           "output_word": w, "query_word": q, "demo_input": demo_in[tag], "role": role,
                           "gold": gold, "gold_n_tokens": len(gold_ids),
                           "gold_first_tok_rank": rank,
                           "top1": rank < 1, "top2": rank < 2, "top3": rank < 3})
                shard_meta.append(md)

        n_done += 1
        words_in_shard += 1
        if words_in_shard >= args.shard_size:
            sp = flush_shard(shard_acts, shard_meta, output_dir, shard_index, config)
            if sp is not None:
                shard_paths.append(str(sp))
            shard_acts, shard_meta = [], []
            shard_index += 1
            words_in_shard = 0
        if n_done % 50 == 0:
            print(f"  {n_done}/{len(label_words)} words done")

    sp = flush_shard(shard_acts, shard_meta, output_dir, shard_index, config)
    if sp is not None:
        shard_paths.append(str(sp))

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "index.json", "w") as f:
        json.dump({"config": config, "shards": shard_paths}, f, indent=2)
    with open(output_dir / "grading.json", "w") as f:
        json.dump(grading_rows, f, indent=2)

    summary = {"task_pair": args.task_pair, "seed": args.seed, "n_shots": 1,
               "query_space": "shared_input", "tasks": {}}
    for task in (f1, f2):
        rs = [g for g in grading_rows if g["function_task"] == task]
        n = len(rs)
        summ = {"n_prompts": n,
                "top1": sum(g["top1"] for g in rs) / n, "top2": sum(g["top2"] for g in rs) / n,
                "top3": sum(g["top3"] for g in rs) / n}
        summary["tasks"][task] = summ
        print(f"{task:>10}: n={n}  top1={summ['top1']:.3f}  top2={summ['top2']:.3f}  "
              f"top3={summ['top3']:.3f}")
    with open(output_dir / "scores.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"processed {n_done} label words -> {len(shard_paths)} shards at {output_dir}")


if __name__ == "__main__":
    main()
