"""
Score GPT-J on the EXACT 1-shot paired prompts used by capture_oneshot_paired.py.

For each shared single-token output word w, the capture built two prompts that differ
only in the demo input (same demo label w, same query word q):
    antonym:  Q: <demo_in_f1>\nA: w\n\nQ: q\nA:
    synonym:  Q: <demo_in_f2>\nA: w\n\nQ: q\nA:
The query q was drawn (not for correctness) from the shared output-word pool. Here we
RE-DERIVE the identical prompts (same stable RNG), run the model, and score the
first-token prediction of the query answer against the dataset gold for q
(top-1/top-2 by rank, matching eval_utils.compute_top_k_accuracy: rank < k).

Only prompts whose query q has a gold entry as an INPUT in that task's dataset are
scored (antonym 464, synonym 389; 853 total for antonym_synonym).

Outputs <LABEL_GEOMETRY_DIR>/oneshot_paired_scored/<pair>/scores.json.
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
from utils.paths import LABEL_GEOMETRY_DIR

TASK_PAIRS = {
    "antonym_synonym": ("antonym", "synonym"),
    "landmark_park": ("landmark-country", "park-country"),
}


def stable_seed(*parts):
    d = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(d[:8], "little") % (2 ** 32)


def stable_rng(*parts):
    return np.random.default_rng(stable_seed(*parts))


def load_task(root, t):
    recs = json.load(open(Path(root) / "abstractive" / f"{t}.json"))
    o2i, i2o = defaultdict(set), {}
    for r in recs:
        inp, out = str(r["input"]).strip(), str(r["output"]).strip()
        o2i[out].add(inp)
        i2o[inp] = out
    return {k: sorted(v) for k, v in o2i.items()}, i2o


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_pair", choices=sorted(TASK_PAIRS), default="antonym_synonym")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_root", type=str, default=str(LABEL_GEOMETRY_DIR / "oneshot_paired_scored"))
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    return p.parse_args()


def build_prompt_string(demo_input, demo_output, query_input, args):
    word_pairs = {"input": [demo_input], "output": [demo_output]}
    pd = word_pairs_to_prompt_data(
        word_pairs,
        query_target_pair={"input": query_input, "output": query_input},  # output unused (prompt ends at "A:")
        prepend_bos_token=False,
        prefixes=args.prefixes,
        separators=args.separators,
        prepend_space=True,
    )
    return create_prompt(pd)


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    f1, f2 = TASK_PAIRS[args.task_pair]

    o2i_f1, i2o_f1 = load_task(args.root_data_dir, f1)
    o2i_f2, i2o_f2 = load_task(args.root_data_dir, f2)

    print("Loading model...")
    model, tokenizer, _cfg = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()

    def single(w):
        return len(tokenizer(" " + w).input_ids) == 1

    shared = sorted(set(o2i_f1) & set(o2i_f2))
    shared_words = [w for w in shared if single(w)]
    query_pool = list(shared_words)
    print(f"shared_words={len(shared_words)}")

    # gold first-token id, space-prefixed (matches prompt label convention).
    def gold_first_token(answer):
        return tokenizer(" " + answer).input_ids[0]

    per_func = {f1: {"records": []}, f2: {"records": []}}
    funcs = (("f1", f1, o2i_f1, i2o_f1), ("f2", f2, o2i_f2, i2o_f2))

    for w in shared_words:
        rng = stable_rng(args.seed, args.task_pair, w)
        d1 = str(rng.choice(o2i_f1[w]))
        d2 = str(rng.choice(o2i_f2[w]))
        demo_in = {"f1": d1, "f2": d2}
        forbidden = {w, d1, d2}
        cand = [q for q in query_pool if q not in forbidden]
        if not cand:
            continue
        q = str(rng.choice(cand))

        for tag, task, _o2i, i2o in funcs:
            if q not in i2o:
                continue  # no gold answer for this query under this task -> skip
            gold = i2o[q]
            prompt = build_prompt_string(demo_in[tag], w, q, args)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
            logits = model(ids).logits[0, -1, :]  # next-token distribution
            gtok = gold_first_token(gold)
            # rank of gold first-token (0 = top-1). rank < k convention.
            rank = int((logits > logits[gtok]).sum().item())
            top1 = tokenizer.decode([int(logits.argmax())])
            per_func[task]["records"].append({
                "output_word": w, "query": q, "demo_input": demo_in[tag],
                "gold": gold, "gold_first_tok_rank": rank,
                "top1_pred": top1, "prompt": prompt,
            })

    out_dir = Path(args.output_root) / args.task_pair
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"task_pair": args.task_pair, "seed": args.seed, "n_shots": 1, "tasks": {}}
    for task in (f1, f2):
        recs = per_func[task]["records"]
        n = len(recs)
        t1 = sum(r["gold_first_tok_rank"] < 1 for r in recs) / n if n else 0.0
        t2 = sum(r["gold_first_tok_rank"] < 2 for r in recs) / n if n else 0.0
        t3 = sum(r["gold_first_tok_rank"] < 3 for r in recs) / n if n else 0.0
        summary["tasks"][task] = {"n_prompts": n, "top1": t1, "top2": t2, "top3": t3}
        print(f"{task:>10}: n={n:4d}  top1={t1:.3f}  top2={t2:.3f}  top3={t3:.3f}")
    with open(out_dir / "scores.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_prompt.json", "w") as f:
        json.dump({task: per_func[task]["records"] for task in (f1, f2)}, f, indent=2)
    print(f"wrote {out_dir}/scores.json")


if __name__ == "__main__":
    main()
