"""
Capture + grade paired TWO-shot ICL activations (matched-label, distinct labels).

Two-demo sibling of capture_and_grade_oneshot_paired.py. Each prompt has TWO demos whose
labels (L1, L2) are matched position-by-position across the two functions of the pair and are
DISTINCT within a prompt; the demo INPUTS differ by function (paired-difference design). The
query q is drawn from the shared INPUT space (gold defined under both functions) and is shared
across f1/f2.

Per matched-label tuple (L1, L2) drawn from the shared OUTPUT pool, with shared query q:
    f1:  Q: <f1-input->L1>\nA: L1\n\nQ: <f1-input->L2>\nA: L2\n\nQ: q\nA:    gold = f1[q]
    f2:  Q: <f2-input->L1>\nA: L1\n\nQ: <f2-input->L2>\nA: L2\n\nQ: q\nA:    gold = f2[q]
For digits, label L => next-input (L-1) and prev-input (L+1).

Enumeration: one tuple per label word w -> L1=w, L2=random distinct label, q from the shared-input
pool minus {L1,L2,all four demo inputs}. Deterministic per (seed, task_pair, w).

Activations captured at FIVE roles per prompt (from selected_token_records):
    demo1_prelabel = pre_label_token  @ icl_example_index 1   (the "A:" before demo-1's label)
    demo1_label    = last_label_token @ icl_example_index 1
    demo2_prelabel = pre_label_token  @ icl_example_index 2   (the "A:" before demo-2's label)
    demo2_label    = last_label_token @ icl_example_index 2
    query_final    = last_prompt_token @ icl_example_index None

Grading (one forward per prompt): first-token rank of the query gold under each function
(gold_first_tok_rank, top1/2/3, model_top1_token), stamped into every captured row + grading.json.

Outputs ARTIFACTS_ROOT/twoshot_paired_graded/<pair>/:
    shard_*.pt (+ index.json)  -- activations [rows, n_layers, hidden] + rich per-row metadata
    grading.json               -- per-prompt grades
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
    "next_number_digits_prev_number_digits": ("next_number_digits", "prev_number_digits"),
}

# (token_role, icl_example_index) -> output role name
ROLE_SPEC = [
    ("pre_label_token", 1, "demo1_prelabel"),
    ("last_label_token", 1, "demo1_label"),
    ("pre_label_token", 2, "demo2_prelabel"),
    ("last_label_token", 2, "demo2_label"),
    ("last_prompt_token", None, "query_final"),
]


def stable_seed(*parts):
    d = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(d[:8], "little") % (2 ** 32)


def stable_rng(*parts):
    return np.random.default_rng(stable_seed(*parts))


def parse_args():
    p = argparse.ArgumentParser(description="Capture+grade paired TWO-shot activations (matched, distinct labels).")
    p.add_argument("--task_pair", choices=sorted(TASK_PAIRS), default="antonym_synonym")
    p.add_argument("--max_words", type=int, default=None)
    p.add_argument("--store_dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--save_path_root", type=str, default=str(ARTIFACTS_ROOT / "twoshot_paired_graded"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--shard_size", type=int, default=100)
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


def build_prompt_string(demo_inputs, demo_outputs, query_input, args):
    pd = word_pairs_to_prompt_data(
        {"input": list(demo_inputs), "output": list(demo_outputs)},
        query_target_pair={"input": query_input, "output": query_input},  # query output unused (ends at "A:")
        prepend_bos_token=False,
        prefixes=args.prefixes,
        separators=args.separators,
        prepend_space=True,
    )
    return pd, create_prompt(pd)


def extract_role_records(token_labels):
    records = selected_token_records(token_labels)
    found = {}
    for rec in records:
        for role_name_key, idx, role in ROLE_SPEC:
            if rec["token_role"] == role_name_key and rec["icl_example_index"] == idx:
                found[role] = rec
    missing = [r for _, _, r in ROLE_SPEC if r not in found]
    if missing:
        raise ValueError(f"Could not derive roles: {missing}")
    return found


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

    # demo label pool = shared OUTPUT (single-token); query pool = shared INPUT (gold under both).
    shared_out = sorted(set(o2i_f1) & set(o2i_f2))
    label_words = [w for w in shared_out if single(w)]
    shared_in = sorted(set(i2o_f1) & set(i2o_f2))
    query_pool = list(shared_in)
    print(f"label words (shared output, single-tok): {len(label_words)}")
    print(f"query pool (shared input, gold under both): {len(query_pool)}")
    if len(label_words) < 2:
        raise SystemExit("Need >=2 label words to form a distinct (L1,L2) tuple.")
    if args.max_words is not None:
        label_words = label_words[: args.max_words]

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
        "n_shots": 2, "roles": [r for _, _, r in ROLE_SPEC],
        "label_design": "two distinct matched labels (L1,L2) from shared_output; demos differ by function",
        "query_space": "shared_input", "label_space": "shared_output",
        "n_label_words": len(label_words), "n_query_pool": len(query_pool),
        "grading": "first-token rank (top-k = rank<k); gold = dataset output for query",
    }

    shard_acts, shard_meta, shard_paths = [], [], []
    shard_index = words_in_shard = n_done = 0
    grading_rows = []
    label_set = list(label_words)
    funcs = (("f1", f1, o2i_f1, i2o_f1), ("f2", f2, o2i_f2, i2o_f2))

    for w in label_words:
        rng = stable_rng(args.seed, args.task_pair, w)
        L1 = w
        # distinct second label
        cand_L2 = [x for x in label_set if x != L1]
        if not cand_L2:
            continue
        L2 = str(rng.choice(cand_L2))
        labels = [L1, L2]

        # demo inputs per function (one per label)
        demo_inputs = {
            "f1": [str(rng.choice(o2i_f1[L1])), str(rng.choice(o2i_f1[L2]))],
            "f2": [str(rng.choice(o2i_f2[L1])), str(rng.choice(o2i_f2[L2]))],
        }
        forbidden = {L1, L2, *demo_inputs["f1"], *demo_inputs["f2"]}
        cand_q = [q for q in query_pool if q not in forbidden]
        if not cand_q:
            continue
        q = str(rng.choice(cand_q))

        expected_label_ids = [tokenizer(" " + lab).input_ids[-1] for lab in labels]

        for tag, task, _o2i, i2o in funcs:
            gold = i2o[q]
            gold_ids = tokenizer(" " + gold).input_ids
            pd, prompt = build_prompt_string(demo_inputs[tag], labels, q, args)
            residual_stack, token_labels, prompt_string = get_residual_stack(
                pd, model, model_config, tokenizer, include_embeddings=False
            )
            roles = extract_role_records(token_labels)

            # paired-design invariant: demo label tokens == ' '+L1 / ' '+L2 last token (identical across f1/f2)
            full_ids = tokenizer(prompt_string).input_ids
            assert full_ids[roles["demo1_label"]["token_position"]] == expected_label_ids[0], f"L1 mismatch w={w!r} {tag}"
            assert full_ids[roles["demo2_label"]["token_position"]] == expected_label_ids[1], f"L2 mismatch w={w!r} {tag}"

            # --- grade: one forward for logits; first-token rank top-k ---
            inp = tokenizer(prompt_string, return_tensors="pt").to(model.device)
            logits = model(**inp).logits[0, -1, :]
            gtok = gold_ids[0]
            rank = int((logits > logits[gtok]).sum().item())
            model_top1 = tokenizer.decode([int(logits.argmax())])

            grade = {
                "task_pair": args.task_pair, "function": tag, "function_task": task,
                "label1": L1, "label2": L2, "query": q, "demo_inputs": demo_inputs[tag],
                "gold": gold, "gold_n_tokens": len(gold_ids),
                "gold_first_tok_rank": rank,
                "top1": rank < 1, "top2": rank < 2, "top3": rank < 3,
                "model_top1_token": model_top1,
            }
            grading_rows.append(grade)

            for _, _, role in ROLE_SPEC:
                rec = roles[role]
                pos = rec["token_position"]
                act = residual_stack[:, pos, :].cpu().to(store_dtype)
                shard_acts.append(act)
                md = dict(make_token_record(rec["token_role"], rec["icl_example_index"],
                                            (rec["token_position"], rec["token_text"], rec["token_label"])))
                md.update({"task_pair": args.task_pair, "function": tag, "function_task": task,
                           "role": role, "label1": L1, "label2": L2, "query_word": q,
                           "demo_inputs": demo_inputs[tag],
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
            print(f"  {n_done}/{len(label_words)} tuples done")

    sp = flush_shard(shard_acts, shard_meta, output_dir, shard_index, config)
    if sp is not None:
        shard_paths.append(str(sp))

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "index.json", "w") as f:
        json.dump({"config": config, "shards": shard_paths}, f, indent=2)
    with open(output_dir / "grading.json", "w") as f:
        json.dump(grading_rows, f, indent=2)

    summary = {"task_pair": args.task_pair, "seed": args.seed, "n_shots": 2,
               "query_space": "shared_input", "tasks": {}}
    for task in (f1, f2):
        rs = [g for g in grading_rows if g["function_task"] == task]
        n = len(rs)
        summ = {"n_prompts": n,
                "top1": sum(g["top1"] for g in rs) / n, "top2": sum(g["top2"] for g in rs) / n,
                "top3": sum(g["top3"] for g in rs) / n}
        summary["tasks"][task] = summ
        print(f"{task:>20}: n={n}  top1={summ['top1']:.3f}  top2={summ['top2']:.3f}  "
              f"top3={summ['top3']:.3f}")
    with open(output_dir / "scores.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"processed {n_done} label tuples -> {len(shard_paths)} shards at {output_dir}")


if __name__ == "__main__":
    main()
