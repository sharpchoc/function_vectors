"""
Capture PAIRED 1-shot ICL residual activations for the `paired_tasks` datasets.

For a shared output word `w` producible by BOTH functions f1 and f2, build two
prompts that are IDENTICAL except for the ICL demo's INPUT word:

    f1:   Q: <demo_in_f1>\nA: <w>\n\nQ: <query>\nA:
    f2:   Q: <demo_in_f2>\nA: <w>\n\nQ: <query>\nA:

The demo label (` w`) and the query are the same in both; only the demo input
differs, so the activation difference at the label token isolates the FUNCTION /
context, not token identity. (Same idea as capture_oneshot_paired.py, generalized
to the paired_tasks pairs; written as a sibling so it doesn't disturb that script.)

Unlike the original, single-token outputs are NOT required: the paired property holds
for multi-token labels too (the label word is identical across f1/f2), and this lets
the number pair reach 100 (only 26 number-words are single-token, but 198 shared). We
capture the full standard location set per prompt (pre/first/last label token + final
query token), all layers; for multi-token labels the function-relevant token is
last_label_token.

Pairs (--pair): antonym_synonym, synonym_rhyme, antonym_rhyme, next_number_prev_number.

Output: results/oneshot_paired_tasks/<pair>/{shard_*.pt, index.json}; each shared word
contributes 2 functions x 6 roles = 12 rows of [n_layers, 4096] (fp32).

Run:  HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 \
      python src/eval_scripts/capture_oneshot_paired_tasks.py --pair antonym_synonym
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import word_pairs_to_prompt_data

from extract_residual_stream_activations import (
    flush_shard,
    get_residual_stack,
    make_token_record,
    selected_token_records,
)

# pair name -> (f1 task, f2 task)
TASK_PAIRS = {
    "antonym_synonym": ("antonym", "synonym"),
    "synonym_rhyme": ("synonym", "rhyme"),
    "antonym_rhyme": ("antonym", "rhyme"),
    "next_number_prev_number": ("next_number", "prev_number"),
}


def stable_seed(*parts):
    digest = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def parse_args():
    p = argparse.ArgumentParser(description="Capture paired 1-shot activations (two prompts differ only in the ICL demo input).")
    p.add_argument("--pair", choices=sorted(TASK_PAIRS), default="antonym_synonym")
    p.add_argument("--data_dir", default="dataset_files/paired_tasks")
    p.add_argument("--n_target", type=int, default=100, help="number of shared output words (paired prompts) to capture")
    p.add_argument("--save_path_root", default="results/oneshot_paired_tasks")
    p.add_argument("--model_name", default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--revision", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard_size", type=int, default=100, help="shared words per shard")
    p.add_argument("--store_dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--prefixes", type=json.loads, default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads, default={"input": "\n", "output": "\n\n", "instructions": ""})
    return p.parse_args()


def load_task(data_dir, task):
    with open(os.path.join(data_dir, f"{task}.json")) as f:
        return json.load(f)


def out_to_in(records):
    m = {}
    for r in records:
        m.setdefault(str(r["output"]).strip(), set()).add(str(r["input"]).strip())
    return {o: sorted(ins) for o, ins in m.items()}


def build_prompt_data(demo_input, demo_output, query_input, query_output, args):
    return word_pairs_to_prompt_data(
        {"input": [demo_input], "output": [demo_output]},
        query_target_pair={"input": query_input, "output": query_output},
        prepend_bos_token=False, prefixes=args.prefixes, separators=args.separators,
        prepend_space=True,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    store_dtype = torch.float16 if args.store_dtype == "float16" else torch.float32

    task_f1, task_f2 = TASK_PAIRS[args.pair]
    print(f"pair={args.pair} -> f1={task_f1}, f2={task_f2}")
    o2i_f1 = out_to_in(load_task(args.data_dir, task_f1))
    o2i_f2 = out_to_in(load_task(args.data_dir, task_f2))
    shared = sorted(set(o2i_f1) & set(o2i_f2))
    print(f"shared output words producible under both: {len(shared)}")

    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    model.eval()

    # Query pool: shared output words (in-domain, identical across f1/f2 by construction).
    query_pool = list(shared)

    output_root = Path(args.save_path_root)
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_dir = output_root / args.pair
    if output_dir.exists():
        for old in output_dir.glob("shard_*.pt"):
            old.unlink()
        if (output_dir / "index.json").exists():
            (output_dir / "index.json").unlink()

    config = {
        "pair": args.pair,
        "function_tasks": {"f1": task_f1, "f2": task_f2},
        "data_dir": args.data_dir,
        "model_name": args.model_name,
        "model_config": model_config,
        "seed": args.seed,
        "n_shots": 1,
        "store_dtype": str(store_dtype),
        "prefixes": args.prefixes,
        "separators": args.separators,
        "token_roles": ["pre_label_token", "first_label_token", "last_label_token",
                        "label_token", "last_prompt_token", "final_token"],
        "design": "two prompts per shared output word differ ONLY in the ICL demo input",
    }

    shard_acts, shard_meta, shard_paths = [], [], []
    shard_index, in_shard, n_done = 0, 0, 0

    for w in shared:
        if n_done >= args.n_target:
            break
        rng = np.random.default_rng(stable_seed(args.seed, args.pair, w))
        demo_in_f1 = str(rng.choice(o2i_f1[w]))
        demo_in_f2 = str(rng.choice(o2i_f2[w]))
        forbidden = {w, demo_in_f1, demo_in_f2}
        candidates = [q for q in query_pool if q not in forbidden]
        if not candidates:
            print(f"  skip w={w!r}: no valid query")
            continue
        q = str(rng.choice(candidates))

        expected_label = " " + w  # prepend_space convention
        for function, demo_input, task_name in (("f1", demo_in_f1, task_f1), ("f2", demo_in_f2, task_f2)):
            prompt_data = build_prompt_data(demo_input, w, q, w, args)
            residual_stack, token_labels, prompt_string = get_residual_stack(
                prompt_data, model, model_config, tokenizer, include_embeddings=False
            )
            recs = {r["token_role"]: r for r in selected_token_records(token_labels)}

            # Verify the label span decodes to ' '+w (handles single- AND multi-token labels).
            ids = tokenizer(prompt_string).input_ids
            f_pos = recs["first_label_token"]["token_position"]
            l_pos = recs["last_label_token"]["token_position"]
            decoded = tokenizer.decode(ids[f_pos:l_pos + 1])
            assert decoded == expected_label, f"label mismatch w={w!r} ({function}): {decoded!r} != {expected_label!r}"

            for role in ("pre_label_token", "first_label_token", "last_label_token",
                         "label_token", "last_prompt_token", "final_token"):
                rec = recs[role]
                pos = rec["token_position"]
                if pos >= residual_stack.shape[1]:
                    raise IndexError(f"pos {pos} >= seq len {residual_stack.shape[1]}")
                shard_acts.append(residual_stack[:, pos, :].cpu().to(store_dtype))
                meta = dict(make_token_record(rec["token_role"], rec["icl_example_index"],
                                              (rec["token_position"], rec["token_text"], rec["token_label"])))
                meta.update({
                    "pair": args.pair, "function": function, "function_task": task_name,
                    "output_word": w, "query_word": q, "demo_input": demo_input, "role": role,
                })
                shard_meta.append(meta)

        n_done += 1
        in_shard += 1
        if in_shard >= args.shard_size:
            sp = flush_shard(shard_acts, shard_meta, output_dir, shard_index, config)
            if sp is not None:
                shard_paths.append(str(sp))
            shard_acts, shard_meta = [], []
            shard_index += 1
            in_shard = 0

    sp = flush_shard(shard_acts, shard_meta, output_dir, shard_index, config)
    if sp is not None:
        shard_paths.append(str(sp))

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "index.json", "w") as f:
        json.dump({"config": config, "shards": shard_paths, "n_paired_words": n_done}, f, indent=2)
    print(f"{args.pair}: captured {n_done} shared words ({2 * n_done} prompts) -> {len(shard_paths)} shard(s) at {output_dir}")
    if n_done < args.n_target:
        print(f"  NOTE: only {n_done} shared words available (<{args.n_target}).")


if __name__ == "__main__":
    main()
