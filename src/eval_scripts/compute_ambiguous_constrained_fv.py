"""Constrained FV extraction for the ambiguous task pairs.

Per the agreed recipe, every 10-shot extraction prompt is built with:
  - 5 OVERLAP demos + 5 DIFFERENTIATOR demos (order shuffled), and
  - the query is a DIFFERENTIATOR,
and only queries the model answers CORRECTLY (under such a prompt) are kept.
This makes the FV encode the function that DISTINGUISHES the pair, not the shared
(copy-able) overlap behaviour.

Mean head activations + CIE are computed with this sampler (additive `prompt_sampler`
hook on get_mean_head_activations / compute_indirect_effect); the task-specific FV is
the top-N heads by CIE. Outputs mirror artifacts/gptj_fv/<task>/ so the downstream
train-pooled builder and steering eval can consume them via --fv_root.

Run: HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 python <this>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
from utils.extract_utils import get_mean_head_activations, compute_function_vector
from utils.paths import ARTIFACTS_ROOT
from compute_indirect_effect import compute_indirect_effect

PARTNER = {"magnitude": "identity", "identity": "magnitude",
           "count_vowels": "count_consonants", "count_consonants": "count_vowels"}
PREF = {"input": "Q:", "output": "A:", "instructions": ""}
SEP = {"input": "\n", "output": "\n\n", "instructions": ""}


def differ_overlap_indices(split_ds, partner_map):
    """Indices in split where output differs from / equals the partner's output for that input."""
    ins, outs = split_ds["input"], split_ds["output"]
    differ, overlap = [], []
    for i, (x, y) in enumerate(zip(ins, outs)):
        (differ if partner_map[str(x)] != str(y) else overlap).append(i)
    return differ, overlap


def first_tok(tok, s):
    return tok(" " + str(s), add_special_tokens=False).input_ids[0]


def build_sampler(dataset, rng, train_overlap, train_differ, query_pool, n_each=5):
    def sampler():
        demo_idx = np.concatenate([rng.choice(train_overlap, n_each, replace=False),
                                   rng.choice(train_differ, n_each, replace=False)])
        rng.shuffle(demo_idx)  # randomize demo positions
        q = int(rng.choice(query_pool))
        return dataset["train"][demo_idx], dataset["valid"][np.array([q])]
    return sampler


def correct_valid_differ(dataset, tok, model, model_config, rng, train_overlap, train_differ,
                         valid_differ, n_each, device):
    """Keep valid-differentiator queries the model answers correctly under a 5+5 constrained prompt."""
    prompts, golds = [], []
    for q in valid_differ:
        demo_idx = np.concatenate([rng.choice(train_overlap, n_each, replace=False),
                                   rng.choice(train_differ, n_each, replace=False)])
        rng.shuffle(demo_idx)
        wp = dataset["train"][demo_idx]
        qt = dataset["valid"][np.array([q])]
        pd = word_pairs_to_prompt_data(wp, query_target_pair=qt, prepend_bos_token=False,
                                       shuffle_labels=False, prefixes=PREF, separators=SEP)
        prompts.append(create_prompt(pd))
        golds.append(str(qt["output"][0]))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    correct = []
    bs = 8
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**enc).logits[:, -1, :].argmax(-1).tolist()
        for j, p in enumerate(out):
            if p == first_tok(tok, golds[i + j]):
                correct.append(valid_differ[i + j])
    torch.cuda.empty_cache()
    return correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["magnitude", "identity", "count_vowels", "count_consonants"])
    ap.add_argument("--model_name", default="EleutherAI/gpt-j-6b")
    ap.add_argument("--root_data_dir", default="dataset_files")
    ap.add_argument("--output_root", default=str(ARTIFACTS_ROOT / "gptj_fv_ambiguous_constrained"))
    ap.add_argument("--n_each", type=int, default=5, help="demos per region (overlap / differ)")
    ap.add_argument("--n_top_heads", type=int, default=10)
    ap.add_argument("--n_trials", type=int, default=100, help="mean-activation trials")
    ap.add_argument("--n_ie_trials", type=int, default=25)
    ap.add_argument("--test_split", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    n_shots = 2 * args.n_each

    set_seed(args.seed)
    torch.set_grad_enabled(False)  # critical: otherwise extraction forwards retain the autograd graph -> OOM
    model, tok, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    summary = {}

    for task in args.tasks:
        print(f"\n=== {task} (partner {PARTNER[task]}) ===")
        dataset = load_dataset(task, root_data_dir=args.root_data_dir,
                               test_size=args.test_split, seed=args.seed)
        partner_map = {str(x["input"]): str(x["output"])
                       for x in json.load(open(f"{args.root_data_dir}/abstractive/{PARTNER[task]}.json"))}
        tr_diff, tr_over = differ_overlap_indices(dataset["train"], partner_map)
        va_diff, _ = differ_overlap_indices(dataset["valid"], partner_map)
        rng = np.random.RandomState(args.seed)
        qpool = correct_valid_differ(dataset, tok, model, model_config, rng,
                                     tr_over, tr_diff, va_diff, args.n_each, args.device)
        print(f"  train: overlap={len(tr_over)} differ={len(tr_diff)} | "
              f"valid differ={len(va_diff)} -> correct query pool={len(qpool)}")
        if len(tr_over) < args.n_each or len(tr_diff) < args.n_each or len(qpool) == 0:
            print(f"  SKIP {task}: insufficient pool (need >={args.n_each} each demo region, >0 query).")
            summary[task] = {"status": "insufficient_pool", "n_query_pool": len(qpool)}
            continue

        sampler = build_sampler(dataset, rng, tr_over, tr_diff, qpool, n_each=args.n_each)
        set_seed(args.seed)
        mean_act = get_mean_head_activations(dataset, model, model_config, tok,
                                             n_icl_examples=n_shots, N_TRIALS=args.n_trials,
                                             prefixes=PREF, separators=SEP, filter_set=np.array(qpool),
                                             batch_size=args.batch_size, prompt_sampler=sampler)
        set_seed(args.seed)
        ie = compute_indirect_effect(dataset, mean_act, model, model_config, tok,
                                     n_shots=n_shots, n_trials=args.n_ie_trials, last_token_only=True,
                                     prefixes=PREF, separators=SEP, filter_set=np.array(qpool),
                                     batch_size=args.batch_size, prompt_sampler=sampler)
        fv, top_heads = compute_function_vector(mean_act, ie, model, model_config,
                                                n_top_heads=args.n_top_heads)
        fv = fv.detach().float().cpu().reshape(-1)
        th = [[int(l), int(h), float(v)] for (l, h, v) in top_heads]

        d = Path(args.output_root) / task
        d.mkdir(parents=True, exist_ok=True)
        torch.save(mean_act.detach().cpu(), d / f"{task}_mean_head_activations.pt")
        torch.save(ie.detach().cpu(), d / f"{task}_indirect_effect.pt")
        torch.save({"function_vector": fv, "top_heads": th, "n_top_heads": args.n_top_heads,
                    "dataset_name": task, "model_name": args.model_name}, d / f"{task}_function_vector.pt")
        meta = {"task": task, "partner": PARTNER[task], "recipe": f"{args.n_each}+{args.n_each} demos, differ query",
                "n_query_pool": len(qpool), "train_overlap": len(tr_over), "train_differ": len(tr_diff),
                "fv_norm": float(fv.norm()), "top_heads": th, "n_trials": args.n_trials,
                "n_ie_trials": args.n_ie_trials, "filter_to_correct_icl": True}
        json.dump(meta, open(d / f"{task}_function_vector_metadata.json", "w"), indent=2)
        print(f"  saved FV norm={fv.norm():.2f}  top5 heads={th[:5]}")
        summary[task] = {"status": "ok", **meta}

    json.dump(summary, open(Path(args.output_root) / "constrained_fv_summary.json", "w"), indent=2)
    print("\nDONE", Path(args.output_root) / "constrained_fv_summary.json")


if __name__ == "__main__":
    main()
