"""
Mixed-task ICL probe.

Build prompts with 10 ICL examples where the first 5 are antonym (input,output)
pairs and the next 5 are synonym pairs, then query the model with either an
antonym or a synonym query word. Measure how often the correct answer's first
token lands in the model's top-1/2/3 next-token predictions.

Two conditions share the SAME mixed ICL context structure (5 antonym + 5 synonym
demonstrations); only the final query task differs. Produces a clustered bar
chart comparing top-k accuracy for the antonym-query vs synonym-query conditions.

Cross-prompt batching: all prompts in a condition are padded and run in a single
(or few) forward pass.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.model_utils import load_gpt_model_and_tokenizer


# --- inlined from utils.eval_utils (avoids the baukit import chain) ---
def get_answer_id(query, answer, tokenizer):
    source = tokenizer(query, truncation=False, padding=False).input_ids
    target = tokenizer(query + answer, truncation=False, padding=False).input_ids
    assert len(source) < len(target) < tokenizer.model_max_length
    return target[len(source):]


def compute_individual_token_rank(prob_dist, target_id):
    if isinstance(target_id, list):
        target_id = target_id[0]
    return torch.where(torch.argsort(prob_dist.squeeze(), descending=True) == target_id)[0].item()


def compute_top_k_accuracy(target_token_ranks, k=10):
    target_token_ranks = np.array(target_token_ranks)
    return (target_token_ranks < k).sum(axis=0) / len(target_token_ranks)
# ----------------------------------------------------------------------

# Default ICL template (matches prompt_utils defaults for GPT-J).
PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}


def build_primer(examples):
    """examples: list of {'input','output'} dicts -> ICL primer string."""
    s = ""
    for ex in examples:
        s += PREFIXES["input"] + ex["input"] + SEPARATORS["input"]
        s += PREFIXES["output"] + ex["output"] + SEPARATORS["output"]
    return s


def build_prompt(icl_examples, query_word):
    primer = build_primer(icl_examples)
    return primer + PREFIXES["input"] + query_word + SEPARATORS["input"] + PREFIXES["output"]


def sample_pairs(dataset, n, rng, exclude_inputs=None):
    """Sample n distinct pairs from dataset, skipping any whose input is in
    exclude_inputs (so the query word never appears among the ICL demos)."""
    exclude_inputs = exclude_inputs or set()
    idxs = [i for i in range(len(dataset)) if dataset[i]["input"] not in exclude_inputs]
    chosen = rng.choice(idxs, size=n, replace=False)
    return [dataset[int(i)] for i in chosen]


def build_condition(antonym, synonym, query_dataset, n_prompts, rng, icl_order="antonym_first"):
    """
    Build n_prompts prompts: 5 antonym + 5 synonym ICL examples (ordered per
    icl_order), then a query word drawn from query_dataset. Returns (prompts, targets).
    The query word is excluded from the ICL demos of both tasks.
    """
    prompts, targets = [], []
    for _ in range(n_prompts):
        # draw the query first, then exclude its input from the ICL demos
        q = query_dataset[int(rng.choice(len(query_dataset)))]
        excl = {q["input"]}
        ant_ex = sample_pairs(antonym, 5, rng, exclude_inputs=excl)
        syn_ex = sample_pairs(synonym, 5, rng, exclude_inputs=excl)
        if icl_order == "antonym_first":
            icl = ant_ex + syn_ex
        elif icl_order == "synonym_first":
            icl = syn_ex + ant_ex
        else:
            raise ValueError(f"unknown icl_order: {icl_order}")
        prompts.append(build_prompt(icl, q["input"]))
        targets.append(q["output"])
    return prompts, targets


def build_pure_condition(dataset, n_prompts, rng):
    """
    Reference baseline: 10 ICL examples from the SAME task as the query.
    The query word is excluded from the ICL demos. Returns (prompts, targets).
    """
    prompts, targets = [], []
    for _ in range(n_prompts):
        q = dataset[int(rng.choice(len(dataset)))]
        icl = sample_pairs(dataset, 10, rng, exclude_inputs={q["input"]})
        prompts.append(build_prompt(icl, q["input"]))
        targets.append(q["output"])
    return prompts, targets


@torch.no_grad()
def eval_condition(prompts, targets, model, tokenizer, batch_size):
    """Return rank list (rank of each target's first token in the next-token dist)."""
    old_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    rank_list = []
    try:
        for start in range(0, len(prompts), batch_size):
            bp = prompts[start:start + batch_size]
            bt = targets[start:start + batch_size]
            target_ids = [get_answer_id(p, t, tokenizer) for p, t in zip(bp, bt)]
            inputs = tokenizer(bp, return_tensors="pt", padding=True).to(model.device)
            last_idx = inputs.attention_mask.sum(dim=1) - 1
            logits = model(**inputs).logits
            out = logits[torch.arange(logits.shape[0], device=model.device), last_idx]
            for row, tid in zip(out, target_ids):
                rank_list.append(compute_individual_token_rank(row, tid))
    finally:
        tokenizer.padding_side = old_side
    return rank_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="EleutherAI/gpt-j-6b")
    p.add_argument("--n_prompts", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_dir", default="dataset_files/abstractive")
    p.add_argument("--output_dir", default="results/mixed_icl_antonym_synonym")
    p.add_argument("--icl_order", choices=["antonym_first", "synonym_first"], default="antonym_first",
                   help="order of the 10 ICL demos: 5 antonym then 5 synonym, or vice versa")
    p.add_argument("--baseline_path", default="results/mixed_icl_antonym_synonym_baseline/pure_baseline.json",
                   help="cache for the same-task 10-shot reference; computed once, reused across graphs")
    p.add_argument("--refresh_baseline", action="store_true",
                   help="recompute the same-task baseline even if the cache exists")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    antonym = json.load(open(os.path.join(args.dataset_dir, "antonym.json")))
    synonym = json.load(open(os.path.join(args.dataset_dir, "synonym.json")))
    print(f"antonym pairs: {len(antonym)}, synonym pairs: {len(synonym)}")

    model, tokenizer, _ = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()

    # Condition 1: antonym query.  Condition 2: synonym query.  Same ICL structure.
    conditions = {
        "antonym query": antonym,
        "synonym query": synonym,
    }

    results = {}
    for name, query_ds in conditions.items():
        prompts, targets = build_condition(antonym, synonym, query_ds, args.n_prompts, rng,
                                           icl_order=args.icl_order)
        ranks = eval_condition(prompts, targets, model, tokenizer, args.batch_size)
        acc = {K: float(compute_top_k_accuracy(ranks, K)) for K in (1, 2, 3)}
        results[name] = acc
        print(f"{name}: top1={acc[1]:.3f} top2={acc[2]:.3f} top3={acc[3]:.3f} (n={len(ranks)})")

    # Same-task 10-shot reference baseline (order-independent): compute once, cache, reuse.
    # Maps each query-task cluster to its pure-task accuracy.
    pure_query_ds = {"antonym query": antonym, "synonym query": synonym}
    if os.path.exists(args.baseline_path) and not args.refresh_baseline:
        baseline = json.load(open(args.baseline_path))["results"]
        print(f"loaded same-task baseline from {args.baseline_path}")
    else:
        baseline = {}
        for name, ds in pure_query_ds.items():
            bp, bt = build_pure_condition(ds, args.n_prompts, rng)
            branks = eval_condition(bp, bt, model, tokenizer, args.batch_size)
            baseline[name] = {K: float(compute_top_k_accuracy(branks, K)) for K in (1, 2, 3)}
            print(f"[baseline 10-shot same-task] {name}: "
                  f"top1={baseline[name][1]:.3f} top2={baseline[name][2]:.3f} top3={baseline[name][3]:.3f}")
        os.makedirs(os.path.dirname(args.baseline_path), exist_ok=True)
        with open(args.baseline_path, "w") as f:
            json.dump({"n_prompts": args.n_prompts, "seed": args.seed,
                       "description": "10-shot same-task ICL reference (antonym ICL->antonym, synonym ICL->synonym)",
                       "results": baseline}, f, indent=2)
    # json keys come back as strings when loaded from cache; normalize to int K.
    baseline = {c: {int(k): v for k, v in d.items()} for c, d in baseline.items()}

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"n_prompts": args.n_prompts, "seed": args.seed,
                   "icl_order": args.icl_order, "results": results,
                   "same_task_baseline": baseline}, f, indent=2)

    order_label = ("5 antonym + 5 synonym" if args.icl_order == "antonym_first"
                   else "5 synonym + 5 antonym")

    # Clustered bar chart: x = condition cluster, bars = top-1/2/3.
    cond_names = list(results.keys())
    ks = [1, 2, 3]
    x = np.arange(len(cond_names))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, K in enumerate(ks):
        vals = [results[c][K] for c in cond_names]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=f"top-{K}", color=colors[i], zorder=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, zorder=4)
        # Shaded reference overlay: same-task 10-shot accuracy at this top-k.
        bvals = [baseline[c][K] for c in cond_names]
        ax.bar(x + (i - 1) * width, bvals, width, facecolor=colors[i], alpha=0.3,
               hatch="////", edgecolor="0.25", linewidth=0.8, zorder=3)
        for xc, bv in zip(x + (i - 1) * width, bvals):
            ax.text(xc, bv + 0.01, f"{bv:.2f}", ha="center", va="bottom",
                    fontsize=8, color="0.25", zorder=4)
    # proxy legend handle for the reference overlay
    ax.bar(np.nan, np.nan, facecolor="0.7", alpha=0.3, hatch="////", edgecolor="0.25",
           label="same-task 10-shot (ref)")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_names)
    ax.set_ylabel("accuracy (correct first token in top-k)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Mixed ICL ({order_label}) — top-k accuracy by query task\n"
                 f"GPT-J, n={args.n_prompts} prompts/condition")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.output_dir, "topk_accuracy_bar.png")
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
