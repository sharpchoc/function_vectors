"""
Capture GPT-J residual-stream activations on magnitude/identity 3+1+1 disambiguating prompts,
labeled by whether the model answered the query correctly (for later analysis).

Per task t in {magnitude, identity}:
  - n=200 prompts = 3 overlap demos + 1 differentiator demo (t's output -> disambiguates to t)
    + 1 differentiator query (scored against t's output). Queries are BALANCED over the 150
    differentiator items (each used 1-2x). Same seed for both tasks => paired prompts (identical
    demo/query INPUTS; only the differentiator demo's output label differs by task).
  - Correctness: batched greedy generation to newline, whole-answer token-id exact match to t's
    gold (and partner_match = produced the OTHER function's answer).
  - Activations: residual stream per layer (+embeddings) captured at pre_label / first_label /
    last_label tokens of all 4 demos, plus the query's final predictive position. One row per
    (prompt, position), tagged with role/region/correctness/metadata.

Outputs: artifacts/magnitude_identity_activations/gpt-j-6b/<task>.pt  (+ <task>_correctness.json)

Run:
  python src/eval_scripts/capture_magnitude_identity_activations.py --model_name EleutherAI/gpt-j-6B
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "utils"))
sys.path.append(os.path.join(HERE, ".."))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt
from utils.paths import ARTIFACTS_ROOT
from extract_residual_stream_activations import get_residual_stack, selected_token_records
from eval_scripts.eval_ambiguous_disambiguation import split_overlap_differ, batched_generate

AMBIG_DIR = os.path.join(HERE, "..", "..", "dataset_files", "ambiguous")
PAIR = ("magnitude", "identity")
# roles we keep (drop the backward-compat aliases 'label_token' / 'final_token')
DEMO_ROLES = {"pre_label_token", "first_label_token", "last_label_token"}


def load_task(name):
    return json.load(open(os.path.join(AMBIG_DIR, name + ".json")))


def balanced_query_list(differ_idx, n_prompts, rng):
    """Spread queries as evenly as possible over the differentiator items."""
    base = list(differ_idx)
    reps = n_prompts // len(base)
    rem = n_prompts - reps * len(base)
    q = base * reps + list(rng.choice(base, size=rem, replace=False))
    rng.shuffle(q)
    return [int(x) for x in q]


def build_prompts(task_data, partner_data, overlap_idx, differ_idx, n_prompts, rng, n_shared=3):
    """Return list of per-prompt dicts (paired across tasks via the shared rng seed)."""
    queries = balanced_query_list(differ_idx, n_prompts, rng)
    prompts = []
    for query_idx in queries:
        shared = [int(x) for x in rng.choice(overlap_idx, size=n_shared, replace=False)]
        diff_pool = [d for d in differ_idx if d != query_idx]
        diff_demo = int(rng.choice(diff_pool, size=1)[0])
        demo_idxs = shared + [diff_demo]                       # demos 1..3 overlap, 4 differentiator
        inputs = [task_data[i]["input"] for i in demo_idxs]
        outputs = [task_data[i]["output"] for i in demo_idxs]
        query = task_data[query_idx]
        prompt_data = word_pairs_to_prompt_data(
            {"input": inputs, "output": outputs},
            query_target_pair={"input": query["input"], "output": query["output"]},
            prepend_bos_token=False, shuffle_labels=False, prepend_space=True,
        )
        prompts.append({
            "prompt_data": prompt_data,
            "prompt_str": create_prompt(prompt_data),
            "demo_idxs": demo_idxs,
            "demos": [{"input": task_data[i]["input"], "output": task_data[i]["output"]} for i in demo_idxs],
            "query_idx": query_idx,
            "query_input": query["input"],
            "query_gold": query["output"],
            "partner_gold": partner_data[query_idx]["output"],
        })
    return prompts


def _num(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def score_generation(tok, new_ids, gold, partner):
    """Generate-until-newline then NUMERIC equality (decimals are multi-token; do NOT use
    normalize_answer, which strips '.' -> the round/truncate bug)."""
    parsed = tok.decode(new_ids).split("\n")[0].strip()
    p, g, a = _num(parsed), _num(gold), _num(partner)
    correct = (p is not None and g is not None and p == g)
    partner_match = (p is not None and a is not None and p == a)
    return bool(correct), bool(partner_match), parsed


@torch.inference_mode()
def run_task(task, model, tok, cfg, n_prompts, seed, batch_size, max_new_tokens, out_dir, device):
    da, db = load_task(PAIR[0]), load_task(PAIR[1])
    overlap, differ = split_overlap_differ(da, db)
    task_data = da if task == PAIR[0] else db
    partner_data = db if task == PAIR[0] else da

    rng = np.random.RandomState(seed)                          # same seed both tasks => paired
    prompts = build_prompts(task_data, partner_data, overlap, differ, n_prompts, rng)
    prompt_strs = [p["prompt_str"] for p in prompts]

    # ---- correctness via batched greedy generation ----
    old_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    new_tokens = batched_generate(model, tok, prompt_strs, max_new_tokens, batch_size, device)
    tok.padding_side = old_side
    for p, new in zip(prompts, new_tokens):
        p["correct"], p["partner_match"], p["parsed"] = score_generation(
            tok, new, p["query_gold"], p["partner_gold"])
        p["generated"] = tok.decode(new)

    # ---- residual-stream capture at target positions ----
    act_rows, meta_rows = [], []
    for pi, p in enumerate(prompts):
        residual_stack, token_labels, _ = get_residual_stack(
            p["prompt_data"], model, cfg, tok, include_embeddings=True)   # (n_layers+1, seq, hidden)
        recs = [r for r in selected_token_records(token_labels)
                if r["token_role"] in DEMO_ROLES or r["token_role"] == "last_prompt_token"]
        for r in recs:
            pos = r["token_position"]
            act_rows.append(residual_stack[:, pos, :].to("cpu", torch.float16))
            is_query = r["token_role"] == "last_prompt_token"
            di = r["icl_example_index"]                         # 1..4 for demos, None for query
            meta = {
                "task": task, "prompt_index": pi,
                "token_role": "query_predictive_token" if is_query else r["token_role"],
                "demo_index": None if is_query else di,
                "region": ("differentiator" if (is_query or di == 4) else "overlap"),
                "token_position": pos, "token_text": r["token_text"],
                "query_input": p["query_input"], "query_gold": p["query_gold"],
                "partner_gold": p["partner_gold"],
                "generated": p["generated"], "parsed": p["parsed"],
                "correct": p["correct"], "partner_match": p["partner_match"],
            }
            if not is_query:
                meta["demo_input"] = p["demos"][di - 1]["input"]
                meta["demo_output"] = p["demos"][di - 1]["output"]
            meta_rows.append(meta)

    activations = torch.stack(act_rows, dim=0)                 # (n_rows, n_layers+1, hidden)
    os.makedirs(out_dir, exist_ok=True)
    config = {
        "model": cfg["name_or_path"], "task": task, "pair": "|".join(PAIR),
        "n_prompts": n_prompts, "seed": seed, "include_embeddings": True,
        "n_shared_demos": 3, "n_diff_demos": 1, "max_new_tokens": max_new_tokens,
        "n_layers_with_embed": activations.shape[1], "hidden_dim": activations.shape[2],
        "prefixes": {"input": "Q:", "output": "A:", "instructions": ""},
        "separators": {"input": "\n", "output": "\n\n", "instructions": ""},
        "prepend_bos_token": False, "prepend_space": True,
        "roles": sorted(DEMO_ROLES) + ["query_predictive_token"],
    }
    torch.save({"activations": activations, "metadata": meta_rows, "config": config},
               os.path.join(out_dir, task + ".pt"))

    acc = float(np.mean([p["correct"] for p in prompts]))
    pr = float(np.mean([p["partner_match"] for p in prompts]))
    with open(os.path.join(out_dir, task + "_correctness.json"), "w") as f:
        json.dump({"task": task, "n_prompts": n_prompts, "accuracy": acc, "partner_rate": pr,
                   "n_distinct_queries": len({p["query_idx"] for p in prompts}),
                   "records": [{"prompt_index": i, "query_input": p["query_input"],
                                "query_gold": p["query_gold"], "partner_gold": p["partner_gold"],
                                "parsed": p["parsed"], "correct": p["correct"],
                                "partner_match": p["partner_match"]}
                               for i, p in enumerate(prompts)]}, f, indent=2)

    print(f"[{task}] n={n_prompts} acc={acc:.3f} partner={pr:.3f} | "
          f"rows={activations.shape[0]} act_shape={tuple(activations.shape)} | "
          f"distinct_queries={len({p['query_idx'] for p in prompts})} "
          f"distinct_diff_demos={len({p['demo_idxs'][3] for p in prompts})}")
    return acc, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="EleutherAI/gpt-j-6B")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--out_dir", default=str(ARTIFACTS_ROOT / "magnitude_identity_activations" / "gpt-j-6b"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, cfg = load_gpt_model_and_tokenizer(args.model_name, device=device)
    model.eval()
    print(f"loaded {cfg['name_or_path']} | layers={cfg['n_layers']} resid={cfg['resid_dim']}")
    for task in PAIR:
        run_task(task, model, tok, cfg, args.n_prompts, args.seed,
                 args.batch_size, args.max_new_tokens, args.out_dir, device)


if __name__ == "__main__":
    main()
