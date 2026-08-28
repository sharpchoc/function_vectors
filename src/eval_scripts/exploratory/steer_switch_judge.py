#!/usr/bin/env python
"""Behavioral task-switch steering on paired 1-shot prompts (GPT-J-6B).

Take a 1-shot prompt whose single ICL demo is the *source* task, inject the mean
function-difference vector at a chosen token site + layer, and measure how often the steered
model's sampled answers are correct for the *target* task (GPT-4-judged).

  steer toward target: add  (sign * alpha * Delta_site(L))  at one prompt position.
  Delta_site = mean_w[ act(f1) - act(f2) ] from artifacts/oneshot_paired_graded/<pair>/ shards.
    - site "label": Delta_label  (source role = demo last_label_token) injected at the demo label token.
    - site "final": Delta_final  (target role = query last_prompt_token) injected at the final prompt token.
  sign: +1 if target is f1 else -1   (Delta is f1-f2).

4 directions x 2 sites x layer-sweep x alpha-sweep -> per-(layer,alpha) target-task accuracy
over n_samples temperature-1 generations per query. alpha=0 baseline is layer/site-independent,
generated once per direction.

Two stages (run separately):
  --stage generate : loads GPT-J + baukit, writes generations.jsonl per (direction, site) and baseline.
  --stage judge    : no model; GPT-4.1-judges the target task, writes judged.jsonl + accuracy.json.

Reuses helpers from steer_label_to_query.py (load_capture_diffs, build_prompt_data,
extract_positions, pick_demo_pair, build_input_to_outputs, load_task_json) and
judge_oneshot_paired.py (JUDGE_SYSTEMS, extract_answer, get_openai_key, judge).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # src/ (for utils.*)
sys.path.insert(0, str(HERE))               # src/eval_scripts/ (for sibling imports)

from utils.prompt_utils import get_token_meta_labels  # noqa: E402
from utils.paths import ARTIFACTS_ROOT, LABEL_GEOMETRY_DIR  # noqa: E402
from steer_label_to_query import (  # noqa: E402
    load_capture_diffs, build_prompt_data, extract_positions, pick_demo_pair,
    build_input_to_outputs, load_task_json,
)
from judge_oneshot_paired import (  # noqa: E402
    JUDGE_SYSTEMS, extract_answer, get_openai_key, _judge_batch,
)
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

# direction -> pair, source task (demo context), target task (steer toward + judge), sign on Delta=f1-f2.
DIRECTIONS = {
    "synonym_to_antonym":          dict(pair="antonym_synonym",          source="synonym",     target="antonym",     sign=+1.0),
    "antonym_to_synonym":          dict(pair="antonym_synonym",          source="antonym",     target="synonym",     sign=-1.0),
    "prev_number_to_next_number":  dict(pair="next_number_prev_number",  source="prev_number", target="next_number", sign=+1.0),
    "next_number_to_prev_number":  dict(pair="next_number_prev_number",  source="next_number", target="prev_number", sign=-1.0),
    "prev_number_digits_to_next_number_digits": dict(pair="next_number_digits_prev_number_digits", source="prev_number_digits", target="next_number_digits", sign=+1.0),
    "next_number_digits_to_prev_number_digits": dict(pair="next_number_digits_prev_number_digits", source="next_number_digits", target="prev_number_digits", sign=-1.0),
}
SITES = ["label", "final"]


def judge_system_task(target_task):
    """Map a target task to its GPT-4 judge prompt key (digit variants reuse the number judges)."""
    return target_task.replace("_digits", "")


def is_multiword(target_task):
    """Number answers (word or digit form) may be multi-token phrases -> keep full phrase."""
    return "number" in target_task


def parse_args():
    p = argparse.ArgumentParser(description="Behavioral task-switch steering + GPT-4 judge.")
    p.add_argument("--stage", choices=["generate", "judge"], required=True)
    p.add_argument("--directions", nargs="+", default=sorted(DIRECTIONS), choices=sorted(DIRECTIONS))
    p.add_argument("--sites", nargs="+", default=SITES, choices=SITES)
    p.add_argument("--layers", type=int, nargs="+",
                   default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0],
                   help="Nonzero alphas (alpha=0 baseline is generated separately, once per direction).")
    p.add_argument("--n_queries", type=int, default=100)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--gen_batch", type=int, default=20, help="Queries per generate call (x n_samples rows).")
    p.add_argument("--capture_root", type=Path, default=ARTIFACTS_ROOT / "oneshot_paired_graded")
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--output_root", type=Path, default=LABEL_GEOMETRY_DIR / "oneshot_switch_steering")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--judge_model", type=str, default="gpt-4.1")
    p.add_argument("--judge_batch_size", type=int, default=50)
    p.add_argument("--judge_workers", type=int, default=24, help="Concurrent judge requests.")
    p.add_argument("--rejudge", action="store_true", help="Re-judge files even if accuracy.json exists.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Prompt set (shared across all conditions of a direction; seeded)
# --------------------------------------------------------------------------- #
def build_query_set(source_records, target_records, n_queries, seed):
    """Shared input words (valid input under both source and target); deterministic subset."""
    src_in = build_input_to_outputs(source_records)
    tgt_in = build_input_to_outputs(target_records)
    shared = sorted(set(src_in).intersection(tgt_in))
    rng = np.random.default_rng(seed)
    rng.shuffle(shared)
    return shared[:n_queries]


def build_prompts(direction, tokenizer, model_config, args):
    """Return list of dicts {query, demo_in, demo_out, prompt, label_idx, final_idx}."""
    cfg = DIRECTIONS[direction]
    source_records = load_task_json(args.root_data_dir, cfg["source"])
    target_records = load_task_json(args.root_data_dir, cfg["target"])
    queries = build_query_set(source_records, target_records, args.n_queries, args.seed)
    rng_master = np.random.default_rng(args.seed)
    prompts = []
    for q in queries:
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        demo_in, demo_out = pick_demo_pair(source_records, q, rng)
        prompt_data = build_prompt_data(demo_in, demo_out, q, demo_out)
        token_labels, prompt_string = get_token_meta_labels(
            prompt_data, tokenizer, query=q, prepend_bos=model_config["prepend_bos"])
        label_idx, final_idx = extract_positions(token_labels)
        prompts.append(dict(query=q, demo_in=demo_in, demo_out=demo_out,
                            prompt=prompt_string, label_idx=label_idx, final_idx=final_idx))
    return prompts


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def make_hook(edit_layer, add_vec, idx_per_row, n_rows):
    """baukit edit_output hook: add add_vec at per-row position, ONLY on the prompt forward
    (seq_len>1); cached single-token steps are untouched so the perturbation propagates via KV."""
    rows = torch.arange(n_rows, device=add_vec.device)

    def hook(output, layer_name):
        if isinstance(output, tuple) and int(layer_name.split(".")[2]) == edit_layer:
            if output[0].shape[1] > 1:
                output[0][rows, idx_per_row] += add_vec
        return output
    return hook


def generate_condition(model, tokenizer, layer_hook_names, prompts, site, edit_layer, add_vec,
                       n_samples, max_new_tokens, gen_batch, device, dtype):
    """Generate n_samples temp-1 completions per prompt with optional steering. Returns list of
    (prompt_index, sample_index, answer_text_raw) for every (prompt, sample)."""
    from baukit import TraceDict
    out_rows = []
    addv = None if add_vec is None else add_vec.to(device=device, dtype=dtype)
    for start in range(0, len(prompts), gen_batch):
        chunk = prompts[start:start + gen_batch]
        # replicate each prompt n_samples times (explicit order: prompt-major).
        strings, base_idx = [], []
        for j, pr in enumerate(chunk):
            pos = pr["label_idx"] if site == "label" else pr["final_idx"]
            for _ in range(n_samples):
                strings.append(pr["prompt"])
                base_idx.append(pos)
        enc = tokenizer(strings, return_tensors="pt", padding=True).to(device)
        input_len = enc.input_ids.shape[1]
        pad_len = input_len - enc.attention_mask.sum(dim=1)  # left-pad amount per row
        if site == "final":
            idx_per_row = torch.full((len(strings),), input_len - 1, device=device, dtype=torch.long)
        else:
            idx_per_row = (pad_len + torch.tensor(base_idx, device=device)).to(torch.long)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0,
                          pad_token_id=tokenizer.eos_token_id)
        if addv is None:
            gen = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, **gen_kwargs)
        else:
            hook = make_hook(edit_layer, addv, idx_per_row, len(strings))
            with TraceDict(model, layers=layer_hook_names, edit_output=hook):
                gen = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, **gen_kwargs)
        texts = tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True)
        for r, txt in enumerate(texts):
            pidx = start + (r // n_samples)
            sidx = r % n_samples
            out_rows.append((pidx, sidx, txt))
    return out_rows


def done_keys(path):
    """Set of (site, layer, alpha) already present in a generations.jsonl (for resume)."""
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            keys.add((r["site"], r["layer"], r["alpha"]))
    return keys


def stage_generate(args):
    torch.set_grad_enabled(False)
    from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
    set_seed(args.seed)
    print("Loading GPT-J ...")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()
    device = model.device
    dtype = next(model.parameters()).dtype
    layer_hook_names = model_config["layer_hook_names"]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder generation

    # cache deltas per pair (sign applied per direction below).
    delta_cache = {}
    deltas_used = {}

    for direction in args.directions:
        cfg = DIRECTIONS[direction]
        pair = cfg["pair"]
        if pair not in delta_cache:
            dl, df, n_layers, n_words = load_capture_diffs(args.capture_root / pair)
            assert n_layers == model_config["n_layers"], f"{n_layers} != {model_config['n_layers']}"
            delta_cache[pair] = (dl, df)
            print(f"[{pair}] deltas loaded: n_layers={n_layers}, n_words={n_words}")
        delta_label, delta_final = delta_cache[pair]
        sign = cfg["sign"]

        prompts = build_prompts(direction, tokenizer, model_config, args)
        print(f"\n=== {direction} (source={cfg['source']} -> target={cfg['target']}, sign={sign:+.0f}) "
              f"| {len(prompts)} prompts ===")

        deltas_used[direction] = {
            "label": {int(L): float(torch.linalg.norm(delta_label[L])) for L in args.layers},
            "final": {int(L): float(torch.linalg.norm(delta_final[L])) for L in args.layers},
        }

        # --- baseline (alpha=0, no injection): once per direction ---
        base_dir = args.output_root / f"{direction}__baseline"
        base_dir.mkdir(parents=True, exist_ok=True)
        base_path = base_dir / "generations.jsonl"
        if ("baseline", -1, 0.0) not in done_keys(base_path):
            rows = generate_condition(model, tokenizer, layer_hook_names, prompts, "final", -1, None,
                                      args.n_samples, args.max_new_tokens, args.gen_batch, device, dtype)
            with open(base_path, "a") as f:
                for pidx, sidx, txt in rows:
                    pr = prompts[pidx]
                    f.write(json.dumps(dict(direction=direction, site="baseline", layer=-1, alpha=0.0,
                                            query=pr["query"], demo_in=pr["demo_in"],
                                            sample_idx=sidx, raw=txt)) + "\n")
            print(f"  baseline: {len(rows)} samples -> {base_path}")
        else:
            print("  baseline: already done (resume)")

        # --- steered conditions ---
        for site in args.sites:
            site_delta = delta_label if site == "label" else delta_final
            out_dir = args.output_root / f"{direction}__{site}"
            out_dir.mkdir(parents=True, exist_ok=True)
            gen_path = out_dir / "generations.jsonl"
            have = done_keys(gen_path)
            with open(gen_path, "a") as f:
                for L in args.layers:
                    base_vec = (sign * site_delta[L]).to(device)
                    for alpha in args.alphas:
                        if (site, int(L), float(alpha)) in have:
                            continue
                        add_vec = alpha * base_vec
                        rows = generate_condition(model, tokenizer, layer_hook_names, prompts, site,
                                                  int(L), add_vec, args.n_samples, args.max_new_tokens,
                                                  args.gen_batch, device, dtype)
                        for pidx, sidx, txt in rows:
                            pr = prompts[pidx]
                            f.write(json.dumps(dict(direction=direction, site=site, layer=int(L),
                                                    alpha=float(alpha), query=pr["query"],
                                                    demo_in=pr["demo_in"], sample_idx=sidx, raw=txt)) + "\n")
                        f.flush()
                    print(f"  {site} L{L}: done alphas {args.alphas}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "deltas_used.json").write_text(json.dumps(deltas_used, indent=2))
    print(f"\nwrote {args.output_root/'deltas_used.json'}")


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
def _judge_batch_retry(pairs, system, judge_model, api_key, attempts=5):
    """Judge a batch robustly:
      - transient HTTP / JSON-decode errors -> exponential-backoff retry.
      - verdict-count mismatch (deterministic at temp 0; retry won't help) -> recursively SPLIT
        the batch and judge halves; a single stubborn pair defaults to correct=False (rare).
    Always returns exactly len(pairs) verdicts, in order.
    """
    last = None
    for k in range(attempts):
        try:
            return _judge_batch(pairs, system, judge_model, api_key)
        except json.JSONDecodeError as e:        # transient/garbled response
            last = e
            time.sleep(min(2 ** k, 30))
        except ValueError as e:                  # count mismatch from _judge_batch
            if len(pairs) <= 1:
                p = pairs[0]
                print(f"    WARN: unjudgeable pair, defaulting False: {p}", flush=True)
                return [{"input": p["input"], "answer": p["answer"], "correct": False}]
            mid = len(pairs) // 2
            return (_judge_batch_retry(pairs[:mid], system, judge_model, api_key, attempts)
                    + _judge_batch_retry(pairs[mid:], system, judge_model, api_key, attempts))
        except Exception as e:                   # noqa: BLE001 (HTTP/rate-limit)
            last = e
            time.sleep(min(2 ** k, 30))
    raise RuntimeError(f"judge batch failed after {attempts} attempts: {last}")


def judge_parallel(rows, system, judge_model, api_key, batch_size, workers):
    """Parallel, order-preserving judge over batches of (input, answer) pairs."""
    pairs = [{"input": r["query_input"], "answer": r["generated"]} for r in rows]
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    results = [None] * len(batches)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_judge_batch_retry, b, system, judge_model, api_key): i
                for i, b in enumerate(batches)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if done % 25 == 0 or done == len(batches):
                print(f"    judged {done}/{len(batches)} batches", flush=True)
    return [v for b in results for v in b]


def judge_file(gen_path, target_task, api_key, args):
    """Judge every generation in gen_path for target_task; write judged.jsonl + accuracy.json."""
    out_dir = gen_path.parent
    if (out_dir / "accuracy.json").exists() and not args.rejudge:
        print(f"  {out_dir.name}: accuracy.json exists, skipping (use --rejudge to force)")
        return json.load(open(out_dir / "accuracy.json"))
    multiword = is_multiword(target_task)
    rows = [json.loads(l) for l in gen_path.read_text().splitlines() if l.strip()]
    for r in rows:
        r["generated"] = extract_answer(r["raw"], multiword=multiword)
        r["query_input"] = r["query"]
    verdicts = judge_parallel(rows, JUDGE_SYSTEMS[judge_system_task(target_task)], args.judge_model,
                              api_key, args.judge_batch_size, args.judge_workers)
    for r, v in zip(rows, verdicts):
        r["judge_correct"] = bool(v["correct"])
    (out_dir / "judged.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # aggregate: per (layer, alpha) accuracy = correct samples / total samples; CI from per-prompt frac.
    from collections import defaultdict
    by_cond = defaultdict(lambda: defaultdict(list))  # (layer,alpha) -> query -> [0/1]
    for r in rows:
        by_cond[(r["layer"], r["alpha"])][r["query"]].append(1.0 if r["judge_correct"] else 0.0)
    acc = {}
    for (layer, alpha), per_q in by_cond.items():
        fracs = [float(np.mean(v)) for v in per_q.values()]
        flat = [x for v in per_q.values() for x in v]
        mean = float(np.mean(flat))
        sem = float(np.std(fracs, ddof=1) / np.sqrt(len(fracs))) if len(fracs) > 1 else float("nan")
        acc[f"{layer}|{alpha}"] = dict(layer=layer, alpha=alpha, accuracy=mean,
                                       ci95=(1.96 * sem if sem == sem else None),
                                       n_prompts=len(per_q), n_samples=len(flat))
    (out_dir / "accuracy.json").write_text(json.dumps(acc, indent=2))
    overall = sum(r["judge_correct"] for r in rows) / max(1, len(rows))
    print(f"  {gen_path.parent.name}: {len(rows)} samples judged for {target_task}; overall acc {overall:.3f}")
    return acc


def stage_judge(args):
    api_key = get_openai_key()
    for direction in args.directions:
        target = DIRECTIONS[direction]["target"]
        # baseline
        base_path = args.output_root / f"{direction}__baseline" / "generations.jsonl"
        if base_path.exists():
            judge_file(base_path, target, api_key, args)
        for site in args.sites:
            gen_path = args.output_root / f"{direction}__{site}" / "generations.jsonl"
            if gen_path.exists():
                judge_file(gen_path, target, api_key, args)


def main():
    args = parse_args()
    if args.stage == "generate":
        stage_generate(args)
    else:
        stage_judge(args)


if __name__ == "__main__":
    main()
