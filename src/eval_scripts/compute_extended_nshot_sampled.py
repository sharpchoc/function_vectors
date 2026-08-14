#!/usr/bin/env python
"""GPT-J n-shot accuracy sweep over dataset_files/extended_tasks (142 tasks) with
TEMPERATURE-1.0 SAMPLED generations, scored by full-label exact match.

Protocol (user-locked 2026-08-13): for every task and every n in 0..6, 50 prompts; each
prompt independently samples n demos + 1 query (all distinct indices) uniformly from the
task's FULL example list. One sampled generation per prompt (pure ancestral: temperature 1.0,
top_k=0, top_p=1.0, max_new_tokens=12); the continuation is cut at the first newline and
stripped; correct iff it exactly matches the gold label (any acceptable form, stripped).
All generations are stored so alternative metrics can be derived without new GPU time.

NOTE: reads task JSONs directly from dataset_files/extended_tasks (utils.prompt_utils.
load_dataset only searches abstractive/extractive and is bypassed on purpose — this protocol
needs no split machinery).

Modes:
  build          (CPU) write prompt-spec shards to artifacts/extended_tasks_nshot/shards/
  run --shard K  (GPU) batched left-padded sampled generation for one shard; per-task results
                 to artifacts/extended_tasks_nshot/results/<task>.json (resumable per task).

Wall-clock tricks: all prompts tokenized up front, globally sorted by token length, batched
under a token budget, left-padded, one HF generate() call per batch.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from utils.paths import ARTIFACTS_ROOT  # noqa: E402
from utils.prompt_utils import create_prompt, word_pairs_to_prompt_data  # noqa: E402

TASK_ROOT = REPO_ROOT / "dataset_files" / "extended_tasks"
OUT_ROOT = ARTIFACTS_ROOT / "extended_tasks_nshot"
N_SHOTS = list(range(0, 7))
N_PROMPTS = 50
PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["build", "run"], required=True)
    p.add_argument("--num_shards", type=int, default=4)
    p.add_argument("--shard", type=int, default=None, help="run mode: shard index")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42, help="base seed for generation batches")
    p.add_argument("--batch_tokens", type=int, default=14000)
    p.add_argument("--max_batch_size", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--limit_tasks", type=int, default=None, help="smoke: only first K tasks of the shard")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def task_names():
    return sorted(f.stem for f in TASK_ROOT.glob("*.json") if f.stem != "manifest")


def gold_forms(output):
    """Acceptable gold strings (originals sometimes store lists of alternatives)."""
    if isinstance(output, list):
        return [str(o).strip() for o in output]
    return [str(output).strip()]


def build_task_specs(task):
    data = json.load(open(TASK_ROOT / f"{task}.json"))
    N = len(data)
    specs = []
    for n in N_SHOTS:
        for i in range(N_PROMPTS):
            seed = int(hashlib.sha256(f"{task}|{n}|{i}".encode()).hexdigest()[:12], 16)
            rng = random.Random(seed)
            idx = rng.sample(range(N), n + 1)
            demos, q = idx[:n], idx[n]
            word_pairs = {"input": [data[j]["input"] for j in demos],
                          "output": [data[j]["output"] for j in demos]}
            qt = {"input": data[q]["input"], "output": data[q]["output"]}
            prompt_data = word_pairs_to_prompt_data(
                word_pairs, query_target_pair=qt, prepend_bos_token=True, shuffle_labels=False,
                prefixes=PREFIXES, separators=SEPARATORS)
            specs.append({
                "task": task, "n": n, "i": i, "sample_seed": seed,
                "prompt": create_prompt(prompt_data),
                "query": qt["input"] if not isinstance(qt["input"], list) else qt["input"][0],
                "gold": gold_forms(qt["output"]),
            })
    return specs


def mode_build(args):
    (OUT_ROOT / "shards").mkdir(parents=True, exist_ok=True)
    tasks = task_names()
    assert len(tasks) == 142, f"expected 142 tasks, found {len(tasks)}"
    per_task, cost = {}, {}
    for t in tasks:
        specs = build_task_specs(t)
        assert len(specs) == len(N_SHOTS) * N_PROMPTS
        per_task[t] = specs
        cost[t] = sum(len(s["prompt"]) for s in specs) // 4  # rough token estimate
    # spread token-heavy tasks: sort by cost desc, assign to currently-lightest shard
    shards = [[] for _ in range(args.num_shards)]
    loads = [0] * args.num_shards
    for t in sorted(tasks, key=lambda t: -cost[t]):
        k = loads.index(min(loads))
        shards[k].append(t)
        loads[k] += cost[t]
    plan = {"num_shards": args.num_shards, "n_shots": N_SHOTS, "n_prompts": N_PROMPTS,
            "shards": {str(k): shards[k] for k in range(args.num_shards)},
            "est_tokens_per_shard": {str(k): loads[k] for k in range(args.num_shards)}}
    json.dump(plan, open(OUT_ROOT / "shard_plan.json", "w"), indent=1)
    for k in range(args.num_shards):
        out = OUT_ROOT / "shards" / f"shard{k}.json"
        json.dump([s for t in shards[k] for s in per_task[t]], open(out, "w"))
        print(f"shard {k}: {len(shards[k])} tasks, ~{loads[k]:,} est tokens -> {out}")
    total = sum(len(v) for v in per_task.values())
    print(f"built {total} specs ({len(tasks)} tasks x {len(N_SHOTS)} n x {N_PROMPTS})")


def mode_run(args):
    import torch
    from utils.model_utils import load_gpt_model_and_tokenizer

    assert args.shard is not None
    (OUT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    specs = json.load(open(OUT_ROOT / "shards" / f"shard{args.shard}.json"))

    shard_tasks = list(dict.fromkeys(s["task"] for s in specs))
    if args.limit_tasks:
        shard_tasks = shard_tasks[:args.limit_tasks]
    done = {t for t in shard_tasks if (OUT_ROOT / "results" / f"{t}.json").exists() and not args.overwrite}
    todo_tasks = [t for t in shard_tasks if t not in done]
    specs = [s for s in specs if s["task"] in set(todo_tasks)]
    print(f"shard {args.shard}: {len(todo_tasks)} tasks to run ({len(done)} already done), "
          f"{len(specs)} prompts", flush=True)
    if not specs:
        return

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model = model.to(torch.bfloat16)
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("tokenizing", flush=True)
    for s in specs:
        s["ids"] = tokenizer(s["prompt"], truncation=False).input_ids
    specs.sort(key=lambda s: len(s["ids"]))

    # batches under a token budget (padded length x batch size), capped batch size
    batches, cur, cur_max = [], [], 0
    for s in specs:
        L = len(s["ids"])
        if cur and ((max(cur_max, L)) * (len(cur) + 1) > args.batch_tokens or len(cur) >= args.max_batch_size):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(s)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    print(f"{len(batches)} batches", flush=True)

    remaining = {t: sum(1 for s in specs if s["task"] == t) for t in todo_tasks}
    records = {t: [] for t in todo_tasks}
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(batches):
            enc = tokenizer([s["prompt"] for s in batch], return_tensors="pt", padding=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            gen_seed = args.seed * 1_000_003 + args.shard * 10_007 + bi
            torch.manual_seed(gen_seed)
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id, use_cache=True)
            cont = out[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(cont, skip_special_tokens=True)
            for s, text in zip(batch, texts):
                pred = text.split("\n")[0].strip()
                rec = {k: s[k] for k in ("task", "n", "i", "sample_seed", "query", "gold")}
                rec.update(generation=text, pred=pred, gen_seed=gen_seed,
                           match=pred in s["gold"])
                records[s["task"]].append(rec)
                remaining[s["task"]] -= 1
                if remaining[s["task"]] == 0:
                    rs = sorted(records.pop(s["task"]), key=lambda r: (r["n"], r["i"]))
                    json.dump(rs, open(OUT_ROOT / "results" / f"{s['task']}.json", "w"))
                    accs = {n: sum(r["match"] for r in rs if r["n"] == n) / N_PROMPTS for n in N_SHOTS}
                    print(f"  [{s['task']}] done; acc by n: " +
                          " ".join(f"{n}:{a:.2f}" for n, a in accs.items()), flush=True)
            if bi % 20 == 0:
                el = time.time() - t0
                print(f"batch {bi}/{len(batches)} ({el:.0f}s)", flush=True)
    assert not records, f"incomplete tasks: {list(records)}"
    print(f"shard {args.shard} DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "build":
        mode_build(args)
    else:
        mode_run(args)
