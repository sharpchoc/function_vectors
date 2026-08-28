#!/usr/bin/env python
"""Chat-template ICL transfer: Qwen2.5-7B-Instruct 6-shot accuracy over the 117-task
extended_tasks working pool (manifest.json), three prompt formats per task.

Mirrors the GPT-J extended n-shot sweep (compute_extended_nshot_sampled.py) exactly:
same per-(task, n, i) demo/query sampling (seed = sha256(f"{task}|{n}|{i}")[:12], 50
prompts/task), same T=1.0 pure ancestral sampled generation (top_k=0, top_p=1.0,
max_new_tokens=12), same metric (continuation cut at first newline, stripped, exact
match vs stripped gold). Only the prompt rendering differs by format:

  chat_blank_system  tokenizer.apply_chat_template with an EXPLICIT empty system message
                     (Qwen2.5's template auto-inserts its default "You are Qwen, ..."
                     system prompt when none is passed) + one user/assistant turn per
                     demo + the query as a final user turn + generation prompt.
  chat_no_system     chat_blank_system with the leading "<|im_start|>system\n<|im_end|>\n"
                     stripped (asserted present) — no system block at all.
  plain              the classic "Q:"/"A:" text format (word_pairs_to_prompt_data with
                     default prefixes/separators), prepend_bos_token=False (Qwen has no
                     BOS convention).

Qwen2.5 has no thinking mode, so "no thinking" holds by construction.

Per-(format, task) results (full generation records, resumable) go to
artifacts/chat_template_transfer/ext117_6shot/<format>/<task>.json.
Sharding: sorted(pool)[shard_idx::shard_n].
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
OUT_ROOT = ARTIFACTS_ROOT / "chat_template_transfer" / "ext117_6shot"
FORMATS = ["chat_blank_system", "chat_no_system", "plain"]
PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}
SYSTEM_PREFIX = "<|im_start|>system\n<|im_end|>\n"

# User-confirmed rendering (2026-08-28): blank system block, then user/assistant turns,
# query as final user turn, generation prompt last. Checked against the live tokenizer
# in assert_template() before any GPU work.
EXPECTED_EXAMPLE = (
    "<|im_start|>system\n<|im_end|>\n"
    "<|im_start|>user\nChad<|im_end|>\n"
    "<|im_start|>assistant\nN'Djamena<|im_end|>\n"
    "<|im_start|>user\nUnited States of America<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--formats", nargs="+", default=FORMATS, choices=FORMATS)
    p.add_argument("--n", type=int, default=6, help="number of demos per prompt")
    p.add_argument("--n_prompts", type=int, default=50)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42, help="base seed for generation batches")
    p.add_argument("--batch_tokens", type=int, default=14000)
    p.add_argument("--max_batch_size", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--limit_tasks", type=int, default=None, help="smoke: only first K tasks of the shard")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def pool_tasks():
    manifest = json.load(open(TASK_ROOT / "manifest.json"))
    return sorted(manifest["tasks"])


def chat_messages(demos, query_input):
    msgs = [{"role": "system", "content": ""}]
    for d in demos:
        msgs.append({"role": "user", "content": str(d["input"])})
        msgs.append({"role": "assistant", "content": str(d["output"]).strip()})
    msgs.append({"role": "user", "content": str(query_input)})
    return msgs


def render_prompt(fmt, tokenizer, demos, qt):
    if fmt == "plain":
        word_pairs = {"input": [d["input"] for d in demos], "output": [d["output"] for d in demos]}
        prompt_data = word_pairs_to_prompt_data(
            word_pairs, query_target_pair=dict(qt), prepend_bos_token=False, shuffle_labels=False,
            prefixes=PREFIXES, separators=SEPARATORS)
        return create_prompt(prompt_data)
    s = tokenizer.apply_chat_template(
        chat_messages(demos, qt["input"]), tokenize=False, add_generation_prompt=True)
    if fmt == "chat_no_system":
        assert s.startswith(SYSTEM_PREFIX), f"unexpected template head: {s[:60]!r}"
        s = s[len(SYSTEM_PREFIX):]
    return s


def assert_template(tokenizer):
    demos = [{"input": "Chad", "output": "N'Djamena"}]
    got = render_prompt("chat_blank_system", tokenizer,
                        demos, {"input": "United States of America", "output": "Washington, D.C."})
    assert got == EXPECTED_EXAMPLE, (
        "chat template no longer renders the user-confirmed format — STOP and show the user.\n"
        f"got: {got!r}")


def build_specs(task, n, n_prompts):
    """Same sampling as compute_extended_nshot_sampled.build_task_specs (seeds identical)."""
    data = json.load(open(TASK_ROOT / f"{task}.json"))
    N = len(data)
    specs = []
    for i in range(n_prompts):
        seed = int(hashlib.sha256(f"{task}|{n}|{i}".encode()).hexdigest()[:12], 16)
        rng = random.Random(seed)
        idx = rng.sample(range(N), n + 1)
        demos = [data[j] for j in idx[:n]]
        qt = {"input": data[idx[n]]["input"], "output": data[idx[n]]["output"]}
        gold = [str(o).strip() for o in (qt["output"] if isinstance(qt["output"], list) else [qt["output"]])]
        specs.append({"task": task, "n": n, "i": i, "sample_seed": seed,
                      "demos": demos, "qt": qt, "query": str(qt["input"]), "gold": gold})
    return specs


def main():
    args = parse_args()
    import torch
    from utils.model_utils import load_gpt_model_and_tokenizer

    tasks = pool_tasks()[args.shard_idx::args.shard_n]
    if args.limit_tasks:
        tasks = tasks[:args.limit_tasks]
    for fmt in args.formats:
        (OUT_ROOT / fmt).mkdir(parents=True, exist_ok=True)
    todo = [(fmt, t) for fmt in args.formats for t in tasks
            if args.overwrite or not (OUT_ROOT / fmt / f"{t}.json").exists()]
    n_done = len(args.formats) * len(tasks) - len(todo)
    print(f"shard {args.shard_idx}/{args.shard_n}: {len(tasks)} tasks x {len(args.formats)} formats, "
          f"{len(todo)} (format, task) cells to run ({n_done} already done)", flush=True)
    if not todo:
        return

    model, tokenizer, _ = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model = model.to(torch.bfloat16)
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert_template(tokenizer)

    specs_by_task = {t: build_specs(t, args.n, args.n_prompts) for t in dict.fromkeys(t for _, t in todo)}
    flat = []
    print("tokenizing", flush=True)
    for fmt, t in todo:
        for s in specs_by_task[t]:
            prompt = render_prompt(fmt, tokenizer, s["demos"], s["qt"])
            flat.append({**{k: s[k] for k in ("task", "n", "i", "sample_seed", "query", "gold")},
                         "format": fmt, "prompt": prompt,
                         "ids": tokenizer(prompt, truncation=False).input_ids})
    flat.sort(key=lambda s: len(s["ids"]))

    batches, cur, cur_max = [], [], 0
    for s in flat:
        L = len(s["ids"])
        if cur and (max(cur_max, L) * (len(cur) + 1) > args.batch_tokens or len(cur) >= args.max_batch_size):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(s)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    print(f"{len(batches)} batches", flush=True)

    remaining = {(fmt, t): args.n_prompts for fmt, t in todo}
    records = {(fmt, t): [] for fmt, t in todo}
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(batches):
            enc = tokenizer([s["prompt"] for s in batch], return_tensors="pt", padding=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            gen_seed = args.seed * 1_000_003 + args.shard_idx * 10_007 + bi
            torch.manual_seed(gen_seed)
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id, use_cache=True)
            cont = out[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(cont, skip_special_tokens=True)
            for s, text in zip(batch, texts):
                pred = text.split("\n")[0].strip()
                key = (s["format"], s["task"])
                rec = {k: s[k] for k in ("task", "format", "n", "i", "sample_seed", "query", "gold")}
                rec.update(generation=text, pred=pred, gen_seed=gen_seed, match=pred in s["gold"])
                records[key].append(rec)
                remaining[key] -= 1
                if remaining[key] == 0:
                    rs = sorted(records.pop(key), key=lambda r: r["i"])
                    json.dump(rs, open(OUT_ROOT / key[0] / f"{key[1]}.json", "w"))
                    acc = sum(r["match"] for r in rs) / args.n_prompts
                    print(f"  [{key[0]}/{key[1]}] acc {acc:.2f}", flush=True)
            if bi % 20 == 0:
                print(f"batch {bi}/{len(batches)} ({time.time()-t0:.0f}s)", flush=True)
    assert not records, f"incomplete cells: {list(records)}"
    print(f"shard {args.shard_idx} DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
