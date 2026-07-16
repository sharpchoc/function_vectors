#!/usr/bin/env python
"""GPT-4-judged top-1 accuracy vs n_shots for open-ended tasks (antonym/synonym).

Companion to compute_task_accuracy_by_nshots.py, whose gold-first-token scoring undercounts
open-ended tasks (many valid answers; see the 1-shot judge study). Three resumable stages:

  generate (GPU): rebuild the EXACT prompts of the gold-scored run — same seed protocol
      (set_seed(seed + n) then per-item np.random.choice demo draws in test order, same Q:/A:
      template) — and record each prompt's top-1 next token (plus top-5 and the gold token's
      rank, so the gold-scored curve can be re-derived from the same records).
  judge (CPU): send (query, top-1 token) pairs to the OpenAI judge with the JUDGE_SYSTEMS
      prompts from judge_oneshot_paired.py (same-word/inflection = false). The top-1 token is
      judged AS-IS after whitespace trimming — word fragments are expected to be judged false.
  summarize (CPU): per-(task, n) judged + gold top-1 accuracy table; consistency-check the
      re-derived gold top-1 against results/general/task_accuracies/by_nshots/.

Output (TRACKED): results/general/task_accuracies/by_nshots_judged/{task}_n{n}.json
  (per-prompt records: query, gold, top-5 tokens, gold rank, judge verdict) + summary.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import GENERAL_DIR, REPO_ROOT
from eval_scripts.judge_oneshot_paired import JUDGE_SYSTEMS, _judge_batch

PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}
TASKS = ["antonym", "synonym"]


def parse_args():
    p = argparse.ArgumentParser(description="Judged top-1 accuracy vs n_shots (antonym/synonym).")
    p.add_argument("--stage", choices=["generate", "judge", "summarize"], required=True)
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--n_shots", type=int, nargs="+", default=list(range(0, 11)))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--judge_model", type=str, default="gpt-4.1")
    p.add_argument("--judge_batch_size", type=int, default=50)
    p.add_argument("--root_data_dir", type=str, default=str(REPO_ROOT / "dataset_files"))
    p.add_argument("--output_root", type=Path,
                   default=GENERAL_DIR / "task_accuracies" / "by_nshots_judged")
    p.add_argument("--gold_root", type=Path,
                   default=GENERAL_DIR / "task_accuracies" / "by_nshots",
                   help="Gold-scored run to consistency-check against in summarize.")
    return p.parse_args()


def out_path(args, task, n):
    return args.output_root / f"{task}_n{n}.json"


def stage_generate(args):
    import torch
    from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
    from utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
    from utils.eval_utils import get_answer_id

    todo = [(t, n) for t in args.tasks for n in args.n_shots
            if not out_path(args, t, n).exists()]
    print(f"{len(todo)} (task, n) cells to generate", flush=True)
    if not todo:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    prepend_bos = False if model_config["prepend_bos"] else True

    for task in args.tasks:
        dataset = load_dataset(task, root_data_dir=args.root_data_dir, seed=args.seed)
        for n in args.n_shots:
            path = out_path(args, task, n)
            if path.exists():
                print(f"skip {task} n={n} (exists)", flush=True)
                continue
            set_seed(args.seed + n)  # same protocol as compute_task_accuracy_by_nshots.py
            items = []
            for j in range(len(dataset["test"])):  # j order => same RNG draws as gold run
                if n == 0:
                    word_pairs = {"input": [], "output": []}
                else:
                    word_pairs = dataset["train"][np.random.choice(len(dataset["train"]), n, replace=False)]
                pd = word_pairs_to_prompt_data(word_pairs, query_target_pair=dataset["test"][j],
                                               prepend_bos_token=prepend_bos,
                                               prefixes=PREFIXES, separators=SEPARATORS)
                target = pd["query_target"]["output"]
                target = target[0] if isinstance(target, list) else target
                sent = create_prompt(pd)
                items.append((sent, pd["query_target"]["input"], target,
                              get_answer_id(sent, target, tokenizer)[0]))

            records = []
            for bstart in range(0, len(items), args.batch_size):
                batch = items[bstart:bstart + args.batch_size]
                enc = tokenizer([x[0] for x in batch], return_tensors="pt", padding=True).to(model.device)
                with torch.no_grad():
                    logits = model(**enc).logits
                last = enc.attention_mask.sum(dim=1) - 1
                for k, (sent, query, target, gold_id) in enumerate(batch):
                    row = logits[k, last[k]]
                    top5 = torch.topk(row, 5).indices.tolist()
                    gold_rank = int((row > row[gold_id]).sum())
                    records.append({
                        "query": query, "gold": target,
                        "top5_tokens": [tokenizer.decode([t]) for t in top5],
                        "top1_answer": tokenizer.decode([top5[0]]).strip(),
                        "gold_rank": gold_rank,
                    })
            with open(path, "w") as f:
                json.dump({"task": task, "n_shots": n, "seed": args.seed,
                           "model_name": args.model_name, "records": records}, f, indent=1)
            gold_top1 = sum(r["gold_rank"] == 0 for r in records) / len(records)
            print(f"{task} n={n}: {len(records)} prompts | rederived gold top1={gold_top1:.3f}",
                  flush=True)


def stage_judge(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    for task in args.tasks:
        system = JUDGE_SYSTEMS[task]
        for n in args.n_shots:
            path = out_path(args, task, n)
            data = json.loads(path.read_text())
            records = data["records"]
            if all("judge_correct" in r for r in records):
                print(f"skip {task} n={n} (judged)", flush=True)
                continue
            pairs = [{"input": r["query"].strip(), "answer": r["top1_answer"]} for r in records]
            verdicts = []
            for i in range(0, len(pairs), args.judge_batch_size):
                verdicts.extend(_judge_batch(pairs[i:i + args.judge_batch_size], system,
                                             args.judge_model, api_key))
            for r, v in zip(records, verdicts):
                r["judge_correct"] = bool(v["correct"])
            data["judge_model"] = args.judge_model
            path.write_text(json.dumps(data, indent=1))
            acc = sum(r["judge_correct"] for r in records) / len(records)
            print(f"{task} n={n}: judged top1={acc:.3f}", flush=True)


def stage_summarize(args):
    summary = {}
    for task in args.tasks:
        summary[task] = {}
        for n in args.n_shots:
            data = json.loads(out_path(args, task, n).read_text())
            records = data["records"]
            gold_top1 = sum(r["gold_rank"] == 0 for r in records) / len(records)
            judged = sum(r.get("judge_correct", False) for r in records) / len(records)
            row = {"n_test": len(records), "gold_top1": gold_top1, "judged_top1": judged}
            ref_path = args.gold_root / f"{task}_n{n}.json"
            if ref_path.exists():
                ref = json.loads(ref_path.read_text())["topk"]["1"]
                row["gold_top1_reference"] = ref
                row["matches_reference"] = abs(ref - gold_top1) < 1e-9
            summary[task][str(n)] = row
            print(f"{task} n={n}: gold={gold_top1:.3f} judged={judged:.3f} "
                  f"(ref match: {row.get('matches_reference')})", flush=True)
    with open(args.output_root / "summary.json", "w") as f:
        json.dump({"judge_model": args.judge_model,
                   "note": "top-1 token judged as-is (whitespace-trimmed); fragments count false",
                   "tasks": summary}, f, indent=1)
    print(f"Wrote {args.output_root / 'summary.json'}")


def main():
    args = parse_args()
    {"generate": stage_generate, "judge": stage_judge, "summarize": stage_summarize}[args.stage](args)


if __name__ == "__main__":
    main()
