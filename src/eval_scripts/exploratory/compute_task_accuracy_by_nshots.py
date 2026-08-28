"""
Baseline GPT-J accuracy vs number of ICL examples (no interventions; GPU).

For each task and each n_shots in 0..10, runs n_shot_eval_no_intervention over the full test
split and records top-1/2/3 accuracy. Uses the same Q:/A: prompt template as the 10-shot strip
study (steer_tenshot_strip_cos_heatmap.py) so the curves are directly comparable with the
strip-figure summaries. Resumable: one JSON per (task, n_shots), skipped if it already exists;
merge/plot with plot_task_accuracy_by_nshots.py (CPU-only).

Output (TRACKED): results/exploratory/general/task_accuracies/by_nshots/{task}_n{n}.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import load_dataset
from utils.eval_utils import n_shot_eval_no_intervention
from utils.paths import GENERAL_DIR, REPO_ROOT

TASKS = ["antonym", "synonym", "next_number_digits", "prev_number_digits"]
# same template as the strip study
PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--tasks", type=str, nargs="+", default=TASKS)
    p.add_argument("--n_shots", type=int, nargs="+", default=list(range(0, 11)))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--root_data_dir", type=str, default=str(REPO_ROOT / "dataset_files"))
    p.add_argument("--output_root", type=Path, default=GENERAL_DIR / "task_accuracies" / "by_nshots")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    todo = [(t, n) for t in args.tasks for n in args.n_shots
            if not (args.output_root / f"{t}_n{n}.json").exists()]
    print(f"{len(todo)} (task, n_shots) cells to compute", flush=True)
    if not todo:
        return

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)

    for task in args.tasks:
        dataset = load_dataset(task, root_data_dir=args.root_data_dir, seed=args.seed)
        for n in args.n_shots:
            out_path = args.output_root / f"{task}_n{n}.json"
            if out_path.exists():
                print(f"skip {task} n={n} (exists)", flush=True)
                continue
            set_seed(args.seed + n)
            # no_grad: eval_utils only wraps the generate_str path; the batched logit
            # path would otherwise build the autograd graph and OOM at batch_size 32
            with torch.no_grad():
                res = n_shot_eval_no_intervention(dataset, n, model, model_config, tokenizer,
                                                  compute_ppl=False, prefixes=PREFIXES,
                                                  separators=SEPARATORS, batch_size=args.batch_size)
            rec = {"task": task, "n_shots": n, "seed": args.seed,
                   "n_test": len(res["clean_rank_list"]),
                   "topk": {str(k): acc for k, acc in res["clean_topk"]}}
            out_path.write_text(json.dumps(rec, indent=2))
            print(f"{task} n={n}: top1={rec['topk']['1']:.3f} "
                  f"top3={rec['topk']['3']:.3f} (n_test={rec['n_test']})", flush=True)


if __name__ == "__main__":
    main()
