#!/usr/bin/env python
"""Stage 0 for the payload-subspace ablation: per-(site, edit layer) mean residual activations
across the 20 varicl train tasks (1-shot Stream W prompt scheme).

For every train task in the 29-task split, build the Stream W one-shot prompts (imported
build_prompts/make_chunks — seed 42, up to 130 train + 40 test queries, fewer for small tasks)
and run CLEAN forwards with hidden states. At each of the 3 Stream W site tokens (cue1 =
pre_label_token, target1 = last_label_token, final_cue = last_prompt_token) record the mean
residual activation at every edit-layer output b in 0..27 (= hidden_states entry b+1; the
embedding entry 0 is not an edit layer).

Saves artifacts/payload_subspace_ablation/train_task_site_means.pt:
  per_task_means (n_tasks, 3, 28, 4096) fp32, counts, tasks, site_roles,
  grand_mean (3, 28, 4096) = UNWEIGHTED mean over tasks (equal task weighting — user-gated:
  small tasks contribute fewer prompts, pooling would overweight the big ones), + metadata.
The grand mean is the mean-clamp target population for ablate_oneshot_payload_subspace_logprob.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.eval_scripts.exploratory.ablate_oneshot_preimage_logprob import (
    N_EDIT_LAYERS,
    SITE_ROLES,
    build_prompts,
    git_commit_hash,
    make_chunks,
)
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.paths import ARTIFACTS_ROOT
from utils.prompt_utils import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_split_path", type=Path,
                   default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--task_split_key", type=str, default="train_tasks")
    p.add_argument("--tasks", nargs="+", default=None, help="Optional explicit override/subset.")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--n_shots", type=int, default=1)
    p.add_argument("--max_train_prompts", type=int, default=130)
    p.add_argument("--max_test_prompts", type=int, default=40)
    p.add_argument("--max_prompts", type=int, default=None, help="Smoke cap on prompts/task.")
    p.add_argument("--batch_size", type=int, default=170)
    p.add_argument("--prefixes", type=json.loads,
                   default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads,
                   default={"input": "\n", "output": "\n\n", "instructions": ""})
    p.add_argument("--output_path", type=Path,
                   default=ARTIFACTS_ROOT / "payload_subspace_ablation" / "train_task_site_means.pt")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_path} exists. Pass --overwrite to recompute.")

    if args.tasks is not None:
        tasks = list(args.tasks)
    else:
        split = json.loads(args.task_split_path.read_text())
        tasks = list(split[args.task_split_key])
    print(f"Capturing site means for {len(tasks)} tasks: {tasks}", flush=True)

    set_seed(args.seed)
    torch.set_grad_enabled(False)
    print("Loading model...", flush=True)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = args.device
    resid_dim = model_config["resid_dim"]

    per_task_means = torch.zeros(len(tasks), len(SITE_ROLES), N_EDIT_LAYERS, resid_dim,
                                 dtype=torch.float32)
    counts = []
    for ti, task in enumerate(tasks):
        t0 = time.time()
        dataset = load_dataset(task, root_data_dir=args.root_data_dir,
                               test_size=args.test_split, seed=args.seed)
        prompts = build_prompts(task, dataset, tokenizer, model_config, args)
        chunks = make_chunks(prompts, tokenizer, args.batch_size, device)
        n = len(prompts)

        st = prompts[0]["site_texts"]
        for role in ("pre_label_token", "last_prompt_token"):
            assert st[role].strip() == ":", f"{task}: {role} decodes to {st[role]!r}, expected ':'"

        acc = torch.zeros(len(SITE_ROLES), N_EDIT_LAYERS, resid_dim, dtype=torch.float64,
                          device=device)
        for ch in chunks:
            out = model.transformer(input_ids=ch["input_ids"],
                                    attention_mask=ch["attention_mask"],
                                    output_hidden_states=True)
            hs = out.hidden_states  # 29 x (B, seq, 4096); entry b+1 = output of h.b
            rows_idx = torch.arange(ch["n"], device=device)
            for si in range(len(SITE_ROLES)):
                pos_vec = ch["pos"][:, si]
                for b in range(N_EDIT_LAYERS):
                    acc[si, b] += hs[b + 1][rows_idx, pos_vec, :].double().sum(dim=0)
        per_task_means[ti] = (acc / n).float().cpu()
        counts.append(n)
        print(f"[{ti + 1}/{len(tasks)}] {task}: n={n} in {time.time() - t0:.0f}s", flush=True)

    grand_mean = per_task_means.mean(dim=0)  # equal task weighting (user-gated)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "per_task_means": per_task_means,
            "grand_mean": grand_mean,
            "tasks": tasks,
            "counts": counts,
            "site_roles": list(SITE_ROLES),
            "edit_layer_mapping": "index b in 0..27 = output of transformer.h.b "
                                  "(hidden_states entry b+1)",
            "weighting": "grand_mean = unweighted mean of per-task means (equal task weight)",
            "prompt_scheme": {k: (str(v) if isinstance(v, Path) else v)
                              for k, v in vars(args).items()},
            "model_name": args.model_name,
            "git_commit": git_commit_hash(),
        },
        args.output_path,
    )
    print(f"Wrote {args.output_path} (per_task_means {tuple(per_task_means.shape)})")
    print("DONE")


if __name__ == "__main__":
    main()
