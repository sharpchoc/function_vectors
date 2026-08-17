#!/usr/bin/env python
"""Per-task mean RESIDUAL-STREAM activation at the last demo label token (all 28 layers).

Baseline companion to capture_label_head_means.py: the FV paper's "average hidden state"
control (isolation_methods_levers.md Lever 1c), re-sited from the cue token to the label
token so it can be injected at the ' _' slot of the 1-shot scaffold.

Same prompts, same position, same gates as the head-mean capture; stores the block OUTPUT
hidden state per layer (what an additive steer at that layer would modify).

Output: artifacts/69_task_run/label_resid_means/<task>.pt
  {resid_means (28, 4096) fp32, n_prompts, label_idx (150,)}
"""
import argparse
import json
import sys
from pathlib import Path

import torch

# local bootstrap for in-repo runs; a PYTHONPATH-supplied repo also works (staged copies)
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from baukit import TraceDict
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
try:
    from src.sandbox.ext_steerability.capture_label_head_means import prompt_and_label_idx
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from capture_label_head_means import prompt_and_label_idx


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])[args.shard_idx::args.shard_n]
    print(f"{len(tasks)} tasks on this shard", flush=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    n_layers, resid = model_config["n_layers"], model_config["resid_dim"]
    tokenizer.padding_side = "right"
    layer_names = model_config["layer_hook_names"]
    assert len(layer_names) == n_layers

    for task in tasks:
        out = args.out_root / f"{task}.pt"
        if out.exists():
            print(f"{task}: exists, skip", flush=True)
            continue
        recs = json.load(open(args.prompts_root / task / "train_prompts.json"))
        assert len(recs) == 150
        built = [prompt_and_label_idx(r, tokenizer) for r in recs]
        for (ps, li, ci), rec in zip(built, recs):
            ids = tokenizer(ps).input_ids
            gold_last = tokenizer(" " + str(rec["demos"][9]["output"]).strip()).input_ids[-1]
            assert ids[li] == gold_last and li != ci, f"{task}: label idx gate failed at {li}"

        resid_sum = torch.zeros(n_layers, resid, dtype=torch.float64)
        n_seen = 0
        for start in range(0, len(built), args.batch_size):
            chunk = built[start:start + args.batch_size]
            sentences = [c[0] for c in chunk]
            label_idx = torch.tensor([c[1] for c in chunk], device=model.device)
            inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
            bidx = torch.arange(len(chunk), device=model.device)
            with torch.no_grad(), TraceDict(model, layers=layer_names,
                                            retain_input=False, retain_output=True) as td:
                model(**inputs)
            for li_, lname in enumerate(layer_names):
                outp = td[lname].output
                outp = outp[0] if isinstance(outp, tuple) else outp
                resid_sum[li_] += outp[bidx, label_idx].double().sum(dim=0).cpu()
            n_seen += len(chunk)
        assert n_seen == 150
        rm = (resid_sum / n_seen).float()
        torch.save({"task": task, "resid_means": rm, "n_prompts": n_seen,
                    "label_idx": torch.tensor([c[1] for c in built]),
                    "site": "block OUTPUT hidden state at the last token of the 10th demo "
                            "label (clean 10-shot prompts)"}, out)
        print(f"{task}: captured {n_seen} prompts | ||resid_mean|| L7={rm[7].norm():.1f} "
              f"L0={rm[0].norm():.1f} L27={rm[27].norm():.1f}", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
