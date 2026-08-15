#!/usr/bin/env python
"""SANDBOX: head-mean captures for the extended_steerable_90 tasks.

Thin wrapper over isolation_upper_bound.run_task.stage_capture: per task, cue-token
per-head means (28x16x256) + per-layer residual means over the task's train_prompts
(fixed 10-shot, correct labels). Linearity gate intact. Fan out with --tasks.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.sandbox.isolation_upper_bound.run_task import stage_capture
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.paths import ARTIFACTS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path, default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capture_batch", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)
    for task in args.tasks:
        (args.out_root / task).mkdir(parents=True, exist_ok=True)
        stage_capture(args, task, model, model_config, tokenizer)
    print("CAPTURES DONE")


if __name__ == "__main__":
    main()
