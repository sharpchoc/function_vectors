#!/usr/bin/env python
"""SANDBOX: clean 10-shot competence per task on the
30 paired test queries (no steering). Rebuilds each test_sametask_shuffled10 record with
its demos' CORRECT outputs, scores full-label teacher-forced accuracy. Predictor for the
steerability-hypotheses analysis (H1: model competence)."""
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.sandbox.isolation_upper_bound.run_task import (
    eval_points_fixed_v,
    record_to_point,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.paths import ARTIFACTS_ROOT

OUT = ARTIFACTS_ROOT / "sandbox" / "isolation_upper_bound" / "clean10_competence.json"


def main():
    set_seed(42)
    split = json.load(open(REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json"))
    tasks = split["train_tasks"] + split["test_tasks"]
    model, tokenizer, model_config = load_gpt_model_and_tokenizer("EleutherAI/gpt-j-6b")
    model = model.to(torch.bfloat16).eval()
    torch.set_grad_enabled(False)

    res = {}
    for t in tasks:
        recs = json.load(open(REPO_ROOT / "dataset_files" / "isolation_prompts" / t /
                              "test_sametask_shuffled10.json"))
        clean = []
        for r in recs:
            demos = [{"input": d["input"], "output": o}
                     for d, o in zip(r["demos"], r["demo_correct_outputs"])]
            clean.append({"task": t, "n_shots": 10, "demos": demos,
                          "query": r["query"], "prompt_index": r["prompt_index"]})
        points = [record_to_point(r, tokenizer, model_config) for r in clean]
        acc = eval_points_fixed_v(model, model_config, tokenizer, points, None, 9)
        res[t] = acc
        print(f"{t:28s} clean10={acc:.3f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
