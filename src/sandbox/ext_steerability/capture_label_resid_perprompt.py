#!/usr/bin/env python
"""Per-PROMPT bank-(a) residuals at the final-demo target token, all 28 layers.

Bank-(a) migration (USER DECISION 2026-09-01): the Claim-6 ridge layer sweep needs
per-prompt X rows; label_resid_means stores only the task mean. Same prompts, same
position, same gates as capture_label_resid_means.py, but the per-prompt block-OUTPUT
hidden states are stored (fp16).

Output: artifacts/69_task_run/label_resid_perprompt/<task>.pt
  {acts (150, 28, 4096) fp16, label_idx (150,), prompt_index}
Sanity: acts.mean(0) must match label_resid_means resid_means to fp16 tolerance.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from baukit import TraceDict  # noqa: E402
from src.utils.model_utils import load_gpt_model_and_tokenizer  # noqa: E402
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT  # noqa: E402
try:
    from src.sandbox.ext_steerability.capture_label_head_means import prompt_and_label_idx
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from capture_label_head_means import prompt_and_label_idx


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_resid_perprompt")
    p.add_argument("--means_root", type=Path,
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
        acts = torch.zeros(150, n_layers, resid, dtype=torch.float16)
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
                acts[n_seen:n_seen + len(chunk), li_] = \
                    outp[bidx, label_idx].half().cpu()
            n_seen += len(chunk)
        assert n_seen == 150
        rm = torch.load(args.means_root / f"{task}.pt", map_location="cpu",
                        weights_only=False)["resid_means"]
        # fp16 storage rounds large-magnitude outlier dims by up to ~0.06 abs; compare
        # with a relative-to-norm tolerance instead of a tight absolute one
        gap = (acts.float().mean(0) - rm).abs().max()
        rel = float(gap / rm.norm(dim=1).max())
        assert gap < 0.5 and rel < 1e-3, \
            f"{task}: per-prompt mean drifts from stored mean (abs {gap:.4f}, rel {rel:.2e})"
        torch.save({"task": task, "acts": acts,
                    "label_idx": torch.tensor([c[1] for c in built]),
                    "site": "block OUTPUT at last token of the 10th demo target, "
                            "per prompt, clean 10-shot prompts"}, out)
        print(f"{task}: captured (mean-gap {gap:.4f})", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
