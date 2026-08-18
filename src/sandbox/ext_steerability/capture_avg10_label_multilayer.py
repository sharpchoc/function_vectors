#!/usr/bin/env python
"""Per-prompt MEAN label-token activation, layers 5..15 (69 tasks).

X-side capture for the layer sweep of the label-token -> FV ridge: for each clean 10-shot
prompt, the block OUTPUT hidden state at the last token of each of the 10 demo labels is
captured at every layer in --layers and AVERAGED over the 10 positions (the user narrowed
the sweep to the avg-of-10 X variant, so only the mean is stored).

Position indexing identical to capture_all10_label_L6.py (BOS-free prompts, per-n last
label token via get_token_meta_labels, hard gates in prompt_and_label_idxs).

Output: artifacts/69_task_run/label_avg10_L5-15_acts/<task>.pt
  {acts (150, n_layers, 4096) fp16 = mean over the 10 label positions, layers, prompt_index}
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
    from src.sandbox.ext_steerability.capture_all10_label_L6 import prompt_and_label_idxs
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from capture_all10_label_L6 import prompt_and_label_idxs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_avg10_L5-15_acts")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--layers", type=int, nargs="+", default=list(range(5, 16)))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])[args.shard_idx::args.shard_n]
    print(f"{len(tasks)} tasks on this shard, layers {args.layers}", flush=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    tokenizer.padding_side = "right"
    lnames = [model_config["layer_hook_names"][l] for l in args.layers]

    for task in tasks:
        out = args.out_root / f"{task}.pt"
        if out.exists():
            print(f"{task}: exists, skip", flush=True)
            continue
        recs = json.load(open(args.prompts_root / task / "train_prompts.json"))
        assert len(recs) == 150
        built = [prompt_and_label_idxs(r, tokenizer) for r in recs]
        acts = torch.zeros(150, len(args.layers), 4096, dtype=torch.float16)
        for start in range(0, len(built), args.batch_size):
            chunk = built[start:start + args.batch_size]
            sentences = [c[0] for c in chunk]
            idxs = torch.tensor([c[1] for c in chunk], device=model.device)  # (B, 10)
            inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad(), TraceDict(model, layers=lnames, retain_output=True) as td:
                model(**inputs)
            bidx = torch.arange(len(chunk), device=model.device).unsqueeze(1)
            for li, ln in enumerate(lnames):
                outp = td[ln].output
                outp = outp[0] if isinstance(outp, tuple) else outp
                # (B, 10, 4096) at the label positions -> mean over the 10 demos
                acts[start:start + len(chunk), li] = \
                    outp[bidx, idxs].mean(dim=1).half().cpu()
        torch.save({"task": task, "acts": acts, "layers": list(args.layers),
                    "prompt_index": [r["prompt_index"] for r in recs],
                    "site": "mean over the 10 last-label-token block outputs"}, out)
        print(f"{task}: captured {tuple(acts.shape)}", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
