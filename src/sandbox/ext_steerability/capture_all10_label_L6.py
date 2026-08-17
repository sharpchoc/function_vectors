#!/usr/bin/env python
"""Per-prompt L6 activations at the LAST token of EACH of the 10 demo labels (69 tasks).

Inputs for the label-token -> per-prompt-FV ridge study: for every clean 10-shot prompt
(dataset_files/isolation_prompts_ext/<task>/train_prompts.json) capture the block-6 OUTPUT
hidden state at the last token of demonstration n's label, for n = 1..10.

Conventions match the earlier label-site captures (capture_label_head_means /
capture_label_resid_means): prompts built with NO bos anywhere, indices located via
get_token_meta_labels ('demonstration_<n>_label_token', last such index per n), and a hard
gate that each captured index decodes to the final token of that demo's label.

Output: artifacts/69_task_run/label_all10_L6_acts/<task>.pt
  {acts (150, 10, 4096) fp16, prompt_index (150,), layer 6}
"""
import argparse
import json
import re
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
from src.utils.prompt_utils import get_token_meta_labels, word_pairs_to_prompt_data
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT

LAYER = 6
LAB_RE = re.compile(r"^demonstration_(\d+)_label_token$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_all10_L6_acts")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def prompt_and_label_idxs(rec, tokenizer):
    """(prompt_string, [last-token index of demo n's label for n=1..10])."""
    wp = {"input": [str(d["input"]) for d in rec["demos"]],
          "output": [str(d["output"]) for d in rec["demos"]]}
    qo = rec["query"]["output"]
    qo = [str(x) for x in qo] if isinstance(qo, list) else str(qo)
    q = {"input": str(rec["query"]["input"]), "output": qo}
    pd_ = word_pairs_to_prompt_data(wp, query_target_pair=q, prepend_bos_token=False,
                                    shuffle_labels=False)
    token_labels, prompt_string = get_token_meta_labels(pd_, tokenizer, query=q["input"],
                                                        prepend_bos=False)
    ids = tokenizer(prompt_string).input_ids
    assert len(ids) == len(token_labels)
    per_n = {}
    for i, _, lab in token_labels:
        m = LAB_RE.match(lab)
        if m:
            n = int(m.group(1))
            per_n[n] = max(per_n.get(n, -1), int(i))
    assert sorted(per_n) == list(range(1, 11)), f"labels found for demos {sorted(per_n)}"
    idxs = [per_n[n] for n in range(1, 11)]
    # gate: each index decodes to the final token of that demo's label
    for n in range(1, 11):
        gold_last = tokenizer(" " + str(rec["demos"][n - 1]["output"]).strip()).input_ids[-1]
        assert ids[idxs[n - 1]] == gold_last, f"demo {n}: label idx token mismatch"
    return prompt_string, idxs


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])[args.shard_idx::args.shard_n]
    print(f"{len(tasks)} tasks on this shard", flush=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    tokenizer.padding_side = "right"
    lname = model_config["layer_hook_names"][LAYER]

    for task in tasks:
        out = args.out_root / f"{task}.pt"
        if out.exists():
            print(f"{task}: exists, skip", flush=True)
            continue
        recs = json.load(open(args.prompts_root / task / "train_prompts.json"))
        assert len(recs) == 150
        built = [prompt_and_label_idxs(r, tokenizer) for r in recs]
        acts = torch.zeros(150, 10, 4096, dtype=torch.float16)
        for start in range(0, len(built), args.batch_size):
            chunk = built[start:start + args.batch_size]
            sentences = [c[0] for c in chunk]
            idxs = torch.tensor([c[1] for c in chunk], device=model.device)  # (B, 10)
            inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad(), TraceDict(model, layers=[lname],
                                            retain_output=True) as td:
                model(**inputs)
            outp = td[lname].output
            outp = outp[0] if isinstance(outp, tuple) else outp        # (B, seq, 4096)
            bidx = torch.arange(len(chunk), device=model.device).unsqueeze(1)
            acts[start:start + len(chunk)] = outp[bidx, idxs].half().cpu()
        torch.save({"task": task, "acts": acts, "layer": LAYER,
                    "prompt_index": [r["prompt_index"] for r in recs],
                    "site": "block-6 output at the LAST token of each demo label"}, out)
        print(f"{task}: captured (150, 10, 4096)", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
