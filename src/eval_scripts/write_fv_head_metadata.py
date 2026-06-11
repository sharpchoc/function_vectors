#!/usr/bin/env python
"""Write a small selected_heads.json into each task's function-vector folder.

For a model folder laid out as results/function_vectors/<model>/<method>/<task>/, this
records which attention heads built each FV:
  * task_specific  -> heads are unique per task; read from that task's *_function_vector.pt
  * train_selected / train_test_selected -> heads are shared; read from the method's heads.pt
    (top --n_top_heads).

Idempotent: rerun after building new FVs or adding a new model.

Example:
  python src/eval_scripts/write_fv_head_metadata.py \
    --model_root results/function_vectors/gpt-j --n_top_heads 10
"""
import argparse
import json
from pathlib import Path

import torch

POOL_DESC = {
    "task_specific": "this task only (per-task CIE)",
    "train_selected": "20 train tasks (pooled CIE)",
    "train_test_selected": "all 29 train+test tasks (pooled CIE)",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_root", type=Path, default=Path("results/function_vectors/gpt-j"),
                   help="Folder containing <method>/<task>/ subfolders.")
    p.add_argument("--methods", nargs="+", default=list(POOL_DESC),
                   help="Method subfolders to annotate.")
    p.add_argument("--n_top_heads", type=int, default=10,
                   help="Heads used per FV (for the shared multitask methods).")
    return p.parse_args()


def torch_load_trusted(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_heads(raw, n=None):
    heads = [[int(t[0]), int(t[1]), (round(float(t[2]), 6) if len(t) > 2 and t[2] is not None else None)]
             for t in raw]
    return heads[:n] if n is not None else heads


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    model = args.model_root.name
    written = 0
    for method in args.methods:
        mdir = args.model_root / method
        if not mdir.is_dir():
            print(f"skip {method}: not found")
            continue

        shared_heads = None
        if method != "task_specific":
            heads_pt = mdir / "heads.pt"
            if not heads_pt.exists():
                print(f"skip {method}: no heads.pt")
                continue
            shared_heads = to_heads(torch_load_trusted(heads_pt)["top_heads"], args.n_top_heads)

        for task_dir in sorted(d for d in mdir.iterdir() if d.is_dir()):
            task = task_dir.name
            if method == "task_specific":
                fv_pt = task_dir / f"{task}_function_vector.pt"
                if not fv_pt.exists():
                    continue
                heads = to_heads(torch_load_trusted(fv_pt)["top_heads"])
            else:
                heads = shared_heads

            write_json(task_dir / "selected_heads.json", {
                "model": model,
                "method": method,
                "task": task,
                "selection_pool": POOL_DESC.get(method, method),
                "n_top_heads": len(heads),
                "selected_heads": heads,
                "format": "[layer, head, mean_indirect_effect]",
            })
            written += 1
        print(f"{method}: wrote {len([d for d in mdir.iterdir() if d.is_dir()])} task metadata files")
    print(f"done: {written} selected_heads.json written under {args.model_root}")


if __name__ == "__main__":
    main()
