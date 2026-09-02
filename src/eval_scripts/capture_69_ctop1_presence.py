#!/usr/bin/env python
"""Presence capture for the decomposed read feature: carrier c and task-unique v1 (GPU).

USER DECISION 2026-09-02: Claim 5 presence maps measured against the two objects of the
L5-7 read-feature decomposition — the task-agnostic carrier c (unit) and the task's own
top-1 direction v1 (unit, bankA/L5to7_top1_bases) — no counterfactual arm.

Everything else identical to capture_69_fv_location.py / capture_69_readdir_location.py:
clean 10-shot forward passes over the 150 fixed train prompts, block outputs 0..27,
per-token cos(z_l, dir) averaged into the 32 structural columns.

Writes ARTIFACTS_ROOT/69_task_run/ctop1_presence/<task>.npz with
  cos_carrier (28, 32), cos_v1 (28, 32), dot_carrier, dot_v1, n_prompts, columns, group.
Fan out with --tasks.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from baukit import TraceDict  # noqa: E402
from src.eval_scripts.capture_69_fv_location import COLUMNS, token_columns  # noqa: E402
from src.sandbox.isolation_upper_bound.run_task import (  # noqa: E402
    auto_batch, load_records, record_to_prompt_data)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: E402
from src.utils.paths import ARTIFACTS_ROOT  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--vectors_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA"
                   / "carrier_plus_top1_vectors.pt")
    p.add_argument("--bases_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" / "bankA"
                   / "L5to7_top1_bases.pt")
    p.add_argument("--out_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "ctop1_presence")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capture_batch", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    model, tokenizer, cfg = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)
    n_layers = cfg["n_layers"]
    args.out_root.mkdir(parents=True, exist_ok=True)
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group_of = {t: "train" for t in split["train_tasks"]}
    group_of.update({t: "heldout" for t in split["heldout_tasks"]})

    c = torch.load(args.vectors_path, map_location="cpu", weights_only=False)["carrier"].double()
    c_hat = (c / c.norm()).float().to(model.device)
    bases = torch.load(args.bases_path, map_location="cpu", weights_only=False)["tasks"]

    for task in args.tasks:
        out = args.out_root / f"{task}.npz"
        if out.exists():
            print(f"[{task}] exists, skip", flush=True)
            continue
        v1 = bases[task]["V"][0].double()
        assert abs(v1.norm().item() - 1) < 1e-4
        v1 = v1.float().to(model.device)
        dirs = {"carrier": c_hat, "v1": v1}

        recs = load_records(args, task, "train_prompts")
        assert len(recs) == 150
        toks, cols = [], []
        for r in recs:
            ids, cc = token_columns(tokenizer, record_to_prompt_data(r, cfg))
            toks.append(ids)
            cols.append(cc)

        sums = {f"{k}_{d}": np.zeros((n_layers, len(COLUMNS)), dtype=np.float64)
                for d in dirs for k in ("dot", "cos")}
        n_seen = 0
        old_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        try:
            max_tok = max(len(t) for t in toks)
            bsz = auto_batch(max_tok, 4000, args.capture_batch)
            for start in range(0, len(recs), bsz):
                bt, bc = toks[start:start + bsz], cols[start:start + bsz]
                lens = [len(t) for t in bt]
                pad_to = max(lens)
                input_ids = torch.full((len(bt), pad_to), tokenizer.eos_token_id, dtype=torch.long)
                attn = torch.zeros((len(bt), pad_to), dtype=torch.long)
                for i, t in enumerate(bt):
                    input_ids[i, :len(t)] = torch.tensor(t)
                    attn[i, :len(t)] = 1
                input_ids, attn = input_ids.to(model.device), attn.to(model.device)
                with TraceDict(model, layers=cfg["layer_hook_names"], retain_output=True) as td:
                    model(input_ids=input_ids, attention_mask=attn)
                col_idx = [[[j for j in range(lens[i]) if bc[i][j] == cidx] for cidx in range(len(COLUMNS))]
                           for i in range(len(bt))]
                for l, name in enumerate(cfg["layer_hook_names"]):
                    h = td[name].output
                    if isinstance(h, tuple):
                        h = h[0]
                    h = h.float()
                    hn = h.norm(dim=-1).clamp_min(1e-8)
                    for dname, dvec in dirs.items():
                        dots = h @ dvec
                        coss = (dots / hn).cpu().numpy()
                        dots = dots.cpu().numpy()
                        for i in range(len(bt)):
                            for cidx in range(len(COLUMNS)):
                                sums[f"dot_{dname}"][l, cidx] += dots[i, col_idx[i][cidx]].mean()
                                sums[f"cos_{dname}"][l, cidx] += coss[i, col_idx[i][cidx]].mean()
                n_seen += len(bt)
                print(f"[{task}] {n_seen}/150", flush=True)
        finally:
            tokenizer.padding_side = old_side

        np.savez(out, **{k: (v / n_seen).astype(np.float32) for k, v in sums.items()},
                 n_prompts=n_seen, columns=np.array(COLUMNS), group=group_of[task],
                 dirs="carrier=unit 69-task mean of L5-7 read features; v1=bankA L5to7_top1")
        print(f"[{task}] done", flush=True)


if __name__ == "__main__":
    main()
