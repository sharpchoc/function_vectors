#!/usr/bin/env python
"""DOT-PRODUCT per-prompt read directions for the 69-task pool (canonical 37-head set).

Companion to compute_perprompt_read_dirs_37.py, which computes the COSINE-maximizing read
directions r = M^+ v / ||.|| (glossary Eq. 4-5). This script instead maximizes the dot
product <M r, v> over unit r, whose solution is the normalized transpose:

    r_dot^j_A = M^T v^j_A / ||M^T v^j_A||,   M = sum_{h in H37} W_O^h W_V^h.

The dot product rewards output magnitude along v (weights M's TOP singular directions),
while the cosine objective is indifferent to it (pseudo-inverse; weights the BOTTOM of the
spectrum). No SVD, no truncation variants — one well-conditioned object per prompt.

Same input contract and gates (a)/(b) as the cosine script; outputs go to a SEPARATE dir
(default artifacts/69_task_run/perprompt_dot_read_dirs/) so the two definitions can be
compared side by side. Per-task .pt keys mirror the cosine files where applicable:
  r (150,4096) fp32 unit rows, prenorm_MTv, cos_Mr_v (diagnostic), r_task, v fp16 copy,
  prompt_index, config.definition.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.utils.paths import ARTIFACTS_ROOT

HEAD_DIM, N_HEADS = 256, 16
DEFINITION = ("dot-product read direction r = M^T v / ||M^T v|| "
              "(argmax_{||r||=1} <M r, v>); NOT the glossary Eq.4-5 cosine/pseudo-inverse one")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fv_dir", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--out_dir", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_dot_read_dirs")
    p.add_argument("--selection_path", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43"
                   / "pooled_sparse" / "selection.json")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--ckpt", type=Path, default=None)
    return p.parse_args()


def torch_load(p, **kw):
    return torch.load(p, map_location="cpu", weights_only=False, **kw)


def main():
    args = parse_args()
    ckpt = args.ckpt or sorted(Path("/workspace/.cache/huggingface/hub/"
                                    "models--EleutherAI--gpt-j-6b/snapshots").glob("*/pytorch_model.bin"))[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"device": device, "dtype": "fp16-cast weights -> fp64 math",
               "fv_source": str(args.fv_dir), "selection_path": str(args.selection_path),
               "definition": DEFINITION}

    sel_flat = sorted(json.load(open(args.selection_path))["selected_flat"])
    heads = [(f // N_HEADS, f % N_HEADS) for f in sel_flat]
    assert len(heads) == 37
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})

    sd = torch_load(ckpt, mmap=True)
    wo_slices = []
    M = torch.zeros(4096, 4096, dtype=torch.float64)
    for l, h in heads:
        wo = sd[f"transformer.h.{l}.attn.out_proj.weight"][:, h * HEAD_DIM:(h + 1) * HEAD_DIM].clone().half()
        wv = sd[f"transformer.h.{l}.attn.v_proj.weight"][h * HEAD_DIM:(h + 1) * HEAD_DIM, :].clone().half()
        wo_slices.append(wo)
        M += wo.double() @ wv.double()
    WO = torch.stack(wo_slices).double().to(device)
    Md = M.to(device)

    tasks = sorted(p.stem for p in args.fv_dir.glob("*.pt"))
    assert len(tasks) == 69 and set(tasks) == set(group), \
        f"expected the 69 split tasks, got {len(tasks)} files"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_task_rows = []
    for task in tasks:
        d = torch_load(args.fv_dir / f"{task}.pt")
        assert sorted(int(x) for x in d["sel_flat"]) == sel_flat, \
            f"{task}: sel_flat mismatch vs canonical selection - HARD STOP, inform user"
        v = d["fv"].double().to(device)
        raw = d["raw"].double().to(device)

        # GATE (a): stored fv == sum_h W_O^h raw_h (fp16-storage tolerance).
        v_rebuilt = torch.einsum("hdk,nhk->nd", WO, raw)
        rel_a = ((v_rebuilt - v).norm(dim=1) / v.norm(dim=1)).max().item()
        assert rel_a < 2e-3, f"GATE(a) FAILED {task}: rel={rel_a:.3e} - HARD STOP, inform user"

        r_raw = v @ Md                                  # rows are M^T v_j
        norms = r_raw.norm(dim=1)
        r = r_raw / norms[:, None]
        Mr = r @ Md.T
        cos_v = torch.nn.functional.cosine_similarity(Mr, v, dim=1)
        v_task = v.mean(0, keepdim=True)
        rt = v_task @ Md

        out = {"task": task, "group": group[task], "heads": [list(x) for x in heads],
               "sel_flat": sel_flat, "prompt_index": d["prompt_index"],
               "v": d["fv"].clone(),
               "r": r.float().cpu(), "prenorm_MTv": norms.float().cpu(),
               "cos_Mr_v": cos_v.float().cpu(),
               "r_task": (rt / rt.norm()).float().cpu().squeeze(0),
               "config": {"definition": DEFINITION, "fv_source": str(args.fv_dir)}}
        torch.save(out, args.out_dir / f"{task}.pt")
        row = {"task": task, "group": group[task], "gate_a_rel": round(rel_a, 8),
               "median_cos_Mr_v": round(float(cos_v.median()), 4),
               "median_prenorm_MTv": round(float(norms.median()), 4)}
        per_task_rows.append(row)
        print(f"{task} [{group[task]}]: gate a={rel_a:.2e} | dot cos(Mr,v) med="
              f"{row['median_cos_Mr_v']} | ||M^T v|| med={row['median_prenorm_MTv']}", flush=True)

    summary["tasks"] = per_task_rows
    with open(args.out_dir / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {len(tasks)} task files to {args.out_dir}")


if __name__ == "__main__":
    main()
