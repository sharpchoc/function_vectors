#!/usr/bin/env python
"""FV-location capture in the 50D causal PC subspace (GPU) — low_dim_FV_presence variant.

Same design as capture_69_fv_location.py (clean 10-shot forward passes, block outputs
0..27, 32 structural token-position columns), except both the residual stream and the task
FV are first projected into the 50-dim causal subspace: the orthonormal span U (50, 4096)
of the c>0.8 PCs from the all-69-task sparse PC selection
(artifacts/69_task_run/pc_sparse_alltasks). Metrics per token:
  dot50 = (U z) . (U v_A) / ||U v_A||     raw amount of the projected-FV direction in the
                                          projected stream
  cos50 = cos(U z, U v_A)                 alignment inside the 50D space
Writes ARTIFACTS_ROOT/69_task_run/fv_location_50d/<task>.npz. Fan out with --tasks.
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
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--pc_sparse_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "pc_sparse_alltasks")
    p.add_argument("--out_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "fv_location_50d")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capture_batch", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    sel = json.load(open(args.pc_sparse_root / "selection.json"))
    basis = torch.load(args.pc_sparse_root / "pc_basis_uncentered.pt",
                       map_location="cpu", weights_only=False)
    U = basis["pcs"][sel["selected_pcs"]].double()          # (50, 4096), orthonormal rows
    gram_err = (U @ U.T - torch.eye(len(U))).abs().max().item()
    assert gram_err < 1e-4, f"selected PCs not orthonormal: max |UU^T - I| = {gram_err:.2e}"

    model, tokenizer, cfg = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)
    n_layers = cfg["n_layers"]
    U_dev = U.float().to(model.device)                      # (50, 4096)
    args.out_root.mkdir(parents=True, exist_ok=True)
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group_of = {t: "train" for t in split["train_tasks"]}
    group_of.update({t: "heldout" for t in split["heldout_tasks"]})

    for task in args.tasks:
        out = args.out_root / f"{task}.npz"
        if out.exists():
            print(f"[{task}] exists, skip", flush=True)
            continue

        pp = torch.load(args.fv_root / f"{task}.pt", map_location="cpu", weights_only=False)
        v = pp["fv"].double().mean(dim=0)
        v50 = (U @ v)                                        # (50,)
        proj_cos = (v50.norm() / v.norm()).item()            # ||P v|| / ||v||
        v50_hat = (v50 / v50.norm()).float().to(model.device)

        recs = load_records(args, task, "train_prompts")
        assert len(recs) == 150
        toks, cols = [], []
        for r in recs:
            ids, c = token_columns(tokenizer, record_to_prompt_data(r, cfg))
            toks.append(ids)
            cols.append(c)

        dot_sum = np.zeros((n_layers, len(COLUMNS)), dtype=np.float64)
        cos_sum = np.zeros((n_layers, len(COLUMNS)), dtype=np.float64)
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
                col_idx = [[[j for j in range(lens[i]) if bc[i][j] == c] for c in range(len(COLUMNS))]
                           for i in range(len(bt))]
                for l, name in enumerate(cfg["layer_hook_names"]):
                    h = td[name].output
                    if isinstance(h, tuple):
                        h = h[0]
                    z50 = h.float() @ U_dev.T                # (B, S, 50)
                    dots = z50 @ v50_hat                     # (B, S)
                    coss = dots / z50.norm(dim=-1).clamp_min(1e-8)
                    dots, coss = dots.cpu().numpy(), coss.cpu().numpy()
                    for i in range(len(bt)):
                        for c in range(len(COLUMNS)):
                            dot_sum[l, c] += dots[i, col_idx[i][c]].mean()
                            cos_sum[l, c] += coss[i, col_idx[i][c]].mean()
                n_seen += len(bt)
                print(f"[{task}] {n_seen}/150", flush=True)
        finally:
            tokenizer.padding_side = old_side

        np.savez(out,
                 dot_mean=(dot_sum / n_seen).astype(np.float32),
                 cos_mean=(cos_sum / n_seen).astype(np.float32),
                 n_prompts=n_seen, columns=np.array(COLUMNS),
                 group=group_of[task], fv_proj_cos=proj_cos,
                 fv_norm=float(v.norm()), fv50_norm=float(v50.norm()))
        print(f"[{task}] done (||Pv||/||v||={proj_cos:.3f})", flush=True)


if __name__ == "__main__":
    main()
