#!/usr/bin/env python
"""Location capture (GPU) for the L6 MEAN LABEL-TOKEN vector — the best steering vector.

Same analysis as capture_69_fv_location.py / capture_69_readdir_location.py, but the task
direction is the vector that actually steers best at the label slot (see
results/69_task_run/raw_mean_steering: peak 0.126 mean accuracy at L6-L7):

    v_hat_A = m_A(L6) / ||m_A(L6)||

where m_A(L6) is task A's mean block-6 output at the last token of the 10th demo's label
(artifacts/69_task_run/label_resid_means/<task>.pt, 'resid_means' row 6).

Everything else is identical to the precedent: clean 10-shot forward passes over the 150
fixed train prompts, block outputs 0..27, per token
    dot = z_l . v_hat_A     cos = cos(z_l, v_hat_A)
averaged into the 32 structural columns (demo1..10 x {input, cue, label} + query
{input, cue}). Writes ARTIFACTS_ROOT/69_task_run/labelmean_location/<task>.npz.
Fan out with --tasks.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# local bootstrap for in-repo runs; a PYTHONPATH-supplied repo also works (staged copies),
# in which case REPO_ROOT below comes from src.utils.paths rather than this file's location
_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from baukit import TraceDict  # noqa: E402
try:
    from src.eval_scripts.capture_69_fv_location import COLUMNS, token_columns  # noqa: E402
except ModuleNotFoundError:  # staged copy outside the repo tree: import the sibling file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from capture_69_fv_location import COLUMNS, token_columns  # noqa: E402
from src.sandbox.isolation_upper_bound.run_task import (  # noqa: E402
    auto_batch, load_records, record_to_prompt_data)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: E402
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT  # noqa: E402

LAYER = 6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--means_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "labelmean_location")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
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
    split = json.load(open(args.split_path))
    group_of = {t: "train" for t in split["train_tasks"]}
    group_of.update({t: "heldout" for t in split["heldout_tasks"]})

    for task in args.tasks:
        out = args.out_root / f"{task}.npz"
        if out.exists():
            print(f"[{task}] exists, skip", flush=True)
            continue

        md = torch.load(args.means_root / f"{task}.pt", map_location="cpu", weights_only=False)
        rm = md["resid_means"].double()
        assert rm.shape == (n_layers, 4096), f"{task}: resid_means {tuple(rm.shape)}"
        v = rm[LAYER]
        v_norm = float(v.norm())
        v_hat = (v / v.norm()).float().to(model.device)

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
                col_idx = [[[j for j in range(lens[i]) if bc[i][j] == c]
                            for c in range(len(COLUMNS))] for i in range(len(bt))]
                for l, name in enumerate(cfg["layer_hook_names"]):
                    h = td[name].output
                    if isinstance(h, tuple):
                        h = h[0]
                    h = h.float()
                    dots = h @ v_hat
                    coss = dots / h.norm(dim=-1).clamp_min(1e-8)
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
                 group=group_of[task], vector="label_resid_mean_L6_unit",
                 source_layer=LAYER, vec_norm=v_norm)
        print(f"[{task}] done (||m_A(L{LAYER})||={v_norm:.1f})", flush=True)


if __name__ == "__main__":
    main()
