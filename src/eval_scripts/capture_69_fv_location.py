#!/usr/bin/env python
"""FV-location capture for the 69-task pool (GPU): where in the residual stream does the
task's FV direction appear during a clean 10-shot forward pass?

Per task: v_A = mean of the 150 per-prompt FVs (artifacts/69_task_run/perprompt_fvs,
canonical 37-head set; consistency-gated against a means.pt + W_O rebuild). For each of the
150 fixed 10-shot train prompts, run a clean forward pass, capture every block output
(layers 0..27, same hook points as injection), and record at every token position both
  dot = z . v_hat          (raw amount of the FV direction present)
  cos = z . v_hat / ||z||  (norm-normalized alignment)
Positions are then averaged into 32 structural columns: demo{1..10} x {input, cue, label}
+ query {input, cue}, where input = the input-word tokens, cue = the "A:" tokens, label =
the label-word tokens (structural "Q:"/newline/BOS tokens excluded). Multi-token spans are
averaged. Writes ARTIFACTS_ROOT/69_task_run/fv_location/<task>.npz with the per-task
prompt-averaged (28, 32) matrices. Fan out with --tasks.
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
from src.sandbox.isolation_upper_bound.run_task import (  # noqa: E402
    auto_batch, build_contributions_single, load_records, record_to_prompt_data)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: E402
from src.utils.paths import ARTIFACTS_ROOT  # noqa: E402
from src.utils.prompt_utils import create_prompt  # noqa: E402

N_DEMOS = 10
COLUMNS = ([f"demo{k}_{c}" for k in range(1, N_DEMOS + 1) for c in ("input", "cue", "label")]
           + ["query_input", "query_cue"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--means_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43")
    p.add_argument("--selection_path", type=Path, required=True)
    p.add_argument("--out_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "fv_location")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capture_batch", type=int, default=8)
    return p.parse_args()


def prompt_segments(prompt_data):
    """Assemble the prompt as (text, column) segments, mirroring create_prompt exactly.
    column is an index into COLUMNS or -1 for structural text."""
    pre, sep = prompt_data["prefixes"], prompt_data["separators"]
    segs = [(pre["instructions"] + prompt_data["instructions"] + sep["instructions"], -1)]
    for k, ex in enumerate(prompt_data["examples"]):
        base = 3 * k
        segs += [(pre["input"], -1), (ex["input"], base), (sep["input"], -1),
                 (pre["output"], base + 1), (ex["output"], base + 2), (sep["output"], -1)]
    q = prompt_data["query_target"]["input"]
    segs += [(pre["input"], -1), (q, 3 * N_DEMOS), (sep["input"], -1),
             (pre["output"], 3 * N_DEMOS + 1)]
    return [(t, c) for t, c in segs if t != ""]


def token_columns(tokenizer, prompt_data):
    """Tokenize the assembled prompt; return (input_ids, col_per_token) with col in
    [0, 32) or -1. Gated: the assembled string must equal create_prompt's output, every
    non-structural column must receive >= 1 token, and token/segment boundaries must
    cover each other exactly (offsets are contiguous for GPT-2 BPE)."""
    segs = prompt_segments(prompt_data)
    text = "".join(t for t, _ in segs)
    assert text == create_prompt(prompt_data), "segment assembly diverged from create_prompt"
    enc = tokenizer(text, return_offsets_mapping=True, truncation=False, padding=False)
    bounds, pos = [], 0
    for t, c in segs:
        bounds.append((pos, pos + len(t), c))
        pos += len(t)
    cols = []
    for (a, b) in enc["offset_mapping"]:
        col = -1
        for (s, e, c) in bounds:
            if a >= s and a < e:
                col = c
                break
        cols.append(col)
    counts = np.bincount([c for c in cols if c >= 0], minlength=len(COLUMNS))
    assert (counts > 0).all(), f"empty column(s): {[COLUMNS[i] for i in np.where(counts == 0)[0]]}"
    return enc["input_ids"], cols


def main():
    args = parse_args()
    set_seed(args.seed)
    sel = json.load(open(args.selection_path))
    sel_flat = sorted(sel["selected_flat"])
    model, tokenizer, cfg = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)
    n_layers = cfg["n_layers"]
    args.out_root.mkdir(parents=True, exist_ok=True)
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group_of = {t: "train" for t in split["train_tasks"]}
    group_of.update({t: "heldout" for t in split["heldout_tasks"]})

    for task in args.tasks:
        out = args.out_root / f"{task}.npz"
        if out.exists():
            print(f"[{task}] exists, skip", flush=True)
            continue

        # --- task FV direction: mean of per-prompt FVs, gated vs means.pt rebuild ---
        pp = torch.load(args.fv_root / f"{task}.pt", map_location="cpu", weights_only=False)
        v = pp["fv"].double().mean(dim=0)
        means = torch.load(args.means_root / task / "means.pt", map_location="cpu", weights_only=False)
        C = build_contributions_single(means["head_means"], model, cfg)
        v_rebuild = C[sel_flat].sum(dim=0).double().cpu()
        gate_cos = torch.nn.functional.cosine_similarity(v, v_rebuild, dim=0).item()
        assert gate_cos > 0.999, f"{task}: perprompt-mean vs means.pt FV cos {gate_cos:.5f}"
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
                # per-prompt per-column token index lists (fixed across layers)
                col_idx = [[[j for j in range(lens[i]) if bc[i][j] == c] for c in range(len(COLUMNS))]
                           for i in range(len(bt))]
                for l, name in enumerate(cfg["layer_hook_names"]):
                    h = td[name].output
                    if isinstance(h, tuple):
                        h = h[0]
                    h = h.float()
                    dots = h @ v_hat                        # (B, S)
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
                 group=group_of[task], gate_cos=gate_cos, fv_norm=float(v.norm()))
        print(f"[{task}] done (gate cos {gate_cos:.5f}, ||v||={v.norm():.2f})", flush=True)


if __name__ == "__main__":
    main()
