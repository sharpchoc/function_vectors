#!/usr/bin/env python
"""Per-prompt FV capture for the 69-task-run train tasks (GPU).

Per task: for each of the 150 fixed 10-shot train prompts, capture the cue-token
out_proj inputs of the 37 pooled-selected heads (prunedfail_seed43 selection) and the
built per-prompt FV (sum of W_O-projected head outputs, fp16). Linearity gate as in
stage_capture. Writes ARTIFACTS_ROOT/69_task_run/perprompt_fvs/<task>.pt with
{sel_flat, raw (150,37,256), fv (150,4096), prompt_index}. Fan out with --tasks.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from baukit import TraceDict  # noqa: E402
from src.sandbox.isolation_upper_bound.run_task import (  # noqa: E402
    auto_batch, load_records, record_to_prompt_data)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: E402
from src.utils.varicl_utils import split_activations_by_head  # noqa: E402
from src.utils.paths import ARTIFACTS_ROOT  # noqa: E402
from src.utils.prompt_utils import create_prompt  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--selection_path", type=Path, required=True)
    p.add_argument("--out_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capture_batch", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    sel = json.load(open(args.selection_path))
    sel_flat = sorted(sel["selected_flat"])
    model, tokenizer, cfg = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)
    n_layers, n_heads, resid = cfg["n_layers"], cfg["n_heads"], cfg["resid_dim"]
    head_dim = resid // n_heads
    by_layer = {}
    for f in sel_flat:
        by_layer.setdefault(f // n_heads, []).append(f % n_heads)
    args.out_root.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        out = args.out_root / f"{task}.pt"
        if out.exists():
            print(f"[{task}] exists, skip", flush=True)
            continue
        recs = load_records(args, task, "train_prompts")
        assert len(recs) == 150
        raw = torch.zeros(len(recs), len(sel_flat), head_dim, dtype=torch.float16)
        fv = torch.zeros(len(recs), resid, dtype=torch.float32)
        old_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        linearity_checked = False
        try:
            sents_all = [create_prompt(record_to_prompt_data(r, cfg)) for r in recs]
            max_tok = max(len(tokenizer(s).input_ids) for s in sents_all)
            cap_batch = auto_batch(max_tok, 4000, args.capture_batch)
            for start in range(0, len(recs), cap_batch):
                sents = sents_all[start:start + cap_batch]
                inputs = tokenizer(sents, return_tensors="pt", padding=True).to(model.device)
                prompt_lens = inputs.attention_mask.sum(dim=1) - 1
                bidx = torch.arange(len(sents), device=model.device)
                with torch.no_grad(), TraceDict(model, layers=cfg["attn_hook_names"],
                                                retain_input=True, retain_output=True) as td:
                    model(**inputs)
                col = 0
                for li in range(n_layers):
                    if li not in by_layer:
                        continue
                    lname = cfg["attn_hook_names"][li]
                    inp = td[lname].input
                    inp = inp[0] if isinstance(inp, tuple) else inp
                    heads = split_activations_by_head(inp, cfg)
                    cue = heads[bidx, prompt_lens]  # (B, H, hd)
                    w = model.transformer.h[li].attn.out_proj.weight.detach()
                    wv = w.view(resid, n_heads, head_dim)
                    if not linearity_checked:
                        rebuilt = torch.einsum("bhd,ehd->be", cue.to(w.dtype), wv)
                        outp = td[lname].output
                        outp = outp[0] if isinstance(outp, tuple) else outp
                        ref = outp[bidx, prompt_lens]
                        dev = (rebuilt - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
                        assert dev < 5e-2, f"linearity gate failed L{li}: {dev:.3e}"
                        linearity_checked = True
                    for h in by_layer[li]:
                        act = cue[:, h]  # (B, hd)
                        raw[start:start + len(sents), col] = act.half().cpu()
                        fv[start:start + len(sents)] += torch.einsum(
                            "bd,ed->be", act.float(), wv[:, h].float()).cpu()
                        col += 1
                assert col == len(sel_flat)
        finally:
            tokenizer.padding_side = old_side
        torch.save({"task": task, "sel_flat": sel_flat,
                    "selection_path": str(args.selection_path),
                    "raw": raw, "fv": fv.half(),
                    "prompt_index": [r["prompt_index"] for r in recs]}, out)
        print(f"[{task}] saved {tuple(fv.shape)}", flush=True)
    print("PERPROMPT CAPTURE DONE")


if __name__ == "__main__":
    main()
