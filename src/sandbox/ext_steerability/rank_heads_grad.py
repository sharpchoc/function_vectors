#!/usr/bin/env python
"""Gradient of the pooled zero-shot label NLL w.r.t. each head coefficient at a finished sparse fit.

Many heads end a pooled-sparse fit pinned at c = 1 (the L1 clamp), so "top-k by c" is undefined among
them. First-order, a head pinned at 1 drops out of the solution when lambda exceeds -dNLL/dc_h, so
-dNLL/dc_h (the "dropout lambda") ranks the saturated heads by how much the objective needs them.
Replicates refit_selection_fixedlam.py's setup (same split, points, C, injection layer), evaluates one
pass over the train points at c = coeffs_final.pt and writes <out_root>/pooled_sparse/grad_at_final.pt
{"grad": (n_heads,), "c": (n_heads,), "n_points"}.
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
from src.sandbox.isolation_upper_bound.run_task import build_contributions_single, load_records, record_to_point  # noqa: E402
from src.sandbox.sparse_head_selection.train_sparse_heads import batch_label_logprobs  # noqa: E402
from src.utils.model_utils import load_gpt_model_and_tokenizer  # noqa: E402
from src.utils.paths import REPO_ROOT, QWEN25_MODEL  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_path", type=Path, required=True)
    ap.add_argument("--prompts_root", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True, help="the fit root (has pooled_sparse/ and <task>/means.pt)")
    ap.add_argument("--model_name", default=QWEN25_MODEL)
    ap.add_argument("--points_per_task", type=int, default=100)
    ap.add_argument("--micro_batch_size", type=int, default=32)
    a = ap.parse_args()
    split = json.load(open(a.split_path))
    train_tasks = split["train_tasks"]
    task_index = {t: i for i, t in enumerate(train_tasks)}
    sel = json.load(open(a.out_root / "pooled_sparse" / "selection.json"))
    inject_layer = int(sel["inject_layer"])
    c = torch.load(a.out_root / "pooled_sparse" / "coeffs_final.pt", map_location="cpu", weights_only=False)["c"].float()

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(a.model_name)
    model.eval(); torch.set_grad_enabled(False)
    C = torch.stack([build_contributions_single(
            torch.load(a.out_root / t / "means.pt", map_location="cpu", weights_only=False)["head_means"],
            model, model_config) for t in train_tasks])
    model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    points = []
    for t in train_tasks:
        recs = load_records(a, t, "train_zeroshot")[:a.points_per_task]
        points += [record_to_point(r, tokenizer, model_config) for r in recs]
    print(f"train tasks {len(train_tasks)}, points {len(points)}, inject L{inject_layer}, "
          f"c==1: {int((c >= .999).sum())}, c>.8: {int((c > .8).sum())}", flush=True)

    c = c.to(C.device).requires_grad_(True)
    torch.set_grad_enabled(True)
    total = 0.0
    for s in range(0, len(points), a.micro_batch_size):
        micro = points[s:s + a.micro_batch_size]
        t_idx = torch.tensor([task_index[b["task"]] for b in micro], device=C.device)
        v = torch.einsum("h,bhd->bd", c, C[t_idx])
        nll, _ = batch_label_logprobs(model, model_config, tokenizer, micro, v=v, inject_layer=inject_layer)
        (nll.sum() / len(points)).backward()
        total += nll.sum().item()
    g = c.grad.detach().cpu()
    assert torch.isfinite(g).all() and g.abs().sum() > 0
    out = a.out_root / "pooled_sparse" / "grad_at_final.pt"
    torch.save({"grad": g, "c": c.detach().cpu(), "n_points": len(points), "mean_nll": total / len(points),
                "definition": "d(mean zero-shot label NLL over train points)/dc_h at the final c; "
                              "-grad = first-order dropout lambda"}, out)
    sat = c.detach().cpu() >= .999
    print(f"mean NLL {total / len(points):.4f}; grad over saturated heads: min {g[sat].min():.4f} "
          f"median {g[sat].median():.4f} max {g[sat].max():.4f}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
