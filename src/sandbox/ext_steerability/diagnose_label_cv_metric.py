#!/usr/bin/env python
"""Diagnose why every CV fold of the label-slot sparse optimisation scored 0.000.

For a few train tasks, compare on the same 1-shot dummy-label points:
  * unsteered:            mean -log p(gold label), full-label argmax acc, first-token acc
  * steered with the CV c vector (weighted, from a chosen lambda/fold artifact)
  * steered with the ALL-head label-slot sum (c = 1) as an upper reference
If the steered log-prob improves a lot while full-label accuracy stays 0, the plumbing is
fine and the CV criterion is simply too strict to discriminate lambdas.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.sandbox.isolation_upper_bound.run_task import build_contributions_single
from src.sandbox.sparse_head_selection.train_sparse_heads import batch_label_logprobs
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
try:
    from src.sandbox.ext_steerability.train_sparse_label_heads import scaffold_point
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_sparse_label_heads import scaffold_point


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_tasks", type=int, default=4)
    p.add_argument("--n_points", type=int, default=30)
    p.add_argument("--inject_layer", type=int, default=7)
    p.add_argument("--cv_artifact", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "read_vector_head_selection"
                   / "pooled_sparse" / "lambda0.05_fold0.pt")
    p.add_argument("--means_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_head_means")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    return p.parse_args()


def first_token_acc(model, model_config, tokenizer, pts, v, layer):
    """Argmax of the FIRST gold token only (looser than the full-label criterion)."""
    ok = 0
    with torch.no_grad():
        for i in range(0, len(pts), 8):
            b = pts[i:i + 8]
            seqs = [p["prompt_ids"] for p in b]
            L = max(len(s) for s in seqs)
            ids = torch.full((len(b), L), tokenizer.pad_token_id, dtype=torch.long)
            att = torch.zeros(len(b), L, dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, :len(s)] = torch.tensor(s)
                att[r, :len(s)] = 1
            ids, att = ids.cuda(), att.cuda()
            cue = torch.tensor([p["cue_idx"] for p in b], device="cuda")
            bar = torch.arange(len(b), device="cuda")
            handle = None
            if v is not None:
                block = model.transformer.h[layer]

                def hook(module, inputs, output):
                    hid = output[0] if isinstance(output, tuple) else output
                    add = torch.zeros_like(hid, dtype=torch.float32)
                    add[bar, cue] = v
                    hid = hid + add.to(hid.dtype)
                    return (hid,) + tuple(output[1:]) if isinstance(output, tuple) else hid
                handle = block.register_forward_hook(hook)
            try:
                out = model(input_ids=ids, attention_mask=att)
            finally:
                if handle is not None:
                    handle.remove()
            last = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
            pred = out.logits[bar, last].argmax(dim=-1)
            for r, p in enumerate(b):
                ok += int(pred[r].item() == p["label_ids"][0])
    return ok / len(pts)


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    tasks = sorted(split["train_tasks"])[:args.n_tasks]
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)
    cv = torch.load(args.cv_artifact, map_location="cpu", weights_only=False)
    c = cv["c"].cuda()
    print(f"cv artifact {args.cv_artifact.name}: lam={cv['lambda']} fold={cv['fold']} "
          f"c_max={float(c.max()):.3f} n>0.8={int((c > 0.8).sum())}", flush=True)

    for t in tasks:
        recs = json.load(open(args.prompts_root / t / "train_prompts.json"))[:args.n_points]
        pts = [scaffold_point(r, t, tokenizer) for r in recs]
        # label_ids sanity
        dec = tokenizer.decode(pts[0]["label_ids"]).strip()
        C = build_contributions_single(
            torch.load(args.means_root / f"{t}.pt", map_location="cpu",
                       weights_only=False)["head_means"], model, model_config)
        v_cv = (c.unsqueeze(1) * C).sum(dim=0)
        v_all = C.sum(dim=0)
        rows = []
        for name, v in (("unsteered", None), ("cv_c", v_cv), ("all_heads", v_all)):
            lps, accs = [], []
            for i in range(0, len(pts), 8):
                b = pts[i:i + 8]
                vb = None if v is None else v.unsqueeze(0).expand(len(b), -1)
                lp, ac = batch_label_logprobs(model, model_config, tokenizer, b, v=vb,
                                              inject_layer=args.inject_layer)
                lps += [float(x) for x in lp]
                accs += [bool(x) for x in ac]
            ft = first_token_acc(model, model_config, tokenizer, pts, v, args.inject_layer)
            rows.append((name, float(np.mean(lps)), float(np.mean(accs)), ft,
                         0.0 if v is None else float(v.norm())))
        print(f"\n{t}: target={pts[0]['target']!r} label_ids decode={dec!r} "
              f"cue_idx={pts[0]['cue_idx']}", flush=True)
        for name, lp, acc, ft, nrm in rows:
            print(f"   {name:10s} -logp={lp:7.3f} fulllabel_acc={acc:.3f} "
                  f"firsttok_acc={ft:.3f} ||v||={nrm:.1f}", flush=True)


if __name__ == "__main__":
    main()
