#!/usr/bin/env python
"""How much attention does the final cue token pay to the injected ' _' label slot?

Same setup as fv_presence_heatmaps.py (1-shot dummy-label scaffold, modal prompt group,
optional dot_perhead read-direction injection at the ' _' slot at L3). Records, for every
layer and head, the attention probability FROM the final cue token TO each source position,
averaged over prompts — unsteered and steered.

Output: artifacts/69_task_run/attn_to_slot/<task>.npz
  attn_unsteered / attn_steered   (n_layers, n_heads, n_positions)
  tokens, inj_idx, n_prompts, fv_head_mask (n_layers, n_heads) for the canonical 37 FV heads
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
try:
    from src.sandbox.ext_steerability.steer_read_dir_1shot import load_model
    from src.sandbox.ext_steerability.steer_read_dir_methods import Injector, build_items
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from steer_read_dir_1shot import load_model
    from steer_read_dir_methods import Injector, build_items


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--inject_layer", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--bracket", type=str, default="dot_perhead")
    p.add_argument("--fv_selection", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43"
                   / "pooled_sparse" / "selection.json")
    p.add_argument("--sweep_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "read_dir_sweep")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "attn_to_slot")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    sel = sorted(json.load(open(args.fv_selection))["selected_flat"])
    model, tok = load_model(args.model_dir)
    tok.padding_side = "right"
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    fv_mask = np.zeros((n_layers, n_heads), dtype=bool)
    for f in sel:
        fv_mask[f // n_heads, f % n_heads] = True
    inj = Injector(model, [args.inject_layer])

    for task in args.tasks:
        items = build_items(task, args.prompts_root, tok)
        (Lp, inj_idx), _ = Counter((len(it["ids"]), it["inj_idx"]) for it in items).most_common(1)[0]
        group = [it for it in items if len(it["ids"]) == Lp and it["inj_idx"] == inj_idx]
        d = torch.load(args.sweep_root / args.bracket / f"{task}.pt", map_location="cpu",
                       weights_only=False)
        rvec = (d["r_task"].float() * float(d["r_task_norm"])).cuda()
        print(f"{task}: {len(group)} prompts, len={Lp}, ' _' at {inj_idx}, cue at {Lp-1}",
              flush=True)

        out = {}
        for cond, vec in (("unsteered", None), ("steered", args.alpha * rvec)):
            acc = torch.zeros(n_layers, n_heads, Lp, dtype=torch.float64)
            seen = 0
            for s in range(0, len(group), args.batch_size):
                b = group[s:s + args.batch_size]
                ids = torch.tensor([it["ids"] for it in b]).cuda()
                mask = torch.zeros(len(b), Lp, dtype=torch.bool)
                mask[:, inj_idx] = True
                inj.vec = vec
                inj.mask = mask.cuda()
                with torch.no_grad():
                    o = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                              output_attentions=True)
                inj.mask = None
                inj.vec = None
                # attentions: tuple of (B, H, T, T); take the FINAL query row
                a = torch.stack([x[:, :, -1, :].float() for x in o.attentions], dim=0)
                acc += a.sum(dim=1).double().cpu()      # (L, H, T)
                seen += len(b)
            out[f"attn_{cond}"] = (acc / seen).numpy()
            u = out[f"attn_{cond}"][:, :, inj_idx]
            print(f"   {cond}: attention(final -> ' _') mean over heads/layers "
                  f"{u.mean():.4f}, max {u.max():.4f} at "
                  f"L{np.unravel_index(u.argmax(), u.shape)[0]}"
                  f"H{np.unravel_index(u.argmax(), u.shape)[1]}", flush=True)

        np.savez_compressed(args.out_root / f"{task}.npz",
                            tokens=np.array([tok.decode([t]) for t in group[0]["ids"]]),
                            inj_idx=inj_idx, n_prompts=len(group), fv_head_mask=fv_mask,
                            inject_layer=args.inject_layer, alpha=args.alpha, **out)
        print(f"   wrote {args.out_root / (task + '.npz')}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
