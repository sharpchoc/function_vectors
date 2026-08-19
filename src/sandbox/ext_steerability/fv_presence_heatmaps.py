#!/usr/bin/env python
"""Is there more "FV" in the residual stream when we steer with a read direction?

For a task, run the 1-shot dummy-label scaffold twice — unsteered, and with the task's
dot_perhead read direction (natural magnitude, alpha=1) injected at the ' _' token at L3 —
and record, at every (layer, token position):
    cos(residual, v_A)                and         <residual, v_A/||v_A||>   (projection)
where v_A is the canonical 37-head CUE-TOKEN function vector for that task (built from
artifacts/sandbox/ext_steerability/prunedfail_seed43/<task>/means.pt with the 37-head
pooled_sparse selection).

Prompts vary in length, so we average over the modal (length, ' _' index) group, which keeps
token positions exactly aligned and lets the x-axis be actual tokens.

Output: artifacts/69_task_run/fv_presence/<task>.npz
  cos_unsteered/proj_unsteered/cos_steered/proj_steered  (n_layers, n_positions)
  tokens (representative prompt), inj_idx, n_prompts, fv_norm, readdir_norm
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
    from src.sandbox.ext_steerability.eval_label_slot_vectors import wo_slices, head_sum_vector
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from steer_read_dir_1shot import load_model
    from steer_read_dir_methods import Injector, build_items
    from eval_label_slot_vectors import wo_slices, head_sum_vector


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
    p.add_argument("--fv_means_root", type=Path,
                   default=ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43")
    p.add_argument("--sweep_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "read_dir_sweep")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "fv_presence")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    sel = sorted(json.load(open(args.fv_selection))["selected_flat"])
    assert len(sel) == 37
    model, tok = load_model(args.model_dir)
    tok.padding_side = "right"
    wos = wo_slices(model)
    n_layers = len(model.transformer.h)
    inj = Injector(model, [args.inject_layer])

    for task in args.tasks:
        items = build_items(task, args.prompts_root, tok)
        key = Counter((len(it["ids"]), it["inj_idx"]) for it in items).most_common(1)[0]
        (Lp, inj_idx), n = key
        group = [it for it in items if len(it["ids"]) == Lp and it["inj_idx"] == inj_idx]
        print(f"{task}: modal group len={Lp} inj_idx={inj_idx} -> {len(group)}/150 prompts",
              flush=True)

        # canonical 37-head CUE-token FV for this task
        hm_cue = torch.load(args.fv_means_root / task / "means.pt", map_location="cpu",
                            weights_only=False)["head_means"]
        fv = head_sum_vector(hm_cue, sel, wos)                       # (4096,) fp32 cuda
        fv_u = fv / fv.norm()
        # the read direction we steer with (natural magnitude)
        d = torch.load(args.sweep_root / args.bracket / f"{task}.pt", map_location="cpu",
                       weights_only=False)
        rvec = (d["r_task"].float() * float(d["r_task_norm"])).cuda()
        print(f"   ||FV||={fv.norm():.1f}  ||readdir||={rvec.norm():.1f}  "
              f"cos(readdir,FV)={torch.nn.functional.cosine_similarity(rvec, fv, dim=0):.3f}",
              flush=True)

        out = {}
        for cond, vec in (("unsteered", None), ("steered", args.alpha * rvec)):
            cos_sum = torch.zeros(n_layers, Lp, dtype=torch.float64)
            proj_sum = torch.zeros(n_layers, Lp, dtype=torch.float64)
            seen = 0
            for s in range(0, len(group), args.batch_size):
                b = group[s:s + args.batch_size]
                ids = torch.tensor([it["ids"] for it in b]).cuda()
                att = torch.ones_like(ids)
                mask = torch.zeros(len(b), Lp, dtype=torch.bool)
                mask[:, inj_idx] = True
                inj.vec = vec
                inj.mask = mask.cuda()
                with torch.no_grad():
                    o = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
                inj.mask = None
                inj.vec = None
                # hidden_states[0] is the embedding output; blocks are 1..n_layers
                hs = torch.stack(o.hidden_states[1:], dim=0).float()   # (L, B, T, D)
                c = torch.nn.functional.cosine_similarity(
                    hs, fv_u.view(1, 1, 1, -1).expand_as(hs), dim=-1)  # (L, B, T)
                pr = hs @ fv_u                                          # (L, B, T)
                cos_sum += c.sum(dim=1).double().cpu()
                proj_sum += pr.sum(dim=1).double().cpu()
                seen += len(b)
            out[f"cos_{cond}"] = (cos_sum / seen).numpy()
            out[f"proj_{cond}"] = (proj_sum / seen).numpy()
            print(f"   {cond}: max cos {out[f'cos_{cond}'].max():.3f}", flush=True)

        np.savez_compressed(
            args.out_root / f"{task}.npz",
            tokens=np.array([tok.decode([t]) for t in group[0]["ids"]]),
            inj_idx=inj_idx, n_prompts=len(group), inject_layer=args.inject_layer,
            alpha=args.alpha, bracket=args.bracket,
            fv_norm=float(fv.norm()), readdir_norm=float(rvec.norm()), **out)
        print(f"   wrote {args.out_root / (task + '.npz')}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
