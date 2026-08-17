#!/usr/bin/env python
"""Narrow subspace PATCHING at the '_' label slot, L6 (69 tasks).

Variant of raw-mean steering (eval-levers S4.3, the patched multi-direction form): with V
the 41-PC / 95%-variance basis of the 69 task-mean L6 label-token activations
(build_L6_pc41_basis.py) and P = V^T V:

    z(' _', L6)  <-  (I - P) z + alpha * P m_A(L6)

i.e. the prompt's own content in the subspace is REMOVED and replaced by the task mean's
projection. alpha scales the replacement only (the removal always happens); alpha = 1 is
the pure patch. The PCA center cancels exactly in remove-and-replace, so P is applied to
raw vectors. Also runs a remove-only control (alpha = 0).

Scaffold, prompts, readout, seeding identical to sweep_raw_mean_layers.py.
Outputs: artifacts/69_task_run/raw_mean_steering/narrow_patch/<task>.json
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
try:
    from src.sandbox.ext_steerability.steer_read_dir_1shot import load_model, batches_by_len
    from src.sandbox.ext_steerability.steer_read_dir_methods import build_items
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from steer_read_dir_1shot import load_model, batches_by_len
    from steer_read_dir_methods import build_items

ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0)   # 0 = remove-only control; 1 = pure patch
LAYER = 6


class Patcher:
    """Forward hook on block LAYER: z <- (I-P) z + repl at masked positions (prefill only).
    repl is set per condition as alpha * (m_A @ V^T) @ V (a fixed vector)."""

    def __init__(self, model, V):
        self.V = V            # (k, 4096) fp32 cuda
        self.repl = None      # (4096,) fp32 cuda, or None -> hook inactive
        self.mask = None
        self.h = model.transformer.h[LAYER].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self.repl is None or self.mask is None or hs.shape[1] != self.mask.shape[1]:
            return None
        z = hs[self.mask].float()
        z = z - (z @ self.V.T) @ self.V + self.repl
        hs = hs.clone()
        hs[self.mask] = z.to(hs.dtype)
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "pc41_basis.pt")
    p.add_argument("--resid_means_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "narrow_patch")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=24000)
    p.add_argument("--batch_cap", type=int, default=48)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)[args.shard_idx::args.shard_n]
    args.out_root.mkdir(parents=True, exist_ok=True)

    basis = torch.load(args.basis_path, map_location="cpu", weights_only=False)
    V = basis["V"].float().cuda()
    print(f"{len(tasks)} tasks on this shard; basis k={basis['k']} "
          f"(cum var {basis['cum_at_k']:.4f}), L{basis['layer']}", flush=True)
    assert basis["layer"] == LAYER

    model, tok = load_model(args.model_dir)
    tok.padding_side = "left"
    patch = Patcher(model, V)

    for task in tasks:
        out_path = args.out_root / f"{task}.json"
        if out_path.exists():
            print(f"{task}: exists, skip", flush=True)
            continue
        items = build_items(task, args.prompts_root, tok)
        m = torch.load(args.resid_means_root / f"{task}.pt", map_location="cpu",
                       weights_only=False)["resid_means"][LAYER].float().cuda()
        proj_m = (m @ V.T) @ V
        res = {"task": task, "group": group[task], "n_prompts": len(items), "k": basis["k"],
               "norm_m": float(m.norm()), "norm_proj_m": float(proj_m.norm()),
               "definition": "z <- (I-P) z + alpha * P m_A at ' _' L6; alpha=0 remove-only, "
                             "alpha=1 pure patch", "conditions": {}}

        def run(cname, repl):
            patch.repl = repl
            preds = [None] * len(items)
            for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
                lens = [len(items[i]["ids"]) for i in b]
                L = max(lens)
                ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
                att = torch.zeros(len(b), L, dtype=torch.long)
                mask = torch.zeros(len(b), L, dtype=torch.bool)
                for r, i in enumerate(b):
                    n = lens[r]; off = L - n
                    ids[r, off:] = torch.tensor(items[i]["ids"])
                    att[r, off:] = 1
                    mask[r, off + items[i]["inj_idx"]] = True
                patch.mask = mask.cuda()
                max_new = min(max(items[i]["gold_len"] for i in b) + 3, 16)
                torch.manual_seed(zlib.crc32(f"{task}|{cname}|{bi}".encode()))
                with torch.no_grad():
                    gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                         do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                         max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
                patch.mask = None
                for r, i in enumerate(b):
                    preds[i] = tok.decode(gen[r, L:], skip_special_tokens=True).split("\n")[0].strip()
            patch.repl = None
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | {cname}: acc={acc:.3f}", flush=True)

        for a in ALPHAS:
            run(f"patch_a{a}", a * proj_m if a > 0 else torch.zeros_like(proj_m))
        res["golds"] = [it["gold"] for it in items]
        with open(out_path, "w") as f:
            json.dump(res, f)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
