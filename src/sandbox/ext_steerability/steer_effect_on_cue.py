#!/usr/bin/env python
"""How label-slot steering changes the FINAL CUE TOKEN representation (read side -> write side).

Setup is the 6-shot dummy scaffold of sixshot_dummy_steer.py: six demos with in-distribution
inputs and '_' as every label, then the real query. We inject alpha * m_A(L6) at ALL SIX '_'
slots at block-6 output, alpha in {0, 0.5, 1, 2, 4}, and then — instead of generating — read
the residual stream at the FINAL prompt token (the query's cue) and ask how far it has moved
towards the task's function vector.

Per prompt, per alpha, at every layer (L13 is the headline) we record, for two reference
directions:
    v_task    = the task's function vector (mean of its 150 per-prompt FVs at the cue token)
    v_generic = the mean of v_task over all 69 tasks (a task-agnostic "general" direction)
  cos(a, v)                 raw cosine (stored; deltas are formed at plot time)
  proj(a, v) = a . v/||v||  projection magnitude along the reference
Deltas vs the alpha=0 run are formed at plot time, so "movement towards the FV" is measured
against each prompt's own unsteered representation, and the generic reference shows whether
that movement is task-specific or towards the shared component.

No generation: one forward pass per (prompt, alpha), all 28 layers captured at once.
Outputs: artifacts/69_task_run/mean_read_steering_effect_on_write/<task>.pt
  {cos_task, cos_gen, proj_task, proj_gen: (n_alpha, n_prompts, 28) fp32, alphas, norms}
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
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
try:
    from src.sandbox.ext_steerability.steer_read_dir_1shot import load_model, batches_by_len
    from src.sandbox.ext_steerability.steer_read_dir_methods import Injector, build_items
    from src.sandbox.ext_steerability.sixshot_dummy_steer import build_items_6shot
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from steer_read_dir_1shot import load_model, batches_by_len
    from steer_read_dir_methods import Injector, build_items
    from sixshot_dummy_steer import build_items_6shot

ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0)
INJECT_LAYER = 6   # default; overridden by --inject_layer
N_LAYERS = 28
D_MODEL = 4096


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resid_means_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means")
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--n_shots", type=int, default=6, choices=(1, 6),
                   help="dummy scaffold: 6 = six '_' slots (original), 1 = the single-'_' "
                        "1-shot scaffold; out_root default gains a _1shot suffix")
    p.add_argument("--out_root", type=Path, default=None)
    p.add_argument("--vectors_path", type=Path, default=None,
                   help="fixed per-task steering vectors {tasks:{task:{vec}}} (e.g. the "
                        "carrier + n_A*v1 bank); overrides --resid_means_root")
    p.add_argument("--inject_layer", type=int, default=None,
                   help="block-output layer to inject at (default INJECT_LAYER=6)")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=9000)
    p.add_argument("--batch_cap", type=int, default=12)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


class CueReader:
    """Forward hooks on every block: stash the hidden state at each row's final token."""

    def __init__(self, model):
        self.last_idx = None      # (B,) long cuda
        self.buf = None           # (B, 28, D) fp32 cuda
        self.handles = [model.transformer.h[l].register_forward_hook(self._make(l))
                        for l in range(N_LAYERS)]

    def _make(self, l):
        def hook(module, args, output):
            if self.last_idx is None:
                return None
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] <= 1:
                return None
            rows = torch.arange(hs.shape[0], device=hs.device)
            self.buf[:, l] = hs[rows, self.last_idx].float()
            return None
        return hook


def main():
    args = parse_args()
    if args.out_root is None:
        args.out_root = (ARTIFACTS_ROOT / "69_task_run" /
                         ("mean_read_steering_effect_on_write" +
                          ("" if args.n_shots == 6 else "_1shot")))
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    all_tasks = sorted(group)
    tasks = all_tasks[args.shard_idx::args.shard_n]
    args.out_root.mkdir(parents=True, exist_ok=True)

    fvs = {t: torch.load(args.fv_root / f"{t}.pt", map_location="cpu",
                         weights_only=False)["fv"].float().mean(0) for t in all_tasks}
    v_generic = torch.stack([fvs[t] for t in all_tasks]).mean(0)
    mean_cos = torch.stack([torch.nn.functional.cosine_similarity(fvs[t], v_generic, dim=0)
                            for t in all_tasks]).mean()
    print(f"{len(tasks)} tasks on this shard | ||v_generic||={v_generic.norm():.2f} | "
          f"mean cos(v_task, v_generic)={mean_cos:.3f}", flush=True)

    model, tok = load_model(args.model_dir)
    tok.padding_side = "left"
    global INJECT_LAYER
    if args.inject_layer is not None:
        INJECT_LAYER = args.inject_layer
    fixed = (torch.load(args.vectors_path, map_location="cpu", weights_only=False)["tasks"]
             if args.vectors_path else None)
    inj = Injector(model, [INJECT_LAYER])
    reader = CueReader(model)
    vg = v_generic.cuda()
    vg_hat = vg / vg.norm()

    for task in tasks:
        out_path = args.out_root / f"{task}.pt"
        if out_path.exists():
            print(f"{task}: exists, skip", flush=True)
            continue
        if args.n_shots == 6:
            items = build_items_6shot(task, args.prompts_root, tok, real_labels=False)
        else:
            items = build_items(task, args.prompts_root, tok)
            for it in items:
                it["inj_idx_list"] = [it["inj_idx"]]
        if fixed is not None:
            m = fixed[task]["vec"].float().cuda()
        else:
            m = torch.load(args.resid_means_root / f"{task}.pt", map_location="cpu",
                           weights_only=False)["resid_means"][INJECT_LAYER].float().cuda()
        vt = fvs[task].cuda()
        vt_hat = vt / vt.norm()
        n = len(items)
        res = {k: torch.zeros(len(ALPHAS), n, N_LAYERS)
               for k in ("cos_task", "cos_gen", "proj_task", "proj_gen")}

        for ai, a in enumerate(ALPHAS):
            inj.vec = None if a == 0 else a * m
            for b in batches_by_len(items, args.token_budget, args.batch_cap):
                lens = [len(items[i]["ids"]) for i in b]
                L = max(lens)
                ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
                att = torch.zeros(len(b), L, dtype=torch.long)
                mask = torch.zeros(len(b), L, dtype=torch.bool)
                for r, i in enumerate(b):
                    nn = lens[r]
                    off = L - nn                      # left padding
                    ids[r, off:] = torch.tensor(items[i]["ids"])
                    att[r, off:] = 1
                    for p_ in items[i]["inj_idx_list"]:
                        mask[r, off + p_] = True
                inj.mask = mask.cuda()
                # left-padded, so every row's final real token is the last column
                reader.last_idx = torch.full((len(b),), L - 1, device="cuda", dtype=torch.long)
                reader.buf = torch.zeros(len(b), N_LAYERS, D_MODEL, device="cuda")
                with torch.no_grad():
                    model(input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False)
                acts = reader.buf
                inj.mask = None
                reader.last_idx, reader.buf = None, None
                ct = torch.nn.functional.cosine_similarity(acts, vt.view(1, 1, -1), dim=2)
                cg = torch.nn.functional.cosine_similarity(acts, vg.view(1, 1, -1), dim=2)
                pt = acts @ vt_hat
                pg = acts @ vg_hat
                for r, i in enumerate(b):
                    res["cos_task"][ai, i] = ct[r].cpu()
                    res["cos_gen"][ai, i] = cg[r].cpu()
                    res["proj_task"][ai, i] = pt[r].cpu()
                    res["proj_gen"][ai, i] = pg[r].cpu()
            print(f"{task} | alpha={a}: L13 cos_task={res['cos_task'][ai, :, 13].mean():.4f} "
                  f"cos_gen={res['cos_gen'][ai, :, 13].mean():.4f} "
                  f"proj_task={res['proj_task'][ai, :, 13].mean():.2f}", flush=True)
        inj.vec = None
        res.update({"task": task, "group": group[task], "alphas": list(ALPHAS),
                    "inject_layer": INJECT_LAYER, "n_prompts": n, "n_shots": args.n_shots,
                    "norm_v_task": float(vt.norm()), "norm_v_generic": float(vg.norm()),
                    "norm_m": float(m.norm()),
                    "site": "final prompt token (query cue), all 28 block outputs"})
        torch.save(res, out_path)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
