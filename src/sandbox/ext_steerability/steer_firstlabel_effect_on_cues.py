#!/usr/bin/env python
"""Steer ONLY the first dummy label; watch every downstream cue token (read -> write chain).

USER REQUEST 2026-08-19: 6-shot dummy scaffold as in steer_effect_on_cue.py, but inject
alpha * m_A(L6) at the FIRST '_' slot only, and read the residual stream at each of the six
downstream cue tokens — the demo-2..demo-6 cues (the last token of each "A:", i.e. the
position directly before that demo's '_') and the final query cue. The demo-1 cue precedes
the intervention causally and is skipped. Shows how a single steered label propagates into
the later write sites.

Per prompt, per alpha in {0, 0.5, 1, 2, 4}, per cue position, at every layer we record
cos/projection of the activation onto v_task (the task's FV) and v_generic (all-task mean
FV), exactly as in steer_effect_on_cue.py. Deltas vs alpha=0 are formed at plot time.

Outputs: artifacts/69_task_run/mean_read_steering_effect_on_write_firstlabel/<task>.pt
  {cos_task, cos_gen, proj_task, proj_gen: (n_alpha, n_prompts, 6, 28) fp32,
   positions: ["cue2".."cue6", "query_cue"], alphas, norms}
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
    from src.sandbox.ext_steerability.steer_read_dir_methods import Injector
    from src.sandbox.ext_steerability.sixshot_dummy_steer import build_items_6shot
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from steer_read_dir_1shot import load_model, batches_by_len
    from steer_read_dir_methods import Injector
    from sixshot_dummy_steer import build_items_6shot

ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0)
INJECT_LAYER = 6
N_LAYERS = 28
D_MODEL = 4096
POSITIONS = ["cue2", "cue3", "cue4", "cue5", "cue6", "query_cue"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resid_means_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means")
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" /
                   "mean_read_steering_effect_on_write_firstlabel")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=9000)
    p.add_argument("--batch_cap", type=int, default=12)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


class MultiCueReader:
    """Forward hooks on every block: stash hidden states at P positions per row."""

    def __init__(self, model, n_pos):
        self.n_pos = n_pos
        self.idx = None           # (B, P) long cuda
        self.buf = None           # (B, P, 28, D) fp32 cuda
        self.handles = [model.transformer.h[l].register_forward_hook(self._make(l))
                        for l in range(N_LAYERS)]

    def _make(self, l):
        def hook(module, args, output):
            if self.idx is None:
                return None
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] <= 1:
                return None
            rows = torch.arange(hs.shape[0], device=hs.device).unsqueeze(1)
            self.buf[:, :, l] = hs[rows, self.idx].float()
            return None
        return hook


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    all_tasks = sorted(group)
    tasks = all_tasks[args.shard_idx::args.shard_n]
    args.out_root.mkdir(parents=True, exist_ok=True)

    fvs = {t: torch.load(args.fv_root / f"{t}.pt", map_location="cpu",
                         weights_only=False)["fv"].float().mean(0) for t in all_tasks}
    v_generic = torch.stack([fvs[t] for t in all_tasks]).mean(0)
    print(f"{len(tasks)} tasks on this shard | ||v_generic||={v_generic.norm():.2f}",
          flush=True)

    model, tok = load_model(args.model_dir)
    tok.padding_side = "left"
    inj = Injector(model, [INJECT_LAYER])
    reader = MultiCueReader(model, len(POSITIONS))
    vg = v_generic.cuda()
    vg_hat = vg / vg.norm()

    for task in tasks:
        out_path = args.out_root / f"{task}.pt"
        if out_path.exists():
            print(f"{task}: exists, skip", flush=True)
            continue
        items = build_items_6shot(task, args.prompts_root, tok, real_labels=False)
        for it in items:
            assert len(it["inj_idx_list"]) == 6
            # cue tokens: directly before each '_' (the last token of "A:"); final query
            # cue is the last prompt token. Skip cue1 (precedes the intervention).
            it["cue_idx"] = [it["inj_idx_list"][k] - 1 for k in range(1, 6)] + \
                            [len(it["ids"]) - 1]
        m = torch.load(args.resid_means_root / f"{task}.pt", map_location="cpu",
                       weights_only=False)["resid_means"][INJECT_LAYER].float().cuda()
        vt = fvs[task].cuda()
        vt_hat = vt / vt.norm()
        n = len(items)
        res = {k: torch.zeros(len(ALPHAS), n, len(POSITIONS), N_LAYERS)
               for k in ("cos_task", "cos_gen", "proj_task", "proj_gen")}

        for ai, a in enumerate(ALPHAS):
            inj.vec = None if a == 0 else a * m
            for b in batches_by_len(items, args.token_budget, args.batch_cap):
                lens = [len(items[i]["ids"]) for i in b]
                L = max(lens)
                ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
                att = torch.zeros(len(b), L, dtype=torch.long)
                mask = torch.zeros(len(b), L, dtype=torch.bool)
                idx = torch.zeros(len(b), len(POSITIONS), dtype=torch.long)
                for r, i in enumerate(b):
                    nn = lens[r]
                    off = L - nn                      # left padding
                    ids[r, off:] = torch.tensor(items[i]["ids"])
                    att[r, off:] = 1
                    mask[r, off + items[i]["inj_idx_list"][0]] = True   # FIRST '_' only
                    idx[r] = torch.tensor([off + p for p in items[i]["cue_idx"]])
                inj.mask = mask.cuda()
                reader.idx = idx.cuda()
                reader.buf = torch.zeros(len(b), len(POSITIONS), N_LAYERS, D_MODEL,
                                         device="cuda")
                with torch.no_grad():
                    model(input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False)
                acts = reader.buf                     # (B, P, 28, D)
                inj.mask = None
                reader.idx, reader.buf = None, None
                ct = torch.nn.functional.cosine_similarity(acts, vt.view(1, 1, 1, -1), dim=3)
                cg = torch.nn.functional.cosine_similarity(acts, vg.view(1, 1, 1, -1), dim=3)
                pt = acts @ vt_hat
                pg = acts @ vg_hat
                for r, i in enumerate(b):
                    res["cos_task"][ai, i] = ct[r].cpu()
                    res["cos_gen"][ai, i] = cg[r].cpu()
                    res["proj_task"][ai, i] = pt[r].cpu()
                    res["proj_gen"][ai, i] = pg[r].cpu()
            print(f"{task} | alpha={a}: L13 cos_task cue2={res['cos_task'][ai, :, 0, 13].mean():.4f} "
                  f"query={res['cos_task'][ai, :, -1, 13].mean():.4f}", flush=True)
        inj.vec = None
        res.update({"task": task, "group": group[task], "alphas": list(ALPHAS),
                    "positions": POSITIONS, "inject_layer": INJECT_LAYER, "n_prompts": n,
                    "norm_v_task": float(vt.norm()), "norm_v_generic": float(vg.norm()),
                    "norm_m": float(m.norm()),
                    "site": "first '_' slot steered; readout at cue2..cue6 + query cue, "
                            "all 28 block outputs"})
        torch.save(res, out_path)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
