#!/usr/bin/env python
"""Low-dim task-unique read-feature steering on dummy-label scaffolds, L6, 69 tasks.

USER REQUEST 2026-08-19 (fork "Low Dim Read Feature Steering"): steer by PROJECTION SWAP
in the task's top task-unique SVD direction(s) — the same basis the ablation study removes
(meanremoved_top3_bases.pt, SVD of the 11 unit-normed mean-removed L5-15 features):

    h <- h - P_V h + alpha * r_A,   r_A = sum_i s_i * v_i   over the used directions,

at block-6 OUTPUT at every dummy ' _' label slot. s_i are the singular values of the
unit-normed 11-layer stack ("a measure of how much all layers project along that
direction"); alpha sweeps out the magnitude. Each v_i is SIGN-FIXED so the task's own L6
mean-removed feature projects positively on it (user decision — start with direction 1
ONLY via --n_dirs 1; escalate to 3 if it struggles).

Calibration (2026-08-19): the natural L6 coordinate along v_1 is ~8x s_1 (median; range
4-13), so the sweep extends to 32: alphas (0, .5, 1, 2, 4, 8, 16, 32). alpha=0 = pure
removal control.

Arms: dummy1 (single '_' demo) and dummy6 (six '_' demos), same scaffolds/seeds as
meanfree_dummy_steer.py; dummy{1,6}_baseline reproduce earlier runs as infra cross-checks.
Readout: T=1 sampled exact match, 150 prompts/task.

Outputs: artifacts/69_task_run/raw_mean_steering/taskunique_svd_dummy/<task>.json
(resumable per task; missing conditions filled in).
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
    from src.sandbox.ext_steerability.sixshot_dummy_steer import build_items_6shot
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from steer_read_dir_1shot import load_model, batches_by_len
    from steer_read_dir_methods import build_items
    from sixshot_dummy_steer import build_items_6shot

ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0)
LAYER = 6


class ProjSwap:
    """At masked positions of the hooked block's OUTPUT hidden state (prefill only):
    h <- h - P_V h + vec.  V=(k,D) orthonormal fp32 cuda; vec=(D,) fp32 cuda or None."""

    def __init__(self, model, layer):
        self.V = None
        self.vec = None
        self.mask = None
        self.handle = model.transformer.h[layer].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self.V is None or self.mask is None or hs.shape[1] != self.mask.shape[1]:
            return None
        hs = hs.clone()
        h32 = hs[self.mask].float()
        h32 = h32 - (h32 @ self.V.T) @ self.V
        if self.vec is not None:
            h32 = h32 + self.vec
        hs[self.mask] = h32.to(hs.dtype)
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_dirs", type=int, default=1, choices=(1, 2, 3))
    p.add_argument("--bases_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" /
                   "meanremoved_top3_bases.pt")
    p.add_argument("--acts_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_avg10_L5-15_acts")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" /
                   "taskunique_svd_dummy")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=11000)
    p.add_argument("--batch_cap", type=int, default=16)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=None)
    return p.parse_args()


def signed_dirs(bases, acts_root, split):
    """Per task: V rows sign-fixed toward the task's L6 mean-removed feature.
    Returns {task: (V (3,4096) fp32, s (3,) fp32, c (3,) natural L6 coords, signs)}."""
    all_tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    feats, layers = {}, None
    for t in all_tasks:
        d = torch.load(acts_root / f"{t}.pt", map_location="cpu", weights_only=False)
        layers = d["layers"]
        feats[t] = d["acts"].double().mean(dim=0)
    li6 = layers.index(LAYER)
    X = torch.stack([feats[t] for t in all_tasks])
    mdirs = X.mean(dim=0)
    mdirs = mdirs / mdirs.norm(dim=1, keepdim=True)
    out = {}
    for t in all_tasks:
        V = bases[t]["V"].double()                      # (3, 4096)
        s = bases[t]["s"][:V.shape[0]].double()
        f6 = feats[t][li6]
        f6r = f6 - (f6 @ mdirs[li6]) * mdirs[li6]
        c = V @ f6r                                     # signed natural L6 coords
        signs = torch.sign(c)
        signs[signs == 0] = 1.0
        out[t] = ((signs[:, None] * V).float(), s.float(), (signs * c).float(), signs)
    return out


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)[args.shard_idx::args.shard_n]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    args.out_root.mkdir(parents=True, exist_ok=True)

    bases = torch.load(args.bases_path, map_location="cpu", weights_only=False)["tasks"]
    dirs = signed_dirs(bases, args.acts_root, split)
    k = args.n_dirs
    tag = f"swap{k}"

    model, tok = load_model(args.model_dir)
    tok.padding_side = "left"
    sw = ProjSwap(model, LAYER)

    for task in tasks:
        out_path = args.out_root / f"{task}.json"
        res = json.load(open(out_path)) if out_path.exists() else None
        V, s, c, signs = dirs[task]
        Vk = V[:k].cuda()
        r = (s[:k].cuda()[:, None] * Vk).sum(dim=0)     # (4096,) = sum_i s_i v_i
        if res is None:
            res = {"task": task, "group": group[task], "n_prompts": 150,
                   "layer": LAYER, "alphas": list(ALPHAS), "n_dirs_available": 3,
                   "s_top3": [round(float(x), 4) for x in s],
                   "natural_L6_coords": [round(float(x), 3) for x in c],
                   "svd_signs_flipped": [int(x < 0) for x in signs],
                   "definition": ("h <- h - P_V h + alpha * sum_i s_i v_i at dummy '_' "
                                  "slots, block-6 output; V = top task-unique SVD dirs "
                                  "(meanremoved_top3_bases), sign-fixed toward the "
                                  "task's L6 mean-removed feature"),
                   "conditions": {}}
        arms = {}
        for nshots in (1, 6):
            names = [f"dummy{nshots}_baseline"] + \
                    [f"dummy{nshots}_{tag}_a{a}" for a in ALPHAS]
            if any(cn not in res["conditions"] for cn in names):
                items = (build_items(task, args.prompts_root, tok) if nshots == 1
                         else build_items_6shot(task, args.prompts_root, tok,
                                                real_labels=False))
                for it in items:
                    if "inj_idx_list" not in it:
                        it["inj_idx_list"] = [it["inj_idx"]]
                arms[nshots] = (names, items)

        def run(cname, items, V_used, vec):
            sw.V, sw.vec = V_used, vec
            preds = [None] * len(items)
            for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
                lens = [len(items[i]["ids"]) for i in b]
                L = max(lens)
                ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
                att = torch.zeros(len(b), L, dtype=torch.long)
                mask = torch.zeros(len(b), L, dtype=torch.bool)
                for rr, i in enumerate(b):
                    n = lens[rr]; off = L - n
                    ids[rr, off:] = torch.tensor(items[i]["ids"])
                    att[rr, off:] = 1
                    for p_ in items[i]["inj_idx_list"]:
                        mask[rr, off + p_] = True
                sw.mask = mask.cuda()
                max_new = min(max(items[i]["gold_len"] for i in b) + 3, 16)
                torch.manual_seed(zlib.crc32(f"{task}|{cname}|{bi}".encode()))
                with torch.no_grad():
                    gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                         do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                         max_new_tokens=max_new,
                                         pad_token_id=tok.eos_token_id)
                sw.mask = None
                for rr, i in enumerate(b):
                    preds[i] = tok.decode(gen[rr, ids.shape[1]:],
                                          skip_special_tokens=True).split("\n")[0].strip()
            sw.V = sw.vec = None
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | {cname}: acc={acc:.3f}", flush=True)

        for nshots, (names, items) in arms.items():
            for cname in names:
                if cname in res["conditions"]:
                    continue
                if cname.endswith("_baseline"):
                    run(cname, items, None, None)       # V None -> hook inert
                else:
                    a = float(cname.rsplit("_a", 1)[1])
                    run(cname, items, Vk, (a * r) if a != 0.0 else None)
            with open(out_path, "w") as f:
                json.dump(res, f)
    print("eval done", flush=True)


if __name__ == "__main__":
    main()
