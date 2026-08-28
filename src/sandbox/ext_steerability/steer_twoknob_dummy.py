#!/usr/bin/env python
"""Two-knob carrier/task-unique steering on the 6-shot dummy scaffold, L6, 69 tasks.

USER-ADJUDICATED DESIGN 2026-08-28: discriminator for the ratio-preserving composite-code
account of why full-mean steering beats task-unique-only steering. Full two-coordinate
swap at block-6 OUTPUT at every dummy ' _' label slot:

    h <- h - P_span(c_hat, v1) h + a * c_A * c_hat + b * n_A * v1

  c_hat = unit mean of the 69 task-level L6 acts features (label_avg10_L5-15_acts) —
          EXACTLY the mean direction removed when defining the task-unique bases, so the
          two knobs span carrier + top task-unique coordinates of the same decomposition.
  v1    = task's top task-unique SVD dir (meanremoved_top3_bases), sign-fixed toward the
          task's L6 mean-removed feature (same as steer_taskunique_svd.signed_dirs).
  c_A   = <f6, c_hat>   natural carrier coordinate of the task's L6 acts feature f6.
  n_A   = <f6 - proj_c_hat f6, v1>  natural task-unique coordinate (the prior study's
          natural_L6_coords[0], median ~8x s1).

Both knobs are in x-NATURAL units: (a,b)=(1,1) re-imposes the task's natural (carrier,
unique) coordinates; the diagonal scales the composite code without distorting its
internal ratio. Ratio hypothesis predicts the optimum on the diagonal near a=b=4
(full-mean alpha~4 and swap alpha=32 ~ 4x natural), with the b-only edge topping out
near the known swap ceiling (~.34) and the a-only edge near the shared-mean control
(~nothing).

Grid: a, b in {0, 1, 2, 4, 8} (25 conditions; (0,0) = removal-only control) +
dummy6_baseline (bit-identical cross-check vs the sixshot_dummy run). 6-shot arm only.
Readout: T=1 sampled exact match, 150 prompts/task, established crc32 seeding.

Outputs: artifacts/69_task_run/raw_mean_steering/twoknob_dummy/<task>.json
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
    from src.sandbox.ext_steerability.sixshot_dummy_steer import build_items_6shot
    from src.sandbox.ext_steerability.steer_taskunique_svd import ProjSwap, signed_dirs
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from steer_read_dir_1shot import load_model, batches_by_len
    from sixshot_dummy_steer import build_items_6shot
    from steer_taskunique_svd import ProjSwap, signed_dirs

KNOBS = (0.0, 1.0, 2.0, 4.0, 8.0)
LAYER = 6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bases_path", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "bottom_up_ablation" /
                   "meanremoved_top3_bases.pt")
    p.add_argument("--acts_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_avg10_L5-15_acts")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" /
                   "twoknob_dummy")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=11000)
    p.add_argument("--batch_cap", type=int, default=16)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=None)
    return p.parse_args()


def carrier_dir(acts_root, split):
    """Unit all-69 mean of the task-level L6 acts features (the meanremoved bases' mean
    direction at L6) + per-task natural carrier coordinate c_A."""
    all_tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    f6 = {}
    for t in all_tasks:
        d = torch.load(acts_root / f"{t}.pt", map_location="cpu", weights_only=False)
        f6[t] = d["acts"].double().mean(dim=0)[d["layers"].index(LAYER)]
    m = torch.stack([f6[t] for t in all_tasks]).mean(dim=0)
    c_hat = m / m.norm()
    return c_hat, {t: float(f6[t] @ c_hat) for t in all_tasks}


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
    c_hat64, cA = carrier_dir(args.acts_root, split)

    model, tok = load_model(args.model_dir)
    tok.padding_side = "left"
    sw = ProjSwap(model, LAYER)

    for task in tasks:
        out_path = args.out_root / f"{task}.json"
        res = json.load(open(out_path)) if out_path.exists() else None
        V, s, c, signs = dirs[task]
        v1 = V[0].double()
        n_A = float(c[0])
        assert n_A > 0
        leak = float(v1 @ c_hat64)
        # orthonormal 2-dim removal basis spanning (c_hat, v1)
        u2 = v1 - leak * c_hat64
        B = torch.stack([c_hat64, u2 / u2.norm()]).float().cuda()
        c_hat = c_hat64.float().cuda()
        v1f = v1.float().cuda()
        if res is None:
            res = {"task": task, "group": group[task], "n_prompts": 150,
                   "layer": LAYER, "knobs": list(KNOBS),
                   "c_A": round(cA[task], 3), "n_A": round(n_A, 3),
                   "s1": round(float(s[0]), 4), "cos_v1_carrier": round(leak, 4),
                   "definition": ("h <- h - P_span(c_hat,v1) h + a*c_A*c_hat + b*n_A*v1 "
                                  "at dummy '_' slots, block-6 output; c_hat = unit "
                                  "all-69 L6 acts mean; v1 = top task-unique SVD dir "
                                  "sign-fixed; knobs in x-natural units"),
                   "conditions": {}}

        items = None
        names = ["dummy6_baseline"] + \
                [f"dummy6_twoknob_c{a:g}_u{b:g}" for a in KNOBS for b in KNOBS]
        if any(cn not in res["conditions"] for cn in names):
            items = build_items_6shot(task, args.prompts_root, tok, real_labels=False)

        def run(cname, V_used, vec):
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

        for cname in names:
            if cname in res["conditions"]:
                continue
            if cname.endswith("_baseline"):
                run(cname, None, None)                  # V None -> hook inert
            else:
                ab = cname.split("_twoknob_c")[1]
                a, bknob = (float(x) for x in ab.split("_u"))
                vec = a * cA[task] * c_hat + bknob * n_A * v1f
                run(cname, B, vec if (a or bknob) else None)
        with open(out_path, "w") as f:
            json.dump(res, f)
    print("eval done", flush=True)


if __name__ == "__main__":
    main()
