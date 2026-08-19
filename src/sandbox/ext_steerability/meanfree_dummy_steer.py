#!/usr/bin/env python
"""Mean-free read-feature steering on dummy-label scaffolds, L6, all 69 tasks.

USER REQUEST 2026-08-19: steer with the task's raw L6 read feature AFTER removing its
projection along the overall mean read feature — i.e. only the task-unique part:
    v_A = m_A(L6) - (m_A(L6) . s_hat) s_hat,
    s_hat = unit mean of the 55 TRAIN tasks' m_A(L6) (the sweep_raw_mean_layers
            shared-mean control, so "overall mean" matches the study's existing object).
If steering still works, the task-unique component (not the broad shared mean) is what
carries the task signal. Norms halve, so alpha is swept one octave further: {.5,1,2,4,8}.

Arms (both inject additively at block-6 OUTPUT, prefill only, established protocol):
  dummy1: "Q: {in-dist demo input}\nA: _\n\nQ: {query}\nA:", inject at the single ' _'
          (steer_read_dir_methods.build_items — same scaffold as the L6 layer sweep)
  dummy6: six in-distribution demos with '_' labels then the real query, inject at ALL
          six ' _' (sixshot_dummy_steer.build_items_6shot)
Conditions per task: dummy{1,6}_baseline (unsteered; dummy6_baseline seeds/cnames match
the earlier sixshot_dummy run bit-for-bit as an infra cross-check) + 5 alphas each.
Readout: T=1 sampled exact match, 150 prompts/task, established crc32 seeding.

Outputs: artifacts/69_task_run/raw_mean_steering/meanfree_dummy/<task>.json (resumable
per task; missing conditions filled in).
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
    from src.sandbox.ext_steerability.steer_read_dir_methods import Injector, build_items
    from src.sandbox.ext_steerability.sixshot_dummy_steer import build_items_6shot
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from steer_read_dir_1shot import load_model, batches_by_len
    from steer_read_dir_methods import Injector, build_items
    from sixshot_dummy_steer import build_items_6shot

ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
LAYER = 6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resid_means_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "meanfree_dummy")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=24000)
    p.add_argument("--batch_cap", type=int, default=48)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    tasks = sorted(group)[args.shard_idx::args.shard_n]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    args.out_root.mkdir(parents=True, exist_ok=True)

    shared = torch.stack([
        torch.load(args.resid_means_root / f"{t}.pt", map_location="cpu",
                   weights_only=False)["resid_means"]
        for t in split["train_tasks"]]).mean(0)[LAYER].float()      # (4096,)
    s_hat = (shared / shared.norm()).cuda()
    print(f"shared mean over {len(split['train_tasks'])} train tasks: "
          f"||s||={shared.norm():.1f}", flush=True)

    model, tok = load_model(args.model_dir)
    tok.padding_side = "left"
    inj = Injector(model, [LAYER])

    for task in tasks:
        out_path = args.out_root / f"{task}.json"
        res = json.load(open(out_path)) if out_path.exists() else None
        m = torch.load(args.resid_means_root / f"{task}.pt", map_location="cpu",
                       weights_only=False)["resid_means"][LAYER].float().cuda()
        v = m - (m @ s_hat) * s_hat                                  # mean-free vector
        assert abs(float(v @ s_hat)) < 1e-3 * float(m.norm())
        if res is None:
            res = {"task": task, "group": group[task], "n_prompts": 150,
                   "layer": LAYER, "alphas": list(ALPHAS),
                   "norm_m": round(float(m.norm()), 3),
                   "norm_meanfree": round(float(v.norm()), 3),
                   "cos_m_shared": round(float(m @ s_hat / m.norm()), 4),
                   "definition": ("z += alpha * (m_A(L6) - proj_sharedmean m_A(L6)) at "
                                  "dummy '_' label slot(s), block-6 output"),
                   "conditions": {}}
        arms = {}
        for nshots in (1, 6):
            names = [f"dummy{nshots}_baseline"] + \
                    [f"dummy{nshots}_meanfree_a{a}" for a in ALPHAS]
            if any(c not in res["conditions"] for c in names):
                items = (build_items(task, args.prompts_root, tok) if nshots == 1
                         else build_items_6shot(task, args.prompts_root, tok,
                                                real_labels=False))
                for it in items:   # unify the two builders' injection-index fields
                    if "inj_idx_list" not in it:
                        it["inj_idx_list"] = [it["inj_idx"]]
                arms[nshots] = (names, items)

        def run(cname, items, vec):
            inj.vec = vec
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
                    for p_ in items[i]["inj_idx_list"]:
                        mask[r, off + p_] = True
                inj.mask = mask.cuda()
                max_new = min(max(items[i]["gold_len"] for i in b) + 3, 16)
                torch.manual_seed(zlib.crc32(f"{task}|{cname}|{bi}".encode()))
                with torch.no_grad():
                    gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                         do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                         max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
                inj.mask = None
                for r, i in enumerate(b):
                    preds[i] = tok.decode(gen[r, L:],
                                          skip_special_tokens=True).split("\n")[0].strip()
            inj.vec = None
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | {cname}: acc={acc:.3f}", flush=True)

        for nshots, (names, items) in arms.items():
            for cname in names:
                if cname in res["conditions"]:
                    continue
                a = None if cname.endswith("baseline") else float(cname.rsplit("_a", 1)[1])
                run(cname, items, None if a is None else a * v)
            res[f"golds_n{nshots}"] = [it["gold"] for it in items]
        if arms:
            with open(out_path, "w") as f:
                json.dump(res, f)
        else:
            print(f"{task}: all conditions present, skip", flush=True)
    print("eval done", flush=True)


if __name__ == "__main__":
    main()
