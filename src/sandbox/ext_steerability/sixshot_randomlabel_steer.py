#!/usr/bin/env python
"""Raw-mean steering on a 6-SHOT RANDOM-label scaffold, all label tokens, L6 (69 tasks).

Variant of sixshot_dummy_steer.py: instead of '_' every demo output is a WRONG label sampled
from a different task's output pool, so the labels are mixed and random:
    Q: {in1}\nA: {rand1}\n\n ... Q: {in6}\nA: {rand6}\n\nQ: {query}\nA:
Sampling (deterministic): per (task, record, slot) a seeded RNG picks one of the other 68
tasks uniformly, then one of that task's distinct output strings uniformly; redraw while the
sample equals the demo's TRUE output. Collisions with the query gold are allowed but counted
(n_label_eq_gold) for later analysis.

Steering: z <- z + alpha * m_A(L6) additively at EVERY token of all six random labels at
block-6 output (m_A = the task's L6 label-token mean, the raw_mean_steering vector),
alpha {0.5,1,2,4} — user-adjudicated 2026-08-19 (all label tokens, independent per slot).

Conditions: random 6-shot unsteered | steered x4 alphas. real6/real1/0-shot references are
reused from the sixshot_dummy run at plot time.

Readout: T=1 sampled exact match, 150 prompts/task, established seeding.
Outputs: artifacts/69_task_run/raw_mean_steering/sixshot_randomlabel/<task>.json
"""
import argparse
import json
import random
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
    from src.sandbox.ext_steerability.steer_read_dir_methods import Injector
except ModuleNotFoundError:  # staged copy outside the repo tree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from steer_read_dir_1shot import load_model, batches_by_len
    from steer_read_dir_methods import Injector

ALPHAS = (0.5, 1.0, 2.0, 4.0)
LAYER = 6
N_SHOTS = 6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resid_means_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "label_resid_means")
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "raw_mean_steering" / "sixshot_randomlabel")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=9000)
    p.add_argument("--batch_cap", type=int, default=12)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def _norm_out(o):
    return str(o[0] if isinstance(o, list) else o).strip()


def build_output_pools(all_tasks, prompts_root):
    """task -> sorted list of its distinct output strings (demo + query outputs)."""
    pools = {}
    for t in all_tasks:
        recs = json.load(open(prompts_root / t / "train_prompts.json"))
        outs = set()
        for rec in recs:
            outs.add(_norm_out(rec["query"]["output"]))
            for d in rec["demos"]:
                outs.add(_norm_out(d["output"]))
        pools[t] = sorted(o for o in outs if o)
        assert pools[t], f"{t}: empty output pool"
    return pools


def build_items_randomlabel(task, prompts_root, tok, pools):
    """6-shot scaffold with independently sampled wrong-task labels; injection indices
    cover EVERY token of each label (structural anchoring via growing-prefix tokenization)."""
    other_tasks = sorted(t for t in pools if t != task)
    recs = json.load(open(prompts_root / task / "train_prompts.json"))
    assert len(recs) == 150
    items = []
    for ri, rec in enumerate(recs):
        q = str(rec["query"]["input"])
        gold = _norm_out(rec["query"]["output"])
        demos = rec["demos"][:N_SHOTS]
        assert len(demos) == N_SHOTS
        rand_labels = []
        for si, d in enumerate(demos):
            true_out = _norm_out(d["output"])
            rng = random.Random(zlib.crc32(f"{task}|{ri}|{si}".encode()))
            while True:
                src = rng.choice(other_tasks)
                lab = rng.choice(pools[src])
                if lab != true_out:
                    break
            rand_labels.append(lab)
        prompt = "".join(f"Q: {str(d['input'])}\nA: {lab}\n\n"
                         for d, lab in zip(demos, rand_labels)) + f"Q: {q}\nA:"
        ids = tok(prompt).input_ids
        # anchor each label span STRUCTURALLY: tokenize the growing prefix up to "...A:"
        # (span start) and up to "...A: {label}" (span end), asserting at both stages that
        # the prefix tokenization is a prefix of the full prompt's ids (guards BPE merges
        # across the label/"\n\n" boundary; labels are stripped so none end in whitespace).
        inj = []
        prefix = ""
        for d, lab in zip(demos, rand_labels):
            di = str(d["input"])
            assert di != q
            prefix += f"Q: {di}\nA:"
            pre_ids = tok(prefix).input_ids
            start = len(pre_ids)
            assert ids[:start] == pre_ids, f"{task}#{ri}: cue-prefix tokenization mismatch"
            prefix += f" {lab}"
            lab_ids = tok(prefix).input_ids
            end = len(lab_ids)
            assert end > start and ids[:end] == lab_ids, \
                f"{task}#{ri}: label-span tokenization mismatch"
            inj.extend(range(start, end))
            prefix += "\n\n"
        items.append({"ids": ids, "inj_idx_list": inj, "gold": gold,
                      "gold_len": len(tok(" " + gold).input_ids),
                      "rand_labels": rand_labels,
                      "n_label_eq_gold": sum(l == gold for l in rand_labels)})
    return items


def main():
    args = parse_args()
    split = json.load(open(args.split_path))
    group = {t: "train" for t in split["train_tasks"]}
    group.update({t: "heldout" for t in split["heldout_tasks"]})
    all_tasks = sorted(group)
    tasks = all_tasks[args.shard_idx::args.shard_n]
    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} tasks on this shard", flush=True)
    pools = build_output_pools(all_tasks, args.prompts_root)

    model, tok = load_model(args.model_dir)
    tok.padding_side = "left"
    inj = Injector(model, [LAYER])

    for task in tasks:
        out_path = args.out_root / f"{task}.json"
        if out_path.exists():
            print(f"{task}: exists, skip", flush=True)
            continue
        items = build_items_randomlabel(task, args.prompts_root, tok, pools)
        m = torch.load(args.resid_means_root / f"{task}.pt", map_location="cpu",
                       weights_only=False)["resid_means"][LAYER].float().cuda()
        res = {"task": task, "group": group[task], "n_prompts": len(items),
               "n_shots": N_SHOTS, "layer": LAYER, "norm_m": float(m.norm()),
               "definition": "z += alpha*m_A(L6) at EVERY token of all six random wrong-task labels",
               "rand_labels": [it["rand_labels"] for it in items],
               "n_label_eq_gold": [it["n_label_eq_gold"] for it in items],
               "conditions": {}}

        def run(cname, vec):
            inj.vec = None if vec is None else vec
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
                    preds[i] = tok.decode(gen[r, L:], skip_special_tokens=True).split("\n")[0].strip()
            inj.vec = None
            acc = float(np.mean([p == it["gold"] for p, it in zip(preds, items)]))
            res["conditions"][cname] = {"acc": round(acc, 4), "preds": preds}
            print(f"{task} | {cname}: acc={acc:.3f}", flush=True)

        run("random6_baseline", None)
        for a in ALPHAS:
            run(f"random6_steer_a{a}", a * m)
        res["golds"] = [it["gold"] for it in items]
        with open(out_path, "w") as f:
            json.dump(res, f)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
