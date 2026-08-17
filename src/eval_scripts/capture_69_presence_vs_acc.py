#!/usr/bin/env python
"""FV-presence vs n-shot accuracy capture (GPU).

For every task and every n in 0..6: truncate the task's 150 fixed 10-shot train prompts
(dataset_files/isolation_prompts_ext) to their FIRST n demos (same queries across n —
paired design). Per prompt:
  (a) forward pass, cos(z_l, v_hat_A) at the query cue token for layers 9..20
      (v_hat_A = unit mean of the task's per-prompt FVs, as in FV_location);
  (b) one temperature-1.0 sampled generation (top_k=0, top_p=1.0, max_new_tokens=12,
      seeded), pred = continuation cut at first newline and stripped, correct iff it
      exactly matches the stripped gold label — the compute_extended_nshot_sampled
      convention.
Writes ARTIFACTS_ROOT/69_task_run/presence_vs_acc/<task>.npz with cos (7,150,12) and
match (7,150). Fan out with --tasks.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from baukit import TraceDict  # noqa: E402
from src.sandbox.isolation_upper_bound.run_task import (  # noqa: E402
    auto_batch, load_records, record_to_prompt_data)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed  # noqa: E402
from src.utils.paths import ARTIFACTS_ROOT  # noqa: E402
from src.utils.prompt_utils import create_prompt  # noqa: E402

N_SHOTS = list(range(0, 7))
LAYERS = list(range(9, 21))
MAX_NEW_TOKENS = 12


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs")
    p.add_argument("--out_root", type=Path, default=ARTIFACTS_ROOT / "69_task_run" / "presence_vs_acc")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capture_batch", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    model, tokenizer, cfg = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hooks = [cfg["layer_hook_names"][l] for l in LAYERS]
    args.out_root.mkdir(parents=True, exist_ok=True)
    split = json.load(open(REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json"))
    group_of = {t: "train" for t in split["train_tasks"]}
    group_of.update({t: "heldout" for t in split["heldout_tasks"]})

    for task in args.tasks:
        out = args.out_root / f"{task}.npz"
        if out.exists():
            print(f"[{task}] exists, skip", flush=True)
            continue

        pp = torch.load(args.fv_root / f"{task}.pt", map_location="cpu", weights_only=False)
        v = pp["fv"].double().mean(dim=0)
        v_hat = (v / v.norm()).float().to(model.device)

        recs = load_records(args, task, "train_prompts")
        assert len(recs) == 150
        cos_out = np.zeros((len(N_SHOTS), 150, len(LAYERS)), dtype=np.float32)
        match_out = np.zeros((len(N_SHOTS), 150), dtype=bool)

        for ni, n in enumerate(N_SHOTS):
            sents, golds = [], []
            for r in recs:
                r_n = dict(r)
                r_n["demos"] = r["demos"][:n]
                sent = create_prompt(record_to_prompt_data(r_n, cfg))
                assert sent.rstrip().endswith("A:")
                sents.append(sent)
                golds.append(str(r["query"]["output"]).strip())
            tok_lens = [len(tokenizer(s).input_ids) for s in sents]
            order = sorted(range(150), key=lambda i: tok_lens[i])

            # --- (a) cos capture: right-padded forward, cue = last real token ---
            tokenizer.padding_side = "right"
            bsz = auto_batch(max(tok_lens), 6000, 64)
            for start in range(0, 150, bsz):
                idx = order[start:start + bsz]
                enc = tokenizer([sents[i] for i in idx], return_tensors="pt", padding=True)
                enc = {k: v_.to(model.device) for k, v_ in enc.items()}
                lens = enc["attention_mask"].sum(dim=1)
                with TraceDict(model, layers=hooks, retain_output=True) as td:
                    model(**enc)
                for li, name in enumerate(hooks):
                    h = td[name].output
                    if isinstance(h, tuple):
                        h = h[0]
                    h = h.float()
                    cue = h[torch.arange(len(idx)), lens - 1]          # (B, 4096)
                    c = (cue @ v_hat) / cue.norm(dim=-1).clamp_min(1e-8)
                    c = c.cpu().numpy()
                    for bi, i in enumerate(idx):
                        cos_out[ni, i, li] = c[bi]

            # --- (b) sampled generation: left-padded, repo nshot convention ---
            tokenizer.padding_side = "left"
            for start in range(0, 150, bsz):
                idx = order[start:start + bsz]
                enc = tokenizer([sents[i] for i in idx], return_tensors="pt", padding=True)
                enc = {k: v_.to(model.device) for k, v_ in enc.items()}
                gen_seed = args.seed * 1_000_003 + zlib.crc32(task.encode()) % 10_007 + n * 101 + start
                torch.manual_seed(gen_seed)
                gen = model.generate(**enc, do_sample=True, temperature=1.0, top_k=0,
                                     top_p=1.0, max_new_tokens=MAX_NEW_TOKENS,
                                     pad_token_id=tokenizer.pad_token_id, use_cache=True)
                cont = gen[:, enc["input_ids"].shape[1]:]
                texts = tokenizer.batch_decode(cont, skip_special_tokens=True)
                for i, text in zip(idx, texts):
                    pred = text.split("\n")[0].strip()
                    match_out[ni, i] = (pred == golds[i])
            print(f"[{task}] n={n} acc={match_out[ni].mean():.3f} "
                  f"cosL13={cos_out[ni, :, LAYERS.index(13)].mean():.3f}", flush=True)

        np.savez(out, cos=cos_out, match=match_out, layers=np.array(LAYERS),
                 n_shots=np.array(N_SHOTS), group=group_of[task])
        print(f"[{task}] done", flush=True)


if __name__ == "__main__":
    main()
