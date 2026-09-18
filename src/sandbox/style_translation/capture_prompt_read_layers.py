#!/usr/bin/env python
"""Per-prompt READ features at several layers (GPU), companion to prompt_pairs/<family>.npz.

Same prompts (every k = 3/4 prompt, both poles, all documents), same recipe as the L8 read feature: mean over the prompt's evidence
tokens (code_evidence.py) of hidden_states[L], fp16. Rows are written in the SAME order as prompt_pairs/<family>.npz (asserted on
doc_id / pole / k), so read_L{L} can be used as a drop-in input feature. Output prompt_pairs_layers/<family>.npz:
read_L6, read_L8, read_L10, ... [N, D] fp16, doc_id, pole, k, layers. L8 is recomputed for an alignment check against prompt_pairs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.models import paths as model_paths, arch
from src.sandbox.style_translation.rollout import load_model
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="*", required=True)
    ap.add_argument("--layers", nargs="*", type=int, default=[6, 8, 10, 12, 14, 16, 18])
    ap.add_argument("--token_budget", type=int, default=8000); ap.add_argument("--batch_cap", type=int, default=16)
    args = ap.parse_args()
    MP = model_paths(args.model); OUT = MP["prompt_pairs"].parent / "prompt_pairs_layers"; OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(model=args.model); A = arch(model); trunk = A["trunk"]
    for fam in args.families:
        if (OUT / f"{fam}.npz").exists():
            print(f"{fam}: exists, skip", flush=True); continue
        base = np.load(MP["prompt_pairs"] / f"{fam}.npz")                       # row order to reproduce
        order = {(d, p, int(k)): i for i, (d, p, k) in enumerate(zip(base["doc_id"], base["pole"], base["k"]))}
        ev = {(e["doc_id"], e["pole"], e["k"]): e for e in json.load(open(MP["evidence"] / f"{fam}.json"))}
        items = [None] * len(order)
        for p in json.load(open(MP["prompts"] / f"{fam}.json")):
            key = (p["doc_id"], p["style"], p["k"])
            if key in order:
                items[order[key]] = {"ids": p["prompt_ids"], "idx": ev[key]["idx"]}
        assert all(it is not None for it in items), f"{fam}: prompt_pairs rows without a prompt"
        R = {L: np.zeros((len(items), A["hidden"]), np.float16) for L in args.layers}
        for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
            lens = [len(items[i]["ids"]) for i in b]; Lmax = max(lens)
            ids = torch.full((len(b), Lmax), tok.eos_token_id, dtype=torch.long); att = torch.zeros(len(b), Lmax, dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, Lmax - lens[r]:] = torch.tensor(items[i]["ids"]); att[r, Lmax - lens[r]:] = 1
            with torch.no_grad():
                out = trunk(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
            for r, i in enumerate(b):
                pos = [Lmax - lens[r] + j for j in items[i]["idx"]]
                for L in args.layers:
                    R[L][i] = out.hidden_states[L][r, pos, :].float().mean(0).cpu().numpy().astype(np.float16)
            if (bi + 1) % 25 == 0:
                print(f"{fam}: batch {bi + 1}", flush=True)
        if 8 in args.layers:                      # alignment check: per-row cosine with the stored L8 read feature (bf16 reruns differ slightly in value, not in direction)
            a = R[8].astype(np.float32); b = base["read"].astype(np.float32)
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
            print(f"{fam}: L8 alignment check: per-row cosine min {cos.min():.4f} mean {cos.mean():.5f}", flush=True); assert cos.min() > 0.99, f"{fam}: L8 rows do not match prompt_pairs (min cos {cos.min():.3f})"
        np.savez_compressed(OUT / f"{fam}.npz", **{f"read_L{L}": R[L] for L in args.layers}, doc_id=base["doc_id"], pole=base["pole"], k=base["k"], layers=np.array(args.layers))
        print(f"{fam}: {len(items)} prompts, layers {args.layers}", flush=True)
    print("capture done", flush=True)


if __name__ == "__main__":
    main()
