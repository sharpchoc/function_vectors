#!/usr/bin/env python
"""Per-prompt features at EVERY layer and three sites (GPU), companion to prompt_pairs/<family>.npz (2026-11-01, map layer sweep).

Same prompts and row order as prompt_pairs/<family>.npz (asserted on doc_id / pole / k). Per prompt, one forward pass, hidden_states[L]
for L = 0..28 (0 = embedding output; 28 has the final norm applied) at three sites:
  ev_mean  mean over the prompt's evidence tokens (the read-feature recipe; L8 = prompt_pairs 'read')
  ev_last  the last evidence token
  cue      the final cue token (the write-feature site; L24 = prompt_pairs 'write')
Output prompt_pairs_sites/<family>.npz (uncompressed): ev_mean, ev_last, cue [N, 29, D] fp16, doc_id, pole, k.
Alignment checks: per-row cosine of ev_mean[:, 8] with prompt_pairs 'read' and cue[:, 24] with 'write' must exceed .99.
"""
import argparse
import json
import os
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

SITES = ("ev_mean", "ev_last", "cue")


def rowcos(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="*", required=True)
    ap.add_argument("--token_budget", type=int, default=8000); ap.add_argument("--batch_cap", type=int, default=16)
    args = ap.parse_args()
    MP = model_paths(args.model); OUT = MP["prompt_pairs"].parent / "prompt_pairs_sites"; OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(model=args.model); A = arch(model); trunk = A["trunk"]; NL = A["n_layers"] + 1
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
        F = {s: np.zeros((len(items), NL, A["hidden"]), np.float16) for s in SITES}
        for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
            lens = [len(items[i]["ids"]) for i in b]; Lmax = max(lens)
            ids = torch.full((len(b), Lmax), tok.eos_token_id, dtype=torch.long); att = torch.zeros(len(b), Lmax, dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, Lmax - lens[r]:] = torch.tensor(items[i]["ids"]); att[r, Lmax - lens[r]:] = 1
            with torch.no_grad():
                hs = torch.stack(trunk(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True).hidden_states, 1)   # [B, NL, T, D]
            for r, i in enumerate(b):
                pos = [Lmax - lens[r] + j for j in items[i]["idx"]]
                F["ev_mean"][i] = hs[r, :, pos, :].float().mean(1).cpu().numpy().astype(np.float16)
                F["ev_last"][i] = hs[r, :, pos[-1], :].float().cpu().numpy().astype(np.float16)
                F["cue"][i] = hs[r, :, -1, :].float().cpu().numpy().astype(np.float16)
            del hs
            if (bi + 1) % 25 == 0:
                print(f"{fam}: batch {bi + 1}", flush=True)
        c_r = rowcos(F["ev_mean"][:, 8], base["read"]); c_w = rowcos(F["cue"][:, 24], base["write"])
        print(f"{fam}: alignment read L8 min cos {c_r.min():.4f}, write L24 min cos {c_w.min():.4f}", flush=True)
        assert os.environ.get("ALIGN_SKIP") or (c_r.min() > 0.99 and c_w.min() > 0.99), f"{fam}: rows do not match prompt_pairs"
        np.savez(OUT / f"{fam}.npz", **F, doc_id=base["doc_id"], pole=base["pole"], k=base["k"])
        print(f"{fam}: {len(items)} prompts, sites {SITES}, layers 0..{NL - 1}", flush=True)
    print("capture done", flush=True)


if __name__ == "__main__":
    main()
