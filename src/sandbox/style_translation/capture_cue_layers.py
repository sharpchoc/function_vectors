#!/usr/bin/env python
"""Per-prompt cue-token activations of paired k-shot prompts at several layers (GPU), code-convention families.

Every document has a natural-convention and an alternative-convention k-shot prompt (same documents, same cue token; only the convention
differs). For each pair, one forward pass per prompt; the hidden state at the last prompt position (the cue token) is stored for each
layer in --layers (L = hidden_states[L], output of block L-1; hidden_states[28] has the final norm applied, so the un-normed output of
the last block is stored separately as `*_last_prenorm`). No selection on correctness: all documents, flags are stored.
Output cue_pairs_layers/<family>_k<k>.npz:
  act_nat, act_alt [N_docs, n_layers, D] fp32; diff = act_nat - act_alt; nat_last_prenorm, alt_last_prenorm [N_docs, D];
  doc_id, heldout, cue_tok, correct_nat, correct_alt (convention followed AND judge OK in the stored rollouts), layers, k.
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
from src.sandbox.style_translation.write_split import heldout
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="*", required=True)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--layers", nargs="*", type=int, default=list(range(20, 29)))
    ap.add_argument("--token_budget", type=int, default=8000); ap.add_argument("--batch_cap", type=int, default=16)
    args = ap.parse_args()
    MP = model_paths(args.model); OUT = MP["prompt_pairs"].parent / "cue_pairs_layers"; OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(model=args.model); A = arch(model); trunk = A["trunk"]; NLy = len(args.layers)
    last = {}
    hook = A["blocks"][-1].register_forward_hook(lambda m, i, o: last.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[:, -1, :].float().cpu().numpy()))
    for fam in args.families:
        recs = json.load(open(MP["rollouts"] / f"{fam}.json"))
        ok = {(r["doc_id"], r["style"]): bool(r["style_ok"]) and bool((r.get("judge") or {}).get("ok")) for r in recs if r["k"] == args.k}
        P = {(p["doc_id"], p["style"]): p for p in json.load(open(MP["prompts"] / f"{fam}.json")) if p["k"] == args.k}
        docs = sorted({d for d, _ in P}); assert all((d, s) in P for d in docs for s in ("nat", "alt"))
        for d in docs:
            assert P[(d, "nat")]["cue_tok"] == P[(d, "alt")]["cue_tok"] and P[(d, "nat")]["prompt_ids"][-1] == P[(d, "alt")]["prompt_ids"][-1], d
        items = [{"ids": P[(d, s)]["prompt_ids"], "di": i, "pole": s} for i, d in enumerate(docs) for s in ("nat", "alt")]
        act = {s: np.zeros((len(docs), NLy, A["hidden"]), np.float32) for s in ("nat", "alt")}
        pre = {s: np.zeros((len(docs), A["hidden"]), np.float32) for s in ("nat", "alt")}
        for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
            lens = [len(items[i]["ids"]) for i in b]; L = max(lens)
            ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long); att = torch.zeros(len(b), L, dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, L - lens[r]:] = torch.tensor(items[i]["ids"]); att[r, L - lens[r]:] = 1
            with torch.no_grad():
                out = trunk(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
            hs = torch.stack([out.hidden_states[l][:, -1, :] for l in args.layers], 1).float().cpu().numpy()
            for r, i in enumerate(b):
                act[items[i]["pole"]][items[i]["di"]] = hs[r]; pre[items[i]["pole"]][items[i]["di"]] = last["h"][r]
            if (bi + 1) % 10 == 0:
                print(f"{fam}: batch {bi + 1}", flush=True)
        np.savez_compressed(OUT / f"{fam}_k{args.k}.npz", act_nat=act["nat"], act_alt=act["alt"], diff=act["nat"] - act["alt"],
                            nat_last_prenorm=pre["nat"], alt_last_prenorm=pre["alt"], doc_id=np.array(docs), heldout=np.array([heldout(d) for d in docs]),
                            cue_tok=np.array([P[(d, "nat")]["cue_tok"] for d in docs]), correct_nat=np.array([ok.get((d, "nat"), False) for d in docs]),
                            correct_alt=np.array([ok.get((d, "alt"), False) for d in docs]), layers=np.array(args.layers), k=args.k)
        print(f"{fam}: {len(docs)} pairs x layers {args.layers} -> {OUT / f'{fam}_k{args.k}.npz'}", flush=True)
    hook.remove(); print("capture done", flush=True)


if __name__ == "__main__":
    main()
