#!/usr/bin/env python
"""Per-prompt read / write features for the read->write map (GPU), code-convention families.

Prompts: k in --ks whose sampled completion was correct (style_ok and judge.ok), both poles, ALL documents (the map is fitted on the
training documents and tested on the held-out ones via write_split.heldout). Per prompt, from one forward pass:
  read  = mean over the prompt's evidence tokens (code_evidence.py) of hidden_states[READ_L]   (read feature site, layer 8)
  write = hidden_states[WRITE_L] at the final cue token (last prompt position)                 (write feature site, layer 24)
Output prompt_pairs/<family>.npz: read [N, D] fp16, write [N, D] fp16, doc_id, pole, k, heldout, n_evidence, read_layer, write_layer.
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
    ap.add_argument("--ks", nargs="*", type=int, default=[3, 4]); ap.add_argument("--read_layer", type=int, default=8); ap.add_argument("--write_layer", type=int, default=24)
    ap.add_argument("--token_budget", type=int, default=8000); ap.add_argument("--batch_cap", type=int, default=16)
    ap.add_argument("--select", choices=["correct", "incorrect", "all"], default="correct", help="correct = convention followed AND judge OK (default); incorrect = the rest; all = every prompt")
    ap.add_argument("--out_name", default=None, help="output folder name under the model's artifacts (default prompt_pairs)")
    args = ap.parse_args()
    MP = model_paths(args.model); OUT = MP["prompt_pairs"] if args.out_name is None else MP["prompt_pairs"].parent / args.out_name; OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(model=args.model); A = arch(model); trunk = A["trunk"]
    for fam in args.families:
        if (OUT / f"{fam}.npz").exists():
            print(f"{fam}: exists, skip", flush=True); continue
        recs = json.load(open(MP["rollouts"] / f"{fam}.json"))
        flags = {(r["doc_id"], r["style"], r["k"]): (bool(r["style_ok"]), bool((r.get("judge") or {}).get("ok"))) for r in recs if r["k"] in args.ks}
        keep = {key for key, (so, jo) in flags.items() if args.select == "all" or (args.select == "correct") == (so and jo)}
        ev = {(e["doc_id"], e["pole"], e["k"]): e for e in json.load(open(MP["evidence"] / f"{fam}.json"))}
        items = []
        for p in json.load(open(MP["prompts"] / f"{fam}.json")):
            key = (p["doc_id"], p["style"], p["k"])
            if key not in keep or key not in ev or not ev[key]["idx"]:
                continue
            assert ev[key]["prompt_len"] == len(p["prompt_ids"])
            items.append({"ids": p["prompt_ids"], "doc": p["doc_id"], "pole": p["style"], "k": p["k"], "idx": ev[key]["idx"], "so": flags[key][0], "jo": flags[key][1]})
        R = np.zeros((len(items), A["hidden"]), np.float16); W = np.zeros((len(items), A["hidden"]), np.float16)
        for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
            lens = [len(items[i]["ids"]) for i in b]; L = max(lens)
            ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long); att = torch.zeros(len(b), L, dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, L - lens[r]:] = torch.tensor(items[i]["ids"]); att[r, L - lens[r]:] = 1
            with torch.no_grad():
                out = trunk(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
            hr = out.hidden_states[args.read_layer]; hw = out.hidden_states[args.write_layer]
            for r, i in enumerate(b):
                off = L - lens[r]; pos = [off + j for j in items[i]["idx"]]
                R[i] = hr[r, pos, :].float().mean(0).cpu().numpy().astype(np.float16); W[i] = hw[r, -1, :].float().cpu().numpy().astype(np.float16)
            if (bi + 1) % 25 == 0:
                print(f"{fam}: batch {bi + 1}", flush=True)
        np.savez_compressed(OUT / f"{fam}.npz", read=R, write=W, doc_id=np.array([it["doc"] for it in items]), pole=np.array([it["pole"] for it in items]),
                            k=np.array([it["k"] for it in items]), heldout=np.array([heldout(it["doc"]) for it in items]), n_evidence=np.array([len(it["idx"]) for it in items]),
                            style_ok=np.array([it["so"] for it in items]), judge_ok=np.array([it["jo"] for it in items]),
                            read_layer=args.read_layer, write_layer=args.write_layer, ks=np.array(args.ks), select=args.select)
        n_nat = sum(it["pole"] == "nat" for it in items); n_ho = sum(heldout(it["doc"]) for it in items)
        print(f"{fam}: {len(items)} prompts (nat {n_nat}, alt {len(items) - n_nat}; held-out {n_ho}) -> read L{args.read_layer}, write L{args.write_layer}", flush=True)
    print("capture done", flush=True)


if __name__ == "__main__":
    main()
