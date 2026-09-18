#!/usr/bin/env python
"""Read-feature capture for the code-convention families (GPU): mean residual activation over the EVIDENCE tokens of a prompt.

Prompt set = exactly the write feature's (capture_cues.py): prompts with k in --ks whose sampled completion was correct (style_ok and
judge.ok), --split train = documents that are not held out (write_split.heldout), unpaired pools per pole. Evidence positions come from
code_evidence.py (read_features/evidence/<family>.json, the rule of DECISIONS 2026-09-18). Averaging (user decision): per prompt the
mean over its evidence tokens, then the mean over prompts. Layers: hidden_states[1..28] (output of block L) plus layer 0.

Output read_features/vectors_<out_tag>/<family>.npz: mean_nat, mean_alt [28, D]; diff = v_nat = mean_nat - mean_alt; mean_*_L0 [D];
n_nat, n_alt (prompts); tokens_per_prompt_{nat,alt} (mean, std); norm_diff, norm_mean [28]; split_half_cos [28] (halves by doc_id).
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
from src.sandbox.style_translation.models import paths as model_paths, arch
from src.sandbox.style_translation.rollout import load_model
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="*", required=True)
    ap.add_argument("--ks", nargs="*", type=int, default=[3, 4]); ap.add_argument("--split", choices=["all", "train"], default="train")
    ap.add_argument("--out_tag", default="k3_train"); ap.add_argument("--token_budget", type=int, default=8000); ap.add_argument("--batch_cap", type=int, default=16)
    args = ap.parse_args()
    MP = model_paths(args.model); OUT = MP["read_features"] / f"vectors_{args.out_tag}"; OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(model=args.model); A = arch(model); D, NL, trunk = A["hidden"], A["n_layers"], A["trunk"]
    from src.sandbox.style_translation.write_split import heldout
    for fam in args.families:
        if (OUT / f"{fam}.npz").exists():
            print(f"{fam}: exists, skip", flush=True); continue
        recs = json.load(open(MP["rollouts"] / f"{fam}.json"))
        keep = {(r["doc_id"], r["style"], r["k"]) for r in recs if r["style_ok"] and r.get("judge") and r["judge"]["ok"] and r["k"] in args.ks}
        if args.split == "train":
            keep = {key for key in keep if not heldout(key[0])}
        ev = {(e["doc_id"], e["pole"], e["k"]): e for e in json.load(open(MP["evidence"] / f"{fam}.json"))}
        items = []; missing = 0; empty = 0
        for p in json.load(open(MP["prompts"] / f"{fam}.json")):
            key = (p["doc_id"], p["style"], p["k"])
            if key not in keep:
                continue
            e = ev.get(key)
            if e is None:
                missing += 1; continue
            assert e["prompt_len"] == len(p["prompt_ids"]), f"{fam} {key}: evidence prompt_len mismatch"
            if not e["idx"]:
                empty += 1; continue
            items.append({"ids": p["prompt_ids"], "pole": p["style"], "doc": p["doc_id"], "idx": e["idx"], "half": zlib.crc32(p["doc_id"].encode()) % 2})
        sums = {(s, h): np.zeros((NL, D), np.float64) for s in ("nat", "alt") for h in (0, 1)}
        sums0 = {s: np.zeros(D, np.float64) for s in ("nat", "alt")}
        cnt = {(s, h): 0 for s in ("nat", "alt") for h in (0, 1)}; ntok = {s: [] for s in ("nat", "alt")}
        for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
            lens = [len(items[i]["ids"]) for i in b]; L = max(lens)
            ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long); att = torch.zeros(len(b), L, dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, L - lens[r]:] = torch.tensor(items[i]["ids"]); att[r, L - lens[r]:] = 1
            with torch.no_grad():
                out = trunk(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
            hs = torch.stack(out.hidden_states[1:NL + 1], 1)        # [B, NL, L, D]
            h0 = out.hidden_states[0]
            for r, i in enumerate(b):
                it = items[i]; off = L - lens[r]; pos = [off + j for j in it["idx"]]
                v = hs[r][:, pos, :].float().mean(1).cpu().numpy()    # per-prompt mean over its evidence tokens  [NL, D]
                key = (it["pole"], it["half"]); sums[key] += v; cnt[key] += 1
                sums0[it["pole"]] += h0[r][pos, :].float().mean(0).cpu().numpy(); ntok[it["pole"]].append(len(pos))
            if (bi + 1) % 25 == 0:
                print(f"{fam}: batch {bi + 1}", flush=True)
        n = {s: cnt[(s, 0)] + cnt[(s, 1)] for s in ("nat", "alt")}
        mean = {s: (sums[(s, 0)] + sums[(s, 1)]) / max(n[s], 1) for s in ("nat", "alt")}
        diff = mean["nat"] - mean["alt"]
        d0 = sums[("nat", 0)] / max(cnt[("nat", 0)], 1) - sums[("alt", 0)] / max(cnt[("alt", 0)], 1)
        d1 = sums[("nat", 1)] / max(cnt[("nat", 1)], 1) - sums[("alt", 1)] / max(cnt[("alt", 1)], 1)
        cos = (d0 * d1).sum(1) / (np.linalg.norm(d0, axis=1) * np.linalg.norm(d1, axis=1) + 1e-9)
        np.savez_compressed(OUT / f"{fam}.npz", mean_nat=mean["nat"].astype(np.float32), mean_alt=mean["alt"].astype(np.float32),
                            diff=diff.astype(np.float32), v_nat=diff.astype(np.float32), n_nat=n["nat"], n_alt=n["alt"],
                            tokens_per_prompt_nat=np.array([np.mean(ntok["nat"] or [0]), np.std(ntok["nat"] or [0])]),
                            tokens_per_prompt_alt=np.array([np.mean(ntok["alt"] or [0]), np.std(ntok["alt"] or [0])]),
                            mean_nat_L0=(sums0["nat"] / max(n["nat"], 1)).astype(np.float32), mean_alt_L0=(sums0["alt"] / max(n["alt"], 1)).astype(np.float32),
                            split_half_cos=cos, norm_diff=np.linalg.norm(diff, axis=1), norm_mean=np.linalg.norm(mean["nat"], axis=1),
                            ks=np.array(args.ks), split=args.split, missing_evidence=missing, empty_evidence=empty)
        print(f"{fam}: n_nat={n['nat']} n_alt={n['alt']} (missing evidence {missing}, empty {empty}) | tokens/prompt nat {np.mean(ntok['nat']):.1f} alt {np.mean(ntok['alt']):.1f} | "
              f"|diff|/|mean| L4,8,12,24: " + ", ".join(f"{np.linalg.norm(diff[l-1])/np.linalg.norm(mean['nat'][l-1]):.3f}" for l in (4, 8, 12, 24))
              + " | split-half cos L4,8,12,24: " + ", ".join(f"{cos[l-1]:.2f}" for l in (4, 8, 12, 24)), flush=True)
    print("capture done", flush=True)


if __name__ == "__main__":
    main()
