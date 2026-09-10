#!/usr/bin/env python
"""Step 6b — read-feature capture: mean residual activation at the evidence tokens (GPU).

For every k = 4 prompt (200 texts x 2 poles per family) a forward pass through the transformer trunk
collects hidden_states[1..28] at the evidence token positions (evidence_tokens.py); per prompt the
activations are averaged over ALL evidence tokens of the 4 instances (user decision), and also kept
per instance index k = 0..3. Means over the 200 prompts per pole ->
artifacts/style_translation/read_features/<family>.npz:
  mean_nat, mean_alt [28,4096] fp32; inst_nat, inst_alt [4,28,4096]; n_nat, n_alt;
  tokens_per_prompt_{nat,alt} (mean, std); split_half_cos[28] of (mean_nat - mean_alt) with halves by
  doc_id (paired); norm_diff[28]; norm_mean[28].
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
from src.utils.paths import ARTIFACTS_ROOT
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.rollout import load_model
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len

PROMPTS = ARTIFACTS_ROOT / "style_translation" / "prompts"
EVID = ARTIFACTS_ROOT / "style_translation" / "read_features" / "evidence"
OUT = ARTIFACTS_ROOT / "style_translation" / "read_features"
NL, K = 28, 4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--token_budget", type=int, default=8000)
    ap.add_argument("--batch_cap", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load_model()
    D = model.config.n_embd
    for fam in args.families:
        if (OUT / f"{fam}.npz").exists():
            print(f"{fam}: exists, skip", flush=True); continue
        ev = {(r["doc_id"], r["pole"]): r for r in json.load(open(EVID / f"{fam}.json"))}
        items = []
        for p in json.load(open(PROMPTS / f"{fam}.json")):
            if p["k"] != K:
                continue
            r = ev[(p["doc_id"], p["style"])]
            assert r["prompt_len"] == len(p["prompt_ids"])
            items.append({"ids": p["prompt_ids"], "pole": p["style"], "doc": p["doc_id"],
                          "inst": [e["idx"] for e in r["instances"]], "half": zlib.crc32(p["doc_id"].encode()) % 2})
        sums = {(s, h): np.zeros((NL, D), np.float64) for s in ("nat", "alt") for h in (0, 1)}
        inst = {s: np.zeros((K, NL, D), np.float64) for s in ("nat", "alt")}
        cnt = {(s, h): 0 for s in ("nat", "alt") for h in (0, 1)}
        ntok = {s: [] for s in ("nat", "alt")}
        for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
            lens = [len(items[i]["ids"]) for i in b]; L = max(lens)
            ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros(len(b), L, dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, L - lens[r]:] = torch.tensor(items[i]["ids"]); att[r, L - lens[r]:] = 1
            with torch.no_grad():
                out = model.transformer(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
            hs = torch.stack(out.hidden_states[1:NL + 1], 1)          # [B, 28, L, D] fp16 on GPU
            for r, i in enumerate(b):
                it = items[i]; off = L - lens[r]
                per_inst = [hs[r][:, [off + j for j in idx], :].float().mean(1) for idx in it["inst"]]   # K x [28, D]
                all_idx = [off + j for idx in it["inst"] for j in idx]
                v = hs[r][:, all_idx, :].float().mean(1).cpu().numpy()                                    # [28, D]
                key = (it["pole"], it["half"]); sums[key] += v; cnt[key] += 1
                inst[it["pole"]] += np.stack([x.cpu().numpy() for x in per_inst], 0)
                ntok[it["pole"]].append(len(all_idx))
            if (bi + 1) % 20 == 0:
                print(f"{fam}: batch {bi + 1}", flush=True)
        n = {s: cnt[(s, 0)] + cnt[(s, 1)] for s in ("nat", "alt")}
        mean = {s: (sums[(s, 0)] + sums[(s, 1)]) / max(n[s], 1) for s in ("nat", "alt")}
        diff = mean["nat"] - mean["alt"]
        d0 = sums[("nat", 0)] / max(cnt[("nat", 0)], 1) - sums[("alt", 0)] / max(cnt[("alt", 0)], 1)
        d1 = sums[("nat", 1)] / max(cnt[("nat", 1)], 1) - sums[("alt", 1)] / max(cnt[("alt", 1)], 1)
        cos = (d0 * d1).sum(1) / (np.linalg.norm(d0, axis=1) * np.linalg.norm(d1, axis=1) + 1e-9)
        np.savez_compressed(OUT / f"{fam}.npz", mean_nat=mean["nat"].astype(np.float32), mean_alt=mean["alt"].astype(np.float32),
                            inst_nat=(inst["nat"] / max(n["nat"], 1)).astype(np.float32), inst_alt=(inst["alt"] / max(n["alt"], 1)).astype(np.float32),
                            n_nat=n["nat"], n_alt=n["alt"], tokens_per_prompt_nat=np.array([np.mean(ntok["nat"]), np.std(ntok["nat"])]),
                            tokens_per_prompt_alt=np.array([np.mean(ntok["alt"]), np.std(ntok["alt"])]),
                            split_half_cos=cos, norm_diff=np.linalg.norm(diff, axis=1), norm_mean=np.linalg.norm(mean["nat"], axis=1))
        print(f"{fam}: n_nat={n['nat']} n_alt={n['alt']} | tokens/prompt nat {np.mean(ntok['nat']):.1f} alt {np.mean(ntok['alt']):.1f} | "
              f"|diff|/|mean| L6,12,20: " + ", ".join(f"{np.linalg.norm(diff[l-1])/np.linalg.norm(mean['nat'][l-1]):.3f}" for l in (6, 12, 20))
              + " | split-half cos L6,12,20: " + ", ".join(f"{cos[l-1]:.2f}" for l in (6, 12, 20)), flush=True)
    print("read capture done", flush=True)


if __name__ == "__main__":
    main()
