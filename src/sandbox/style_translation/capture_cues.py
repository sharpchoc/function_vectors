#!/usr/bin/env python
"""Step 4a — mean cue-token activations per family, style and layer (GPU).

Selection (user decision): rollout records whose completion was CORRECT = used the context's
convention AND was judged faithful/coherent (`style_ok and judge.ok`), all k, both styles.
For each such prompt (ids from artifacts/style_translation/prompts) a forward pass collects the
hidden state at the last position (the cue token) for L = 1..28 (= hidden_states[1..28]).

Saves artifacts/style_translation/steering/vectors/<family>.npz:
  mean_nat[28,4096], mean_alt[28,4096] (fp32), v_nat = mean_nat - mean_alt, n_nat, n_alt,
  norm_v[28], norm_mean_nat[28], split_half_cos[28] (v from a random half vs the other half).
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
from src.utils.paths import ARTIFACTS_ROOT
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.rollout import load_model
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len

ROLL = ARTIFACTS_ROOT / "style_translation" / "rollouts"
PROMPTS = ARTIFACTS_ROOT / "style_translation" / "prompts"
OUT = ARTIFACTS_ROOT / "style_translation" / "steering" / "vectors"
NL = 28


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
        recs = json.load(open(ROLL / f"{fam}.json"))
        prompts = {(p["doc_id"], p["style"], p["k"]): p["prompt_ids"] for p in json.load(open(PROMPTS / f"{fam}.json"))}
        items = [{"ids": prompts[(r["doc_id"], r["style"], r["k"])], "style": r["style"],
                  "half": hash((r["doc_id"], r["k"])) % 2}
                 for r in recs if r["style_ok"] and r.get("judge") and r["judge"]["ok"]]
        sums = {(s, h): np.zeros((NL, D), np.float64) for s in ("nat", "alt") for h in (0, 1)}
        cnt = {(s, h): 0 for s in ("nat", "alt") for h in (0, 1)}
        for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
            lens = [len(items[i]["ids"]) for i in b]; L = max(lens)
            ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros(len(b), L, dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, L - lens[r]:] = torch.tensor(items[i]["ids"]); att[r, L - lens[r]:] = 1
            with torch.no_grad():
                # transformer trunk only: skips the fp32 vocab logits that OOM at this batch size
                out = model.transformer(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
            hs = torch.stack([out.hidden_states[l][:, -1, :] for l in range(1, NL + 1)], 1).float().cpu().numpy()  # [B, 28, D]
            for r, i in enumerate(b):
                key = (items[i]["style"], items[i]["half"])
                sums[key] += hs[r]; cnt[key] += 1
            if (bi + 1) % 100 == 0:
                print(f"{fam}: batch {bi + 1}", flush=True)
        mean = {s: (sums[(s, 0)] + sums[(s, 1)]) / max(cnt[(s, 0)] + cnt[(s, 1)], 1) for s in ("nat", "alt")}
        v = mean["nat"] - mean["alt"]
        v0 = sums[("nat", 0)] / max(cnt[("nat", 0)], 1) - sums[("alt", 0)] / max(cnt[("alt", 0)], 1)
        v1 = sums[("nat", 1)] / max(cnt[("nat", 1)], 1) - sums[("alt", 1)] / max(cnt[("alt", 1)], 1)
        cos = (v0 * v1).sum(1) / (np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1) + 1e-9)
        np.savez_compressed(OUT / f"{fam}.npz", mean_nat=mean["nat"].astype(np.float32), mean_alt=mean["alt"].astype(np.float32),
                            v_nat=v.astype(np.float32), n_nat=cnt[("nat", 0)] + cnt[("nat", 1)], n_alt=cnt[("alt", 0)] + cnt[("alt", 1)],
                            norm_v=np.linalg.norm(v, axis=1), norm_mean_nat=np.linalg.norm(mean["nat"], axis=1), split_half_cos=cos)
        print(f"{fam}: n_nat={cnt[('nat',0)]+cnt[('nat',1)]} n_alt={cnt[('alt',0)]+cnt[('alt',1)]} | "
              f"|v|/|mean| at L=6,12,20: " + ", ".join(f"{np.linalg.norm(v[l-1])/np.linalg.norm(mean['nat'][l-1]):.3f}" for l in (6, 12, 20)) +
              f" | split-half cos at L=6,12,20: " + ", ".join(f"{cos[l-1]:.2f}" for l in (6, 12, 20)), flush=True)
    print("capture done", flush=True)


if __name__ == "__main__":
    main()
