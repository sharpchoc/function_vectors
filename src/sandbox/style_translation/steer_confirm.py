#!/usr/bin/env python
"""Step 4c — confirm the shortlisted steering settings on all 200 k=0 texts (GPU).

Arms per family (each 200 texts, one seeded T=1 sample, 48 new tokens, sentence cut + capped):
  base            alpha = 0 (unsteered, same GPU/seeds -> in-run baseline)
  nat_top1/2      steer toward nat with the family's own v_nat at its top-2 screen settings
  alt_top1/2      steer toward alt with -v_nat at its top-2 screen settings
  nat_cf, alt_cf  counterfactual control: the vector of ANOTHER family (cyclic within the shard's
                  family list, same style role) at the family's own top-1 (L, alpha)
Grading = exactly as step 3: style at the cue via scoring.decide, faithfulness/coherence by the
Gemini judge (judge_rollouts.py --dir ...). Records -> artifacts/style_translation/steering/
confirm/<family>.json (same record schema as the step-3 rollouts + arm/layer/alpha/target).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.rollout import load_model
from src.sandbox.style_translation.scoring import decide, cut_sentence
from src.sandbox.style_translation.steer_screen import k0_items, sample, SCREEN_LAYERS, ALPHAS

VEC = ARTIFACTS_ROOT / "style_translation" / "steering" / "vectors"
SCREEN = ARTIFACTS_ROOT / "style_translation" / "steering" / "screen"
OUT = ARTIFACTS_ROOT / "style_translation" / "steering" / "confirm"
MAX_NEW = 48


def top_settings(fam, target, n=2):
    recs = json.load(open(SCREEN / f"{fam}.json"))
    rates = []
    for layer in SCREEN_LAYERS:
        for a in ALPHAS:
            sel = [r for r in recs if r["target"] == target and r["layer"] == layer and r["alpha"] == a]
            rates.append((np.mean([r["decision"] == target for r in sel]), -a, -layer, layer, a))
    rates.sort(reverse=True)
    return [(layer, a, rate) for rate, _, _, layer, a in rates[:n]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES],
                    help="families of THIS shard; the counterfactual vector cycles within this list")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fams = args.families
    model, tok = load_model()
    for fi, fam in enumerate(fams):
        out_path = OUT / f"{fam}.json"
        if out_path.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        items = k0_items(fam)
        vecs = np.load(VEC / f"{fam}.npz")["v_nat"]
        cf_fam = fams[(fi + 1) % len(fams)]
        cf_vecs = np.load(VEC / f"{cf_fam}.npz")["v_nat"]
        arms = [("base", None, 0, 0.0, None)]
        for target in ("nat", "alt"):
            sign = 1 if target == "nat" else -1
            tops = top_settings(fam, target, 2)
            for rank, (layer, a, rate) in enumerate(tops, 1):
                arms.append((f"{target}_top{rank}", target, layer, a, vecs[layer - 1] * sign))
            layer, a, _ = tops[0]
            arms.append((f"{target}_cf", target, layer, a, cf_vecs[layer - 1] * sign))
        recs = []
        for arm, target, layer, alpha, v in arms:
            for bi in range(0, len(items), args.batch):
                chunk = items[bi:bi + args.batch]
                tails = sample(model, tok, chunk, layer, v, alpha, f"{fam}|confirm|{arm}|{bi}", MAX_NEW)
                for it, raw in zip(chunk, tails):
                    cut, capped = cut_sentence(raw)
                    d = decide(fam, it["seg_prefix"], cut, it["next_nat"], it["next_alt"])
                    recs.append({k: it[k] for k in ("doc_id", "family", "k", "cue_tok", "seg_prefix", "next_nat",
                                                    "next_alt", "ref_sentence", "context_tail", "es_text")}
                                | {"style": target or "none", "arm": arm, "layer": layer, "alpha": alpha,
                                   "cf_family": cf_fam if arm.endswith("_cf") else None,
                                   "tail_raw": raw, "tail": cut, "capped": capped, "decision": d,
                                   "style_ok": (d == target) if target else None, "judge": None})
            sel = [r for r in recs if r["arm"] == arm]
            print(f"{fam}: arm {arm:9s} L={layer:2d} a={alpha:<4} | nat {np.mean([r['decision']=='nat' for r in sel]):.2f} "
                  f"alt {np.mean([r['decision']=='alt' for r in sel]):.2f} unscorable {np.mean([r['decision'] is None for r in sel]):.2f} "
                  f"capped {np.mean([r['capped'] for r in sel]):.2f}", flush=True)
        json.dump(recs, open(out_path, "w"), ensure_ascii=False)
    print("confirm done", flush=True)


if __name__ == "__main__":
    main()
