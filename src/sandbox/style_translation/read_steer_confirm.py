#!/usr/bin/env python
"""Step 7b — confirm evidence-token steering on all 200 k = 3 texts (GPU).

Arms per family (200 texts each, one seeded T=1 sample, 48 new tokens, sentence cut + capped):
  base_nat, base_alt         alpha = 0 (unsteered k = 3, both context poles)
  nat2alt_top1/2             nat-context prompts, -u_nat at the evidence tokens, top-2 screen settings
  alt2nat_top1/2             alt-context prompts, +u_nat, top-2 screen settings
  nat2alt_cf, alt2nat_cf     ANOTHER family's read vector (cyclic within the shard) at the same positions
                             and the family's own top-1 (L, alpha)
Grading = steps 3/4 (scoring.decide at the 4th decision; judge_rollouts.py --dir for faithfulness).
Records -> read_steer/confirm/<family>.json (step-3 rollout schema + arm/direction/target/layer/alpha).
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
from src.sandbox.style_translation.steer_screen import SCREEN_LAYERS, ALPHAS
from src.sandbox.style_translation.read_steer_screen import k3_items, read_vectors, sample_positions, DIRECTIONS, ROOT

MAX_NEW = 48
FIELDS = ("doc_id", "family", "k", "cue_tok", "seg_prefix", "next_nat", "next_alt", "ref_sentence", "context_tail", "es_text")


def top_settings(fam, direction, n=2):
    recs = json.load(open(ROOT / "screen" / f"{fam}.json"))
    target = DIRECTIONS[direction][1]
    rates = []
    for layer in SCREEN_LAYERS:
        for a in ALPHAS:
            sel = [r for r in recs if r["direction"] == direction and r["layer"] == layer and r["alpha"] == a]
            rates.append((np.mean([r["decision"] == target for r in sel]), -a, -layer, layer, a))
    rates.sort(reverse=True)
    return [(layer, a, rate) for rate, _, _, layer, a in rates[:n]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES], help="this shard's families; cf cycles within")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    (ROOT / "confirm").mkdir(parents=True, exist_ok=True)
    fams = args.families
    model, tok = load_model()
    for fi, fam in enumerate(fams):
        out_path = ROOT / "confirm" / f"{fam}.json"
        if out_path.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        u = read_vectors(fam); cf_fam = fams[(fi + 1) % len(fams)]; u_cf = read_vectors(cf_fam)
        items = {pole: k3_items(fam, pole) for pole in ("nat", "alt")}
        arms = [("base_nat", "nat", None, 0, 0.0, None, None), ("base_alt", "alt", None, 0, 0.0, None, None)]
        for direction, (ctx_pole, target) in DIRECTIONS.items():
            sign = 1 if target == "nat" else -1
            tops = top_settings(fam, direction, 2)
            for rank, (layer, a, _) in enumerate(tops, 1):
                arms.append((f"{direction}_top{rank}", ctx_pole, target, layer, a, u[layer] * sign, None))
            layer, a, _ = tops[0]
            arms.append((f"{direction}_cf", ctx_pole, target, layer, a, u_cf[layer] * sign, cf_fam))
        recs = []
        for arm, ctx_pole, target, layer, alpha, v, cf in arms:
            its = items[ctx_pole]
            for bi in range(0, len(its), args.batch):
                chunk = its[bi:bi + args.batch]
                tails = sample_positions(model, tok, chunk, layer, v, alpha, f"{fam}|rconfirm|{arm}|{bi}", MAX_NEW)
                for it, raw in zip(chunk, tails):
                    cut, capped = cut_sentence(raw)
                    d = decide(fam, it["seg_prefix"], cut, it["next_nat"], it["next_alt"])
                    recs.append({k: it[k] for k in FIELDS} | {"style": target or ctx_pole, "context": ctx_pole, "arm": arm, "direction": arm.split("_")[0] if target else None,
                                 "target": target, "layer": layer, "alpha": alpha, "cf_family": cf, "n_positions": len(it["positions"]),
                                 "tail_raw": raw, "tail": cut, "capped": capped, "decision": d,
                                 "style_ok": (d == target) if target else (d == ctx_pole), "judge": None})
            sel = [r for r in recs if r["arm"] == arm]
            print(f"{fam}: arm {arm:12s} L={layer:2d} a={alpha:<4} | nat {np.mean([r['decision']=='nat' for r in sel]):.2f} "
                  f"alt {np.mean([r['decision']=='alt' for r in sel]):.2f} unscorable {np.mean([r['decision'] is None for r in sel]):.2f} "
                  f"capped {np.mean([r['capped'] for r in sel]):.2f}", flush=True)
        json.dump(recs, open(out_path, "w"), ensure_ascii=False)
    print("read confirm done", flush=True)


if __name__ == "__main__":
    main()
