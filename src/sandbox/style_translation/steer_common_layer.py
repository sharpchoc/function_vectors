#!/usr/bin/env python
"""Step 4e — can every family steer from ONE common layer? (GPU)

User question (2026-09-10): fix the injection layer for all families (alpha still per family) and ask
whether each family stays within ±0.03 of the accuracy it reached at its own best (layer, alpha).
The screen (style-only) points at L24 as the only candidate. This script samples, for every family
and both targets, all alpha in ALPHAS at --layer on the 200 k = 0 texts (48 tokens, sentence cut,
same records as steer_confirm). Arms already sampled by steer_confirm at exactly this (layer, alpha,
target) are copied over WITH their judge verdicts instead of re-sampled.
Records -> artifacts/style_translation/steering/common_layer/<family>.json, arm = "<target>_L<layer>_a<alpha>".
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
from src.sandbox.style_translation.steer_screen import k0_items, sample, ALPHAS

VEC = ARTIFACTS_ROOT / "style_translation" / "steering" / "vectors"
CONFIRM = ARTIFACTS_ROOT / "style_translation" / "steering" / "confirm"
OUT = ARTIFACTS_ROOT / "style_translation" / "steering" / "common_layer"
MAX_NEW = 48


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    L = args.layer
    model, tok = load_model()
    for fam in args.families:
        out_path = OUT / f"{fam}.json"
        if out_path.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        items = k0_items(fam)
        vecs = np.load(VEC / f"{fam}.npz")["v_nat"]
        conf = json.load(open(CONFIRM / f"{fam}.json"))
        recs = []
        for target in ("nat", "alt"):
            sign = 1 if target == "nat" else -1
            for a in ALPHAS:
                arm = f"{target}_L{L}_a{a:g}"
                reuse = [r for r in conf if r["arm"].startswith(f"{target}_top") and r["layer"] == L and r["alpha"] == a]
                if len(reuse) == len(items):
                    for r in reuse:
                        recs.append(dict(r, arm=arm, reused_from=r["arm"]))
                    src = f"reused {reuse[0]['arm']}"
                else:
                    v = vecs[L - 1] * sign
                    for bi in range(0, len(items), args.batch):
                        chunk = items[bi:bi + args.batch]
                        tails = sample(model, tok, chunk, L, v, a, f"{fam}|common|{arm}|{bi}", MAX_NEW)
                        for it, raw in zip(chunk, tails):
                            cut, capped = cut_sentence(raw)
                            d = decide(fam, it["seg_prefix"], cut, it["next_nat"], it["next_alt"])
                            recs.append({k: it[k] for k in ("doc_id", "family", "k", "cue_tok", "seg_prefix", "next_nat",
                                                            "next_alt", "ref_sentence", "context_tail", "es_text")}
                                        | {"style": target, "arm": arm, "layer": L, "alpha": a, "cf_family": None, "reused_from": None,
                                           "tail_raw": raw, "tail": cut, "capped": capped, "decision": d,
                                           "style_ok": d == target, "judge": None})
                    src = "sampled"
                sel = [r for r in recs if r["arm"] == arm]
                print(f"{fam}: arm {arm:12s} ({src}) | nat {np.mean([r['decision']=='nat' for r in sel]):.2f} "
                      f"alt {np.mean([r['decision']=='alt' for r in sel]):.2f} unscorable {np.mean([r['decision'] is None for r in sel]):.2f} "
                      f"capped {np.mean([r['capped'] for r in sel]):.2f}", flush=True)
        json.dump(recs, open(out_path, "w"), ensure_ascii=False)
    print("common layer done", flush=True)


if __name__ == "__main__":
    main()
