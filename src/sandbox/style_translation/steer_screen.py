#!/usr/bin/env python
"""Step 4b — screen steering settings (GPU): layer x alpha grid on 50 texts per family.

Prompt = the k=0 item (Spanish + English up to the FIRST cue token; identical for both styles,
asserted). For target style in {nat, alt} (v = +v_nat / -v_nat), layer in SCREEN_LAYERS, alpha in
ALPHAS: one seeded T=1 sample, 16 new tokens, added at the cue token only (steer_hooks.CueSteer),
scored with scoring.decide -> target-style rate and unscorable rate. alpha = 0 (unsteered) is
sampled once per text and reported at every layer.
Records -> artifacts/style_translation/steering/screen/<family>.json
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
from src.sandbox.style_translation.scoring import decide
from src.sandbox.style_translation.steer_hooks import CueSteer, unit_test

PROMPTS = ARTIFACTS_ROOT / "style_translation" / "prompts"
VEC = ARTIFACTS_ROOT / "style_translation" / "steering" / "vectors"
OUT = ARTIFACTS_ROOT / "style_translation" / "steering" / "screen"
SCREEN_LAYERS = [2, 4, 6, 8, 10, 12, 16, 20, 24]
ALPHAS = [0.5, 1.0, 2.0, 4.0]
N_SCREEN = 50
MAX_NEW = 16


def k0_items(fam, n=None):
    ps = json.load(open(PROMPTS / f"{fam}.json"))
    nat = {p["doc_id"]: p for p in ps if p["style"] == "nat" and p["k"] == 0}
    alt = {p["doc_id"]: p for p in ps if p["style"] == "alt" and p["k"] == 0}
    items = []
    for d in sorted(nat):
        assert nat[d]["prompt_ids"] == alt[d]["prompt_ids"], f"k=0 prompts differ between styles for {d}"
        items.append(nat[d])
    return items[:n] if n else items


def sample(model, tok, items, layer, vec, alpha, seed_tag, max_new):
    """One batch of items (already same-ish length), returns raw tails."""
    lens = [len(it["prompt_ids"]) for it in items]; L = max(lens)
    ids = torch.full((len(items), L), tok.eos_token_id, dtype=torch.long)
    att = torch.zeros(len(items), L, dtype=torch.long)
    for r, it in enumerate(items):
        ids[r, L - lens[r]:] = torch.tensor(it["prompt_ids"]); att[r, L - lens[r]:] = 1
    torch.manual_seed(zlib.crc32(seed_tag.encode()))
    ctx = CueSteer(model, layer, vec, alpha) if (vec is not None and alpha != 0) else None
    with torch.no_grad():
        if ctx:
            ctx.__enter__()
        try:
            gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), do_sample=True, temperature=1.0,
                                 top_k=0, top_p=1.0, max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
    return [tok.decode(gen[r, L:], skip_special_tokens=True) for r in range(len(items))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--batch", type=int, default=25)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load_model()
    assert unit_test(model, tok, layer=6), "hook unit test failed"
    print("hook unit test passed", flush=True)
    for fam in args.families:
        out_path = OUT / f"{fam}.json"
        if out_path.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        items = k0_items(fam, N_SCREEN)
        vecs = np.load(VEC / f"{fam}.npz")["v_nat"]        # [28, 4096], layer L -> vecs[L-1]
        recs = []
        # alpha = 0 once
        for bi in range(0, len(items), args.batch):
            chunk = items[bi:bi + args.batch]
            tails = sample(model, tok, chunk, 0, None, 0.0, f"{fam}|screen|a0|{bi}", MAX_NEW)
            for it, t in zip(chunk, tails):
                d = decide(fam, it["seg_prefix"], t, it["next_nat"], it["next_alt"])
                recs.append({"doc_id": it["doc_id"], "target": None, "layer": 0, "alpha": 0.0, "tail": t, "decision": d})
        for target in ("nat", "alt"):
            for layer in SCREEN_LAYERS:
                v = vecs[layer - 1] * (1 if target == "nat" else -1)
                for alpha in ALPHAS:
                    for bi in range(0, len(items), args.batch):
                        chunk = items[bi:bi + args.batch]
                        tails = sample(model, tok, chunk, layer, v, alpha, f"{fam}|screen|{target}|{layer}|{alpha}|{bi}", MAX_NEW)
                        for it, t in zip(chunk, tails):
                            d = decide(fam, it["seg_prefix"], t, it["next_nat"], it["next_alt"])
                            recs.append({"doc_id": it["doc_id"], "target": target, "layer": layer, "alpha": alpha,
                                         "tail": t, "decision": d})
                print(f"{fam}: {target} L={layer} done", flush=True)
        json.dump(recs, open(out_path, "w"), ensure_ascii=False)
        # digest
        base = [r for r in recs if r["alpha"] == 0.0]
        print(f"{fam}: unsteered nat={np.mean([r['decision']=='nat' for r in base]):.2f} alt={np.mean([r['decision']=='alt' for r in base]):.2f}", flush=True)
        for target in ("nat", "alt"):
            best = max(((layer, a, np.mean([r["decision"] == target for r in recs if r["target"] == target and r["layer"] == layer and r["alpha"] == a]))
                        for layer in SCREEN_LAYERS for a in ALPHAS), key=lambda x: (x[2], -x[1], -x[0]))
            print(f"{fam}: best {target}: L={best[0]} alpha={best[1]} rate={best[2]:.2f}", flush=True)
    print("screen done", flush=True)


if __name__ == "__main__":
    main()
