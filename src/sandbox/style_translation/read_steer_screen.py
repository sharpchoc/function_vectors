#!/usr/bin/env python
"""Step 7a — screen: steer at the EVIDENCE tokens of a k = 3 prompt with the read-feature difference.

u_nat(L) = r_nat(L) - r_alt(L) (step-6 read features); u_alt = -u_nat. Directions per family:
  nat2alt  nat-context k = 3 prompt, add alpha * u_alt at every evidence token of its 3 instances,
           target = alt at the 4th decision
  alt2nat  alt-context prompt, alpha * u_nat, target = nat
Sweep SCREEN_LAYERS x ALPHAS on the first 50 texts, one seeded T=1 sample, 16 new tokens, style only
(scoring.decide); alpha = 0 sampled once per context pole. Records -> read_steer/screen/<family>.json.
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
from src.sandbox.style_translation.steer_hooks import PositionSteer, unit_test_positions
from src.sandbox.style_translation.steer_screen import SCREEN_LAYERS, ALPHAS

PROMPTS = ARTIFACTS_ROOT / "style_translation" / "prompts"
EVID = ARTIFACTS_ROOT / "style_translation" / "read_features" / "evidence"
RF = ARTIFACTS_ROOT / "style_translation" / "read_features"
ROOT = ARTIFACTS_ROOT / "style_translation" / "read_steer"
K_CTX = 3
N_SCREEN, MAX_NEW = 50, 16
DIRECTIONS = {"nat2alt": ("nat", "alt"), "alt2nat": ("alt", "nat")}   # context pole -> target pole


def k3_items(fam, pole, n=None):
    """k = 3 prompt records of one context pole with `positions` = evidence tokens of instances 0..2."""
    ev = {(r["doc_id"], r["pole"]): r for r in json.load(open(EVID / f"{fam}.json"))}
    items = sorted([p for p in json.load(open(PROMPTS / f"{fam}.json")) if p["style"] == pole and p["k"] == K_CTX], key=lambda p: p["doc_id"])
    for it in items:
        inst = ev[(it["doc_id"], pole)]["instances"][:K_CTX]
        pos = sorted({j for e in inst for j in e["idx"]})
        assert pos and max(pos) < len(it["prompt_ids"]), f"evidence outside the k=3 prompt {it['doc_id']}"
        it["positions"] = pos
        it["cue_is_evidence"] = max(pos) == len(it["prompt_ids"]) - 1   # adjacent opportunities: the 3rd instance's last token is also the 4th decision's cue
    return items[:n] if n else items


def read_vectors(fam):
    d = np.load(RF / f"{fam}.npz")
    return d["mean_nat"] - d["mean_alt"]            # [28, D] = u_nat per layer


def sample_positions(model, tok, items, layer, vec, alpha, seed_tag, max_new):
    lens = [len(it["prompt_ids"]) for it in items]; L = max(lens)
    ids = torch.full((len(items), L), tok.eos_token_id, dtype=torch.long)
    att = torch.zeros(len(items), L, dtype=torch.long)
    for r, it in enumerate(items):
        ids[r, L - lens[r]:] = torch.tensor(it["prompt_ids"]); att[r, L - lens[r]:] = 1
    positions = [[L - lens[r] + j for j in it["positions"]] for r, it in enumerate(items)]
    torch.manual_seed(zlib.crc32(seed_tag.encode()))
    ctx = PositionSteer(model, layer, vec, alpha, positions) if (vec is not None and alpha != 0) else None
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


def run_arm(model, tok, fam, items, layer, vec, alpha, tag, max_new, batch, extra):
    recs = []
    for bi in range(0, len(items), batch):
        chunk = items[bi:bi + batch]
        tails = sample_positions(model, tok, chunk, layer, vec, alpha, f"{fam}|{tag}|{bi}", max_new)
        for it, raw in zip(chunk, tails):
            d = decide(fam, it["seg_prefix"], raw, it["next_nat"], it["next_alt"])
            recs.append({"doc_id": it["doc_id"], "context": it["style"], "layer": layer, "alpha": alpha,
                         "n_positions": len(it["positions"]), "tail": raw, "decision": d} | extra)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--batch", type=int, default=25)
    args = ap.parse_args()
    (ROOT / "screen").mkdir(parents=True, exist_ok=True)
    model, tok = load_model()
    assert unit_test_positions(model, tok, layer=6) and unit_test_positions(model, tok, layer=20)
    print("position hook unit test passed", flush=True)
    for fam in args.families:
        out_path = ROOT / "screen" / f"{fam}.json"
        if out_path.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        u = read_vectors(fam); recs = []
        items = {pole: k3_items(fam, pole, N_SCREEN) for pole in ("nat", "alt")}
        for pole in ("nat", "alt"):
            recs += run_arm(model, tok, fam, items[pole], 0, None, 0.0, f"screen|base|{pole}", MAX_NEW, args.batch, {"direction": None, "target": None})
            sel = [r for r in recs if r["context"] == pole and r["alpha"] == 0.0]
            print(f"{fam}: unsteered {pole}-context: nat {np.mean([r['decision']=='nat' for r in sel]):.2f} alt {np.mean([r['decision']=='alt' for r in sel]):.2f}", flush=True)
        for direction, (ctx_pole, target) in DIRECTIONS.items():
            sign = 1 if target == "nat" else -1
            best = (-1, None, None)
            for layer in SCREEN_LAYERS:
                for a in ALPHAS:
                    rs = run_arm(model, tok, fam, items[ctx_pole], layer, u[layer - 1] * sign, a, f"screen|{direction}|L{layer}|a{a}", MAX_NEW, args.batch,
                                 {"direction": direction, "target": target})
                    recs += rs
                    rate = float(np.mean([r["decision"] == target for r in rs]))
                    if rate > best[0]:
                        best = (rate, layer, a)
                print(f"{fam}: {direction} L={layer} done", flush=True)
            print(f"{fam}: best {direction}: L={best[1]} alpha={best[2]} rate={best[0]:.2f}", flush=True)
        json.dump(recs, open(out_path, "w"), ensure_ascii=False)
    print("read screen done", flush=True)


if __name__ == "__main__":
    main()
