#!/usr/bin/env python
"""Write-feature ABLATION at the cue token (GPU; code-convention families; 2026-09-23, user request — the analogue of the FV cue ablation).

For the k = 4 prompts of the first --limit held-out documents per pole (both contexts: natural / alternative demonstrations), the cue
residual is edited at EVERY layer in --layers (default 1..27; user decision 2026-09-23) by MultiCueAblate (steer_hooks.py), each layer
with that layer's own direction and mean value and the completion is sampled and graded exactly like step 3 (seeded T = 1, 48 tokens,
cut_code, decide_any; Gemini judge afterwards with judge_rollouts.py --dir). Arms:
    base      no edit
    own_zero  h_l -= (h_l.w_A^l) w_A^l           w_A^l = unit write feature of the prompt's family at layer l (ablation/directions_all.npz)
    own_mean  h_l += (m_A^l - h_l.w_A^l) w_A^l    m_A^l = corpus-wide mean projection at layer l (ablation/means_all.json, ablation_means.py)
    cf_zero / cf_mean: the same with the seeded counterfactual partner family's w_B^l, m_B^l (means_all.json 'cf_family').
success = the completion keeps the CONTEXT's convention (style_ok = decision == context style) AND judge OK. Also recorded: first-token
log-prob margin lp(context pole) - lp(other pole) and the pre-ablation projection h.w at L24. Records -> <out_dir>/<family>.json; resumable."""
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
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.rollout import load_model, MAX_NEW
from src.sandbox.style_translation.scoring import cut_code
from src.sandbox.style_translation.code_scoring import decide_any
from src.sandbox.style_translation.steer_hooks import MultiCueAblate, unit_test_ablate
from src.sandbox.style_translation.logprob_margin import candidates
from src.sandbox.style_translation.write_split import heldout
from src.sandbox.style_translation.write_sweep import KEEP, pad


def held_out_k(MP, fam, K):
    return [dict(p) for p in json.load(open(MP["prompts"] / f"{fam}.json")) if p["k"] == K and heldout(p["doc_id"])]


class NoHook:
    proj = None
    def __enter__(self): return self
    def __exit__(self, *a): return False


def run_arm(model, tok, fam, items, lexicon, arm, ctx, layers, w, mode, m, batch):
    recs = []
    for bi in range(0, len(items), batch):
        b = items[bi:bi + batch]; ids, att, L = pad(b, tok)
        torch.manual_seed(zlib.crc32(f"{fam}|ablate|{layers[0]}-{layers[-1]}|{arm}|{ctx}|{bi}".encode()))
        hook = MultiCueAblate(model, layers, w, mode, m) if arm != "base" else NoHook()
        with torch.no_grad(), hook as hk:
            lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, -1, :].float(), -1); top1 = lp.argmax(-1)
            proj = None if arm == "base" else hk.proj_at(24 if 24 in layers else layers[-1]).clone()
            gen = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
        for r, it in enumerate(b):
            raw = tok.decode(gen[r, L:], skip_special_tokens=True); cut, capped = cut_code(raw, fam)
            d = decide_any(fam, tok.decode(it["prompt_ids"]), it["seg_prefix"], cut, it["next_nat"], it["next_alt"], lexicon)
            fc, fo = it["first_ctx"], it["first_other"]
            recs.append({k: it[k] for k in KEEP} | {"style": ctx, "context_style": ctx, "arm": arm, "mode": mode, "layers": [layers[0], layers[-1]], "target": ctx,
                         "ref_sentence": it["ref_sentence"], "tail_raw": raw, "tail": cut, "capped": capped, "decision": d, "style_ok": d == ctx,
                         "lp_ctx": float(lp[r, fc]), "lp_other": float(lp[r, fo]), "margin_ctx": float(lp[r, fc] - lp[r, fo]),
                         "top1": "ctx" if top1[r].item() == fc else ("other" if top1[r].item() == fo else None),
                         "proj_before": None if proj is None else float(proj[r]), "judge": None})
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="*", required=True)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--layers", nargs="*", type=int, default=list(range(1, 28))); ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out_dir", type=Path, required=True); ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()
    MP = model_paths(args.model); AB = MP["steering"].parent / "ablation"; args.out_dir.mkdir(parents=True, exist_ok=True)
    dirs = np.load(AB / "directions_all.npz"); means = json.load(open(AB / "means_all.json"))["families"]; layers = args.layers
    lex_p = MP["prompts"].parent / "scoring_lexicon.json"; lexicon = json.load(open(lex_p))["lexicon"] if lex_p.exists() else {}
    model, tok = load_model(model=args.model)
    assert unit_test_ablate(model, tok, layer=6), "ablation hook unit test failed"
    for fam in args.families:
        out = args.out_dir / f"{fam}.json"
        if out.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        items = candidates(fam, tok, held_out_k(MP, fam, args.k))
        by = {s: [it for it in items if it["style"] == s][: args.limit] for s in ("nat", "alt")}
        cf = means[fam]["cf_family"]; arms = [("base", fam, None, None), ("own_zero", fam, "zero", None), ("own_mean", fam, "mean", means[fam]["mean_by_layer"]),
                                            ("cf_zero", cf, "zero", None), ("cf_mean", cf, "mean", means[cf]["mean_by_layer"])]
        recs = []
        for arm, dfam, mode, m in arms:
            for ctx in ("nat", "alt"):
                rr = run_arm(model, tok, fam, by[ctx], lexicon, arm, ctx, layers, dirs[dfam], mode, m, args.batch)
                for r in rr: r["direction_family"] = dfam
                recs += rr
            print(f"{fam}: {arm} ({dfam}) convention rate nat {np.mean([r['style_ok'] for r in recs if r['arm'] == arm and r['context_style'] == 'nat']):.2f} "
                  f"alt {np.mean([r['style_ok'] for r in recs if r['arm'] == arm and r['context_style'] == 'alt']):.2f}", flush=True)
        json.dump(recs, open(out, "w"), ensure_ascii=False)
        print(f"{fam}: {len(by['nat'])} + {len(by['alt'])} held-out k={args.k} prompts, {len(recs)} records, cf partner {cf}", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
