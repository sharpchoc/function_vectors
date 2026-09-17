#!/usr/bin/env python
"""Write-feature steering sweep (GPU): steer 0-shot prompts of HELD-OUT documents at the cue token with alpha * (mu_nat - mu_alt)
(towards natural) or -alpha * (...) (towards alternative), one layer at a time, and grade exactly like the unsteered 0-shot run:
one seeded T = 1 sample of 48 tokens, cut at the code boundary, context-aware convention decision; the Gemini judge runs afterwards
(judge_rollouts.py --dir <out_dir>). Also records the steered first-token log-prob margin lp(target's first token) - lp(other's).

Vectors: capture_cues.py --ks 3 4 --unpaired --split train --out_tag <vec_tag>  (training documents only).
Records -> <out_dir>/<family>.json, one per (document, arm); arm 'base' = unsteered (only with --with_base). Resumable per family.
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
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.rollout import load_model, MAX_NEW
from src.sandbox.style_translation.scoring import cut_code
from src.sandbox.style_translation.code_scoring import decide_any
from src.sandbox.style_translation.steer_hooks import CueSteer, unit_test
from src.sandbox.style_translation.logprob_margin import candidates
from src.sandbox.style_translation.write_split import heldout

KEEP = ("doc_id", "family", "k", "cue_tok", "seg_prefix", "next_nat", "next_alt", "context_tail", "es_text")


def held_out_items(prompts_dir, fam):
    ps = json.load(open(prompts_dir / f"{fam}.json"))
    nat = {p["doc_id"]: p for p in ps if p["style"] == "nat" and p["k"] == 0}
    alt = {p["doc_id"]: p for p in ps if p["style"] == "alt" and p["k"] == 0}
    items = []
    for d in sorted(nat):
        if not heldout(d):
            continue
        assert nat[d]["prompt_ids"] == alt[d]["prompt_ids"], f"k=0 prompts differ between conventions for {d}"
        it = dict(nat[d]); it["ref_nat"], it["ref_alt"] = nat[d]["ref_sentence"], alt[d]["ref_sentence"]; items.append(it)
    return items


def pad(batch, tok):
    lens = [len(it["prompt_ids"]) for it in batch]; L = max(lens)
    ids = torch.full((len(batch), L), tok.eos_token_id, dtype=torch.long); att = torch.zeros(len(batch), L, dtype=torch.long)
    for r, it in enumerate(batch):
        ids[r, L - lens[r]:] = torch.tensor(it["prompt_ids"]); att[r, L - lens[r]:] = 1
    return ids.cuda(), att.cuda(), L


def run_arm(model, tok, fam, items, texts, lexicon, arm, target, layer, alpha, vec, batch):
    recs = []
    for bi in range(0, len(items), batch):
        b = items[bi:bi + batch]; ids, att, L = pad(b, tok)
        torch.manual_seed(zlib.crc32(f"{fam}|{layer}|{alpha}|{target}|{bi}".encode()))
        with torch.no_grad(), CueSteer(model, max(layer, 1), vec, alpha):
            lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, -1, :].float(), -1); top1 = lp.argmax(-1)
            gen = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
        for r, it in enumerate(b):
            raw = tok.decode(gen[r, L:], skip_special_tokens=True); cut, capped = cut_code(raw, fam)
            d = decide_any(fam, texts[it["doc_id"]], it["seg_prefix"], cut, it["next_nat"], it["next_alt"], lexicon)
            fn, fa = it["first_ctx"], it["first_other"]                      # k = 0 items are the natural-pole items: ctx = natural
            t, o = (fn, fa) if target != "alt" else (fa, fn)
            recs.append({k: it[k] for k in KEEP} | {"style": target or "none", "arm": arm, "target": target, "layer": layer, "alpha": alpha,
                         "ref_sentence": it["ref_alt"] if target == "alt" else it["ref_nat"], "tail_raw": raw, "tail": cut, "capped": capped,
                         "decision": d, "style_ok": (d == target) if target else None,
                         "lp_nat": float(lp[r, fn]), "lp_alt": float(lp[r, fa]), "margin_target": float(lp[r, t] - lp[r, o]),
                         "top1": "nat" if top1[r].item() == fn else ("alt" if top1[r].item() == fa else None), "judge": None})
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="*", required=True)
    ap.add_argument("--layers", nargs="*", type=int, required=True); ap.add_argument("--alphas", nargs="*", type=float, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--vec_tag", default="k3_train"); ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=25); ap.add_argument("--with_base", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    MP = model_paths(args.model); VEC = MP["steering"] / f"vectors_{args.vec_tag}"; args.out_dir.mkdir(parents=True, exist_ok=True)
    lex_p = MP["prompts"].parent / "scoring_lexicon.json"; lexicon = json.load(open(lex_p))["lexicon"] if lex_p.exists() else {}
    model, tok = load_model(model=args.model)
    assert unit_test(model, tok, layer=6), "steering hook unit test failed"
    for fam in args.families:
        out = args.out_dir / f"{fam}.json"
        if out.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        items = candidates(fam, tok, held_out_items(MP["prompts"], fam))[: args.limit]
        texts = {it["doc_id"]: tok.decode(it["prompt_ids"]) for it in items}
        vecs = np.load(VEC / f"{fam}.npz")["v_nat"]                                # [n_layers, hidden]; layer L -> vecs[L-1]
        recs = []
        if args.with_base:
            recs += run_arm(model, tok, fam, items, texts, lexicon, "base", None, 0, 0.0, np.zeros_like(vecs[0]), args.batch)
        for layer in args.layers:
            for alpha in args.alphas:
                for target in ("nat", "alt"):
                    v = vecs[layer - 1] * (1.0 if target == "nat" else -1.0)
                    recs += run_arm(model, tok, fam, items, texts, lexicon, f"{target}_L{layer}_a{alpha:g}", target, layer, alpha, v, args.batch)
            print(f"{fam}: layer {layer} done", flush=True)
        json.dump(recs, open(out, "w"), ensure_ascii=False)
        ok = [r for r in recs if r["target"]]; print(f"{fam}: {len(items)} held-out docs, {len(recs)} records, target-convention rate {np.mean([r['style_ok'] for r in ok]):.2f}", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
