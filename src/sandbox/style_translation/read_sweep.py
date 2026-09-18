#!/usr/bin/env python
"""Read-feature steering sweep (GPU). k = 3 prompts of HELD-OUT documents whose demonstrations are in the WRONG style are steered
towards the other style: an alt-context prompt gets +alpha * r_L (towards natural), a nat-context prompt gets -alpha * r_L (towards
alternative), where r_L = mean_nat - mean_alt of the read feature (capture_read.py, training documents). The vector is added at
layer L at EVERY evidence token of the prompt (code_evidence.py positions for the prompt's own pole; same vector at each), prefill
only; nothing at the query cue or at generated tokens. Grading exactly as step 3 (seeded T = 1 sample, 48 tokens, cut_code,
decide_any); judge afterwards (judge_rollouts.py --dir). success = TARGET convention AND judge OK; first-token margin recorded.

Records -> <out_dir>/<family>.json; arm 'base_<ctx>' = unsteered prompts of each context style (only with --with_base). Resumable.
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
from src.sandbox.style_translation.steer_hooks import PositionSteer, unit_test_positions
from src.sandbox.style_translation.logprob_margin import candidates
from src.sandbox.style_translation.write_split import heldout
from src.sandbox.style_translation.write_sweep import KEEP, pad

K = 3


def held_out_k3(MP, fam):
    ev = {(e["doc_id"], e["pole"], e["k"]): e for e in json.load(open(MP["evidence"] / f"{fam}.json"))}
    items = []; no_ev = 0
    for p in json.load(open(MP["prompts"] / f"{fam}.json")):
        if p["k"] != K or not heldout(p["doc_id"]):
            continue
        e = ev.get((p["doc_id"], p["style"], K))
        if e is None or not e["idx"]:
            no_ev += 1; continue
        assert e["prompt_len"] == len(p["prompt_ids"])
        it = dict(p); it["evidence"] = e["idx"]; items.append(it)
    if no_ev:
        print(f"{fam}: {no_ev} held-out k=3 prompts without evidence positions skipped", flush=True)
    return items


def run_arm(model, tok, fam, items, lexicon, arm, ctx, target, layer, alpha, vec, batch):
    """items = prompts whose context style is `ctx`; steer towards `target` (None = unsteered)."""
    recs = []
    for bi in range(0, len(items), batch):
        b = items[bi:bi + batch]; ids, att, L = pad(b, tok)
        positions = [[L - len(it["prompt_ids"]) + j for j in it["evidence"]] for it in b]
        torch.manual_seed(zlib.crc32(f"{fam}|read|{layer}|{alpha}|{ctx}|{target}|{bi}".encode()))
        with torch.no_grad(), PositionSteer(model, layer, vec, alpha, positions):
            lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, -1, :].float(), -1); top1 = lp.argmax(-1)
            gen = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
        for r, it in enumerate(b):
            raw = tok.decode(gen[r, L:], skip_special_tokens=True); cut, capped = cut_code(raw, fam)
            d = decide_any(fam, tok.decode(it["prompt_ids"]), it["seg_prefix"], cut, it["next_nat"], it["next_alt"], lexicon)
            fc, fo = it["first_ctx"], it["first_other"]              # ctx = the prompt's own style
            tgt = target or ("alt" if ctx == "nat" else "nat")       # unsteered arms are scored against the same (wrong-style) target
            t, o = (fo, fc) if tgt != ctx else (fc, fo)
            recs.append({k: it[k] for k in KEEP} | {"style": tgt, "context_style": ctx, "arm": arm, "target": target, "layer": layer, "alpha": alpha,
                         "n_evidence": len(it["evidence"]), "ref_sentence": it["ref_other"] if tgt != ctx else it["ref_ctx"], "tail_raw": raw, "tail": cut, "capped": capped,
                         "decision": d, "style_ok": d == tgt, "lp_ctx": float(lp[r, fc]), "lp_other": float(lp[r, fo]),
                         "margin_target": float(lp[r, t] - lp[r, o]), "top1": "ctx" if top1[r].item() == fc else ("other" if top1[r].item() == fo else None), "judge": None})
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="*", required=True)
    ap.add_argument("--layers", nargs="*", type=int, required=True); ap.add_argument("--alphas", nargs="*", type=float, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--vec_tag", default="k3_train"); ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=12); ap.add_argument("--with_base", action="store_true"); ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    MP = model_paths(args.model); VEC = MP["read_features"] / f"vectors_{args.vec_tag}"; args.out_dir.mkdir(parents=True, exist_ok=True)
    lex_p = MP["prompts"].parent / "scoring_lexicon.json"; lexicon = json.load(open(lex_p))["lexicon"] if lex_p.exists() else {}
    model, tok = load_model(model=args.model)
    assert unit_test_positions(model, tok, layer=6), "position steering hook unit test failed"
    for fam in args.families:
        out = args.out_dir / f"{fam}.json"
        if out.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        items = candidates(fam, tok, held_out_k3(MP, fam))
        refs = {(it["doc_id"], it["style"]): it["ref_sentence"] for it in items}   # the judge reference follows the TARGET style's own continuation
        for it in items:
            it["ref_ctx"] = it["ref_sentence"]; it["ref_other"] = refs.get((it["doc_id"], "alt" if it["style"] == "nat" else "nat"), it["ref_sentence"])
        by = {s: [it for it in items if it["style"] == s][: args.limit] for s in ("nat", "alt")}
        r = np.load(VEC / f"{fam}.npz")["diff"]                                    # [n_layers, D]; layer L -> r[L-1]
        recs = []
        if args.with_base:
            for ctx in ("nat", "alt"):
                recs += run_arm(model, tok, fam, by[ctx], lexicon, f"base_{ctx}", ctx, None, 0, 0.0, np.zeros_like(r[0]), args.batch)
        for layer in args.layers:
            for alpha in args.alphas:
                for ctx, target, sign in (("alt", "nat", 1.0), ("nat", "alt", -1.0)):
                    recs += run_arm(model, tok, fam, by[ctx], lexicon, f"{target}_L{layer}_a{alpha:g}", ctx, target, layer, alpha, sign * r[layer - 1], args.batch)
            print(f"{fam}: layer {layer} done", flush=True)
        json.dump(recs, open(out, "w"), ensure_ascii=False)
        st = [x for x in recs if x["target"]]
        print(f"{fam}: held-out k=3 prompts nat {len(by['nat'])} alt {len(by['alt'])}, {len(recs)} records, target-convention rate {np.mean([x['style_ok'] for x in st]) if st else float('nan'):.2f}", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
