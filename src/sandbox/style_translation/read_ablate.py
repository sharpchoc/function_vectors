#!/usr/bin/env python
"""READ-feature ABLATION at the evidence tokens (GPU; code-convention families; 2026-09-23, user request — mirror of write_ablate.py).

k = 4 prompts of the first --limit held-out documents, ALTERNATIVE-convention demonstrations (user decision: the natural convention is the
model's default). At EVERY evidence token of the prompt (code_evidence.py positions) and EVERY layer in --layers (default 0..27; 0 = embedding
output) the residual is edited by MultiPositionAblate (steer_hooks.py), each layer with its own direction and mean value:
    base      no edit
    own_zero  h_l -= (h_l.w_A^l) w_A^l          w_A^l = unit READ direction of the prompt's family at layer l (ablation/directions_read_all.npz)
    own_mean  h_l += (m_A^l - h_l.w_A^l) w_A^l   m_A^l = corpus-wide mean evidence-token projection (ablation/means_read_all.json, ablation_means.py --feature read)
    cf_zero / cf_mean: the same with the write ablation's seeded counterfactual partner family's read directions / means.
Sampling and grading as step 3 (seeded T = 1, 48 tokens, cut_code, decide_any; judge_rollouts.py --dir afterwards). success = keeps the
demonstrated convention AND judge OK; also the first-token margin lp(context pole) - lp(other) and the mean pre-ablation projection at L8."""
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
from src.sandbox.style_translation.steer_hooks import MultiPositionAblate, unit_test_position_ablate
from src.sandbox.style_translation.logprob_margin import candidates
from src.sandbox.style_translation.write_split import heldout
from src.sandbox.style_translation.write_sweep import KEEP, pad


def held_out_k(MP, fam, K, ctx):
    ev = {(e["doc_id"], e["pole"], e["k"]): e for e in json.load(open(MP["evidence"] / f"{fam}.json"))}
    items = []; no_ev = 0
    for p in json.load(open(MP["prompts"] / f"{fam}.json")):
        if p["k"] != K or p["style"] != ctx or not heldout(p["doc_id"]):
            continue
        e = ev.get((p["doc_id"], ctx, K))
        if e is None or not e["idx"]:
            no_ev += 1; continue
        assert e["prompt_len"] == len(p["prompt_ids"])
        it = dict(p); it["evidence"] = e["idx"]; items.append(it)
    if no_ev:
        print(f"{fam}: {no_ev} held-out k={K} {ctx} prompts without evidence positions skipped", flush=True)
    return items


class NoHook:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def run_arm(model, tok, fam, items, lexicon, arm, ctx, layers, w, mode, m, batch):
    recs = []
    for bi in range(0, len(items), batch):
        b = items[bi:bi + batch]; ids, att, L = pad(b, tok)
        positions = [[L - len(it["prompt_ids"]) + j for j in it["evidence"]] for it in b]
        torch.manual_seed(zlib.crc32(f"{fam}|read_ablate|{layers[0]}-{layers[-1]}|{arm}|{ctx}|{bi}".encode()))
        hook = MultiPositionAblate(model, layers, w, mode, m, positions) if arm != "base" else NoHook()
        with torch.no_grad(), hook as hk:
            lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, -1, :].float(), -1); top1 = lp.argmax(-1)
            proj = None if arm == "base" else list(hk.proj_at(8 if 8 in layers else layers[-1]))
            gen = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
        for r, it in enumerate(b):
            raw = tok.decode(gen[r, L:], skip_special_tokens=True); cut, capped = cut_code(raw, fam)
            d = decide_any(fam, tok.decode(it["prompt_ids"]), it["seg_prefix"], cut, it["next_nat"], it["next_alt"], lexicon)
            fc, fo = it["first_ctx"], it["first_other"]
            recs.append({k: it[k] for k in KEEP} | {"style": ctx, "context_style": ctx, "arm": arm, "mode": mode, "layers": [layers[0], layers[-1]], "target": ctx,
                         "n_evidence": len(it["evidence"]), "ref_sentence": it["ref_sentence"], "tail_raw": raw, "tail": cut, "capped": capped, "decision": d, "style_ok": d == ctx,
                         "lp_ctx": float(lp[r, fc]), "lp_other": float(lp[r, fo]), "margin_ctx": float(lp[r, fc] - lp[r, fo]),
                         "top1": "ctx" if top1[r].item() == fc else ("other" if top1[r].item() == fo else None),
                         "proj_before_L8": None if proj is None else float(proj[r]), "judge": None})
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="*", required=True)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--layers", nargs="*", type=int, default=list(range(0, 28))); ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--contexts", nargs="*", default=["alt"]); ap.add_argument("--out_dir", type=Path, required=True); ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()
    MP = model_paths(args.model); AB = MP["steering"].parent / "ablation"; args.out_dir.mkdir(parents=True, exist_ok=True)
    dirs = np.load(AB / "directions_read_all.npz"); means = json.load(open(AB / "means_read_all.json"))["families"]; layers = args.layers
    lex_p = MP["prompts"].parent / "scoring_lexicon.json"; lexicon = json.load(open(lex_p))["lexicon"] if lex_p.exists() else {}
    model, tok = load_model(model=args.model)
    assert unit_test_position_ablate(model, tok, layer=6), "position ablation hook unit test failed"
    for fam in args.families:
        out = args.out_dir / f"{fam}.json"
        if out.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        cf = means[fam]["cf_family"]; arms = [("base", fam, None, None), ("own_zero", fam, "zero", None), ("own_mean", fam, "mean", means[fam]["mean_by_layer"]),
                                            ("cf_zero", cf, "zero", None), ("cf_mean", cf, "mean", means[cf]["mean_by_layer"])]
        recs = []
        for ctx in args.contexts:
            items = candidates(fam, tok, held_out_k(MP, fam, args.k, ctx))[: args.limit]
            for arm, dfam, mode, m in arms:
                rr = run_arm(model, tok, fam, items, lexicon, arm, ctx, layers, dirs[dfam], mode, m, args.batch)
                for r in rr: r["direction_family"] = dfam
                recs += rr
                print(f"{fam}: {ctx} {arm} ({dfam}) convention rate {np.mean([r['style_ok'] for r in rr]):.2f}", flush=True)
        json.dump(recs, open(out, "w"), ensure_ascii=False)
        print(f"{fam}: {len(recs)} records, cf partner {cf}", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
