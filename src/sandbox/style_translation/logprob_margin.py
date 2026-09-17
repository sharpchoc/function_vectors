#!/usr/bin/env python
"""Step 3b — first-token log-prob margin at the cue (GPU). Classifier- and judge-free failsafe for the accuracy-by-k curves.

For every prompt item (build_prompts.py: twin P cut right after cue token k) one teacher-forced forward pass gives the next-token
distribution at the cue. The two candidates are the first token of the two renderings of opportunity k in THIS prompt's context:
  first_ctx   = the twin's own next token                         (rendering in the context's convention)
  first_other = the next token of the same twin with opportunity k rendered in the other convention (cue_tokens.variant)
By construction they differ (the cue is the last shared token). margin = log p(first_ctx) - log p(first_other); top1 = argmax is
first_ctx. Output -> artifacts/style_translation/<model>/logprob/<family>.json (one record per prompt item). User decision
2026-09-17: first token only (not the full opportunity span). Resumable per family; --check verifies the tokenisation on CPU.
"""
import argparse
import json
import sys
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.cue_tokens import header_for, variant, decision_points
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len

PAIRS = STYLE_TRANSLATION_DATA / "pairs"


def candidates(fam, tok, items):
    """Attach first_ctx / first_other token ids to every prompt item (and assert the stored prompt_ids match the twin tokenisation)."""
    recs = {r["doc_id"]: r for r in json.load(open(PAIRS / f"{fam}.json"))}
    cache = {}
    for it in items:
        rec = recs[it["doc_id"]]; P = it["style"]; k = it["k"]; header = header_for(rec)
        key = (it["doc_id"], P)
        if key not in cache:
            cache[key] = tok(header + rec[f"text_{P}"]).input_ids
        ids = cache[key]; n = len(it["prompt_ids"])
        assert ids[:n] == it["prompt_ids"], f"{fam} {it['doc_id']} {P} k={k}: prompt_ids do not match the twin tokenisation"
        i = decision_points(rec)[k]
        ids_b = tok(header + variant(rec, P, {i})).input_ids
        assert ids_b[:n] == it["prompt_ids"] and len(ids) > n and len(ids_b) > n, f"{fam} {it['doc_id']} {P} k={k}: variant prefix mismatch"
        it["first_ctx"], it["first_other"] = ids[n], ids_b[n]
        assert it["first_ctx"] != it["first_other"], f"{fam} {it['doc_id']} {P} k={k}: renderings share the first token"
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code", help="models.MODELS key")
    ap.add_argument("--families", nargs="*", required=True)
    ap.add_argument("--model_dir", type=Path, default=None)
    ap.add_argument("--token_budget", type=int, default=8000); ap.add_argument("--batch_cap", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--check", action="store_true", help="tokenisation checks only, no model")
    args = ap.parse_args()
    from src.sandbox.style_translation.models import paths as model_paths
    MP = model_paths(args.model); PROMPTS = MP["prompts"]; OUT = MP["prompts"].parent / "logprob"; OUT.mkdir(parents=True, exist_ok=True)
    if args.check:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MP["tokenizer"])
        for fam in args.families:
            items = candidates(fam, tok, json.load(open(PROMPTS / f"{fam}.json"))[: args.limit])
            ex = items[0]
            print(f"{fam}: {len(items)} items ok; e.g. {ex['doc_id']} {ex['style']} k={ex['k']} ctx={tok.decode([ex['first_ctx']])!r} other={tok.decode([ex['first_other']])!r}", flush=True)
        return
    import torch
    from src.sandbox.style_translation.rollout import load_model
    model, tok = load_model(args.model_dir, args.model)
    for fam in args.families:
        out_path = OUT / f"{fam}.json"
        if out_path.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        items = candidates(fam, tok, json.load(open(PROMPTS / f"{fam}.json"))[: args.limit])
        for it in items:
            it["ids"] = it["prompt_ids"]
        recs = [None] * len(items)
        for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
            lens = [len(items[i]["ids"]) for i in b]; L = max(lens)
            ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long); att = torch.zeros(len(b), L, dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, L - lens[r]:] = torch.tensor(items[i]["ids"]); att[r, L - lens[r]:] = 1
            with torch.no_grad():
                logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits[:, -1, :].float()
                lp = torch.log_softmax(logits, dim=-1); top1 = lp.argmax(-1)
            for r, i in enumerate(b):
                it = items[i]; a, o = it["first_ctx"], it["first_other"]
                la, lo = float(lp[r, a]), float(lp[r, o])
                recs[i] = dict(doc_id=it["doc_id"], family=fam, style=it["style"], k=it["k"], first_ctx=tok.decode([a]), first_other=tok.decode([o]),
                               lp_ctx=la, lp_other=lo, margin=la - lo, top1_ctx=bool(top1[r].item() == a), top1_other=bool(top1[r].item() == o))
        json.dump(recs, open(out_path, "w"), ensure_ascii=False)
        by = {}
        for r in recs:
            d = by.setdefault((r["style"], r["k"]), [0, 0.0, 0]); d[0] += 1; d[1] += r["margin"]; d[2] += r["top1_ctx"]
        print(f"{fam}: done. mean margin / top1 by (style,k): " + "; ".join(f"{s}{k}:{v[1]/v[0]:+.2f}/{v[2]/v[0]:.2f}" for (s, k), v in sorted(by.items())), flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
