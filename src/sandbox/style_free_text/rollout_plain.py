#!/usr/bin/env python
"""style_free_text — GPT-J rollouts on the plain-English prompts (no translation scaffold).

Identical sampling to style_translation.rollout (GPT-J-6B fp16, ONE seeded T=1 sample, 48 new
tokens, cut at the sentence end with a `capped` flag, style decided by scoring.decide); only the
prompt/rollout directories and the seed tag differ. Resumable per family; --families for sharding.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.scoring import cut_sentence, decide
from src.sandbox.style_translation.rollout import load_model, MAX_NEW
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len

PROMPTS = ARTIFACTS_ROOT / "style_free_text" / "prompts"
OUT = ARTIFACTS_ROOT / "style_free_text" / "rollouts"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--model_dir", type=Path, default=None)
    ap.add_argument("--token_budget", type=int, default=8000)
    ap.add_argument("--batch_cap", type=int, default=16)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        for fam in args.families:
            items = json.load(open(PROMPTS / f"{fam}.json"))
            for it in (items[0], items[len(items) // 2]):
                print(f"\n[{fam} style={it['style']} k={it['k']}] {tok.decode(it['prompt_ids'])[-140:]!r}\n   cue={it['cue_tok']!r}")
        return
    model, tok = load_model(args.model_dir)
    for fam in args.families:
        out_path = OUT / f"{fam}.json"
        if out_path.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        items = json.load(open(PROMPTS / f"{fam}.json"))
        for it in items:
            it["ids"] = it["prompt_ids"]
        recs = [None] * len(items)
        for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
            lens = [len(items[i]["ids"]) for i in b]; L = max(lens)
            ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros(len(b), L, dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, L - lens[r]:] = torch.tensor(items[i]["ids"]); att[r, L - lens[r]:] = 1
            seed = zlib.crc32(f"{fam}|plain|{bi}".encode()); torch.manual_seed(seed)
            with torch.no_grad():
                gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), do_sample=True,
                                     temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=MAX_NEW,
                                     pad_token_id=tok.eos_token_id)
            for r, i in enumerate(b):
                it = items[i]
                raw = tok.decode(gen[r, L:], skip_special_tokens=True)
                cut, capped = cut_sentence(raw)
                dec = decide(fam, it["seg_prefix"], cut, it["next_nat"], it["next_alt"])
                recs[i] = {k: it[k] for k in ("doc_id", "family", "style", "k", "cue_tok", "seg_prefix",
                                              "next_nat", "next_alt", "ref_sentence", "context_tail")}
                recs[i].update(tail_raw=raw, tail=cut, capped=capped, decision=dec,
                               style_ok=(dec == it["style"]), seed=seed, judge=None)
            if (bi + 1) % 50 == 0:
                print(f"{fam}: batch {bi + 1}", flush=True)
        json.dump(recs, open(out_path, "w"), ensure_ascii=False)
        by = {}
        for r in recs:
            d = by.setdefault((r["style"], r["k"]), [0, 0, 0, 0]); d[0] += 1; d[1] += r["style_ok"]; d[2] += r["decision"] is None; d[3] += r["capped"]
        print(f"{fam}: done. style_ok / unscorable / capped by (style,k): " +
              "; ".join(f"{s}{k}:{v[1]/v[0]:.2f}/{v[2]/v[0]:.2f}/{v[3]/v[0]:.2f}" for (s, k), v in sorted(by.items())), flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
