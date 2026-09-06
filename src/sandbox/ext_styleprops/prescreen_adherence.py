#!/usr/bin/env python
"""Stage A4 behavioral pre-screen: sampled adherence at style-property cue tokens.

For every (property, context polarity, doc, site): the prompt is the polarity twin's
text truncated AFTER the site's cue token (the point of no return, identity-matched
across twins); readout is ONE T=1 seeded sampled continuation (repo readout convention —
sixshot_dummy protocol; user decision 2026-09-01: sampled adherence is the ONLY readout).
The continuation is classified nat / alt / unscorable by the property's loose classifier
with a strict expected-continuation prefix match as fallback.

Positions are tokenizer-verified: the doc is re-tokenized here and the stored cue
token id is asserted (DECISIONS 2026-07-13).

FRAMING (2026-09-06, translation-framing variant, user decisions in
results/style_properties/translation_framing/README.md):
  --framing plain      prompt = <English twin up to and including the cue token>   (default)
  --framing translate  prompt = "Spanish:\n<neutral Spanish translation of the base doc>\n\n"
                                "English:\n<English twin up to and including the cue token>"
The header is tokenized separately and its ids are PREPENDED to the unchanged English ids,
so every stored cue position/id assertion still holds at the shifted position. Both framings
record the reference continuation of each twin (`ref_nat`, `ref_alt`: the next `max_new`
tokens of the document) and the last 400 chars of the English prefix (`ctx_tail`) so
translation correctness can be scored afterwards on CPU.

  --add_refs   no sampling: rewrite existing record files in --out_root adding
               ref_nat / ref_alt / ctx_tail (used to backfill the plain 2026-09-01 records).

Output: artifacts/style_properties/prescreen[_translate]/<prop>.json (resumable per property)
        [{doc_id, k, dist, pol, label, tail, ref_nat, ref_alt, ctx_tail}]
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
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
from src.sandbox.ext_styleprops.properties import PROPS
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len

PROPS_DIR = REPO_ROOT / "dataset_files" / "style_properties" / "props"
ES_CORPUS = REPO_ROOT / "dataset_files" / "style_properties" / "base_corpus_es.json"
POOL_PATH = REPO_ROOT / "task_splits" / "style_properties_pool.json"
HEADER_ES = "Spanish:\n{es}\n\nEnglish:\n"
MAX_PROMPT_TOKENS = 2000   # GPT-J context 2048 minus generation headroom
CTX_TAIL_CHARS = 400


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--framing", choices=["plain", "translate"], default="plain")
    p.add_argument("--props", nargs="*", default=None,
                   help="default: all 17 (plain) / the 13-property pool (translate)")
    p.add_argument("--out_root", type=Path, default=None,
                   help="default: artifacts/style_properties/prescreen[_translate]")
    p.add_argument("--es_corpus", type=Path, default=ES_CORPUS)
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=16000)
    p.add_argument("--batch_cap", type=int, default=32)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    p.add_argument("--add_refs", action="store_true",
                   help="backfill ref_nat/ref_alt/ctx_tail into existing records; no sampling")
    p.add_argument("--dry_run", action="store_true",
                   help="build items, print layout/length stats and 3 decoded prompts, exit")
    a = p.parse_args()
    if a.props is None:
        a.props = sorted(PROPS) if a.framing == "plain" \
            else json.load(open(POOL_PATH))["pass"]
    if a.out_root is None:
        a.out_root = ARTIFACTS_ROOT / "style_properties" / \
            ("prescreen" if a.framing == "plain" else "prescreen_translate")
    return a


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    md = resolve_model_dir(model_dir)
    tok = AutoTokenizer.from_pretrained(md)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok, md


def resolve_model_dir(model_dir):
    """Snapshot glob filtered to complete snapshots (DECISIONS 2026-08-19: a stale snapshot
    with only a stray safetensors file breaks the naive glob)."""
    if model_dir is not None:
        return model_dir
    snaps = sorted(Path("/workspace/.cache/huggingface/hub/"
                        "models--EleutherAI--gpt-j-6b/snapshots").glob("*"))
    snaps = [s for s in snaps if (s / "config.json").exists()]
    assert snaps, "no complete GPT-J snapshot found; pass --model_dir"
    return snaps[-1]


def load_model(model_dir):
    from transformers import AutoModelForCausalLM
    tok, md = load_tokenizer(model_dir)
    model = AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.float16).cuda().eval()
    return model, tok


def build_items(prop_name, tok, framing="plain", es=None):
    """One item per (doc, site, polarity). `ids` is the full prompt; the English doc ids are
    identical across framings so the stored cue index/id are asserted on them directly."""
    data = json.load(open(PROPS_DIR / f"{prop_name}.json"))
    items, dropped_long = [], 0
    for d in data["docs"]:
        ids = {pol: tok(d[f"text_{pol}"]).input_ids for pol in ("nat", "alt")}
        header = []
        if framing == "translate":
            header = tok(HEADER_ES.format(es=es[d["doc_id"]])).input_ids
        for s in d["sites"]:
            for pol in ("nat", "alt"):
                cue = s["cue_idx"][pol]
                assert ids[pol][cue] == s["cue_tok_id"], \
                    f"{prop_name}/{d['doc_id']}/k{s['k']}/{pol}: cue token mismatch"
                prompt = header + ids[pol][:cue + 1]
                if len(prompt) > MAX_PROMPT_TOKENS:
                    dropped_long += 1
                    continue
                other = "alt" if pol == "nat" else "nat"
                mn = s["max_new"]
                items.append({
                    "ids": prompt,
                    "max_new": mn,
                    "doc_id": d["doc_id"], "k": s["k"], "dist": s["dist"][pol],
                    "pol": pol, "exp": s["exp"][f"{pol}_ctx"],
                    f"ref_{pol}": tok.decode(ids[pol][cue + 1:cue + 1 + mn]),
                    f"ref_{other}": tok.decode(ids[other][s["cue_idx"][other] + 1:
                                                          s["cue_idx"][other] + 1 + mn]),
                    "ctx_tail": tok.decode(ids[pol][:cue + 1])[-CTX_TAIL_CHARS:],
                })
    if dropped_long:
        print(f"{prop_name}: dropped {dropped_long} items over {MAX_PROMPT_TOKENS} tokens",
              flush=True)
    return items


def classify(prop, tail, exp):
    lab = prop.classify(tail)
    if lab is not None:
        return lab
    for label, e in sorted(exp.items(), key=lambda kv: -len(kv[1])):
        if e and tail.startswith(e):
            return label
    return None


def load_es(path):
    return {d["doc_id"]: d["text_es"] for d in json.load(open(path))}


def add_refs(args):
    """Backfill reference continuations into existing records (order = build_items order)."""
    tok, _ = load_tokenizer(args.model_dir)
    es = load_es(args.es_corpus) if args.framing == "translate" else None
    for name in sorted(args.props)[args.shard_idx::args.shard_n]:
        path = args.out_root / f"{name}.json"
        if not path.exists():
            print(f"{name}: no records, skip", flush=True)
            continue
        d = json.load(open(path))
        items = build_items(name, tok, args.framing, es)
        assert len(items) == len(d["records"]), \
            f"{name}: {len(items)} items vs {len(d['records'])} records"
        for it, r in zip(items, d["records"]):
            assert (it["doc_id"], it["k"], it["pol"]) == (r["doc_id"], r["k"], r["pol"]), \
                f"{name}: record order mismatch at {r['doc_id']}/k{r['k']}/{r['pol']}"
            for key in ("ref_nat", "ref_alt", "ctx_tail"):
                r[key] = it[key]
        json.dump(d, open(path, "w"))
        print(f"{name}: refs added to {len(items)} records", flush=True)


def dry_run(args):
    tok, _ = load_tokenizer(args.model_dir)
    es = load_es(args.es_corpus) if args.framing == "translate" else None
    for name in sorted(args.props)[args.shard_idx::args.shard_n]:
        items = build_items(name, tok, args.framing, es)
        lens = [len(it["ids"]) for it in items]
        print(f"{name}: items={len(items)} prompt_len min/median/max = "
              f"{min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}", flush=True)
    it = items[len(items) // 2]
    print("---- example prompt (last 700 chars) ----")
    print(tok.decode(it["ids"])[-700:])
    print(f"---- ref_{it['pol']}: {it['ref_' + it['pol']]!r}  | other twin: "
          f"{it['ref_' + ('alt' if it['pol'] == 'nat' else 'nat')]!r}")


def main():
    args = parse_args()
    if args.add_refs:
        return add_refs(args)
    if args.dry_run:
        return dry_run(args)
    props = sorted(args.props)[args.shard_idx::args.shard_n]
    args.out_root.mkdir(parents=True, exist_ok=True)
    es = load_es(args.es_corpus) if args.framing == "translate" else None
    model, tok = load_model(args.model_dir)
    seed_tag = "prescreen" if args.framing == "plain" else f"prescreen|{args.framing}"

    for name in props:
        out_path = args.out_root / f"{name}.json"
        if out_path.exists():
            print(f"{name}: exists, skip", flush=True)
            continue
        prop = PROPS[name]
        items = build_items(name, tok, args.framing, es)
        recs = [None] * len(items)
        for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
            lens = [len(items[i]["ids"]) for i in b]
            L = max(lens)
            ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros(len(b), L, dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, L - lens[r]:] = torch.tensor(items[i]["ids"])
                att[r, L - lens[r]:] = 1
            max_new = max(items[i]["max_new"] for i in b)
            torch.manual_seed(zlib.crc32(f"{name}|{seed_tag}|{bi}".encode()))
            with torch.no_grad():
                gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                     do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                     max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
            for r, i in enumerate(b):
                it = items[i]
                tail = tok.decode(gen[r, L:], skip_special_tokens=True)
                recs[i] = {"doc_id": it["doc_id"], "k": it["k"], "dist": it["dist"],
                           "pol": it["pol"], "label": classify(prop, tail, it["exp"]),
                           "tail": tail, "ref_nat": it["ref_nat"], "ref_alt": it["ref_alt"],
                           "ctx_tail": it["ctx_tail"]}
            if (bi + 1) % 50 == 0:
                print(f"{name}: batch {bi + 1}", flush=True)
        n = len(recs)
        sc = [r for r in recs if r["label"] is not None]
        cons = [r for r in sc if r["label"] == r["pol"]]
        print(f"{name}: items={n} scorable={len(sc)/max(n,1):.2f} "
              f"adherence={len(cons)/max(len(sc),1):.3f}", flush=True)
        json.dump({"property": name, "framing": args.framing, "records": recs},
                  open(out_path, "w"))
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
