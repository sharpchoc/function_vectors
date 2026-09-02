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

Output: artifacts/style_properties/prescreen/<prop>.json (resumable per property)
        [{doc_id, k, dist, pol, label, tail}]
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--props", nargs="*", default=sorted(PROPS))
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "style_properties" / "prescreen")
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=16000)
    p.add_argument("--batch_cap", type=int, default=32)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--shard_n", type=int, default=1)
    return p.parse_args()


def load_model(model_dir):
    """GPT-J fp16; snapshot glob filtered to complete snapshots (DECISIONS 2026-08-19:
    a stale snapshot with only a stray safetensors file breaks the naive glob)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    md = model_dir
    if md is None:
        snaps = sorted(Path("/workspace/.cache/huggingface/hub/"
                            "models--EleutherAI--gpt-j-6b/snapshots").glob("*"))
        snaps = [s for s in snaps if (s / "config.json").exists()]
        assert snaps, "no complete GPT-J snapshot found; pass --model_dir"
        md = snaps[-1]
    tok = AutoTokenizer.from_pretrained(md)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.float16).cuda().eval()
    return model, tok


def build_items(prop_name, tok):
    data = json.load(open(PROPS_DIR / f"{prop_name}.json"))
    items = []
    for d in data["docs"]:
        ids = {pol: tok(d[f"text_{pol}"]).input_ids for pol in ("nat", "alt")}
        for s in d["sites"]:
            for pol in ("nat", "alt"):
                cue = s["cue_idx"][pol]
                assert ids[pol][cue] == s["cue_tok_id"], \
                    f"{prop_name}/{d['doc_id']}/k{s['k']}/{pol}: cue token mismatch"
                items.append({
                    "ids": ids[pol][:cue + 1],
                    "max_new": s["max_new"],
                    "doc_id": d["doc_id"], "k": s["k"], "dist": s["dist"][pol],
                    "pol": pol, "exp": s["exp"][f"{pol}_ctx"],
                })
    return items


def classify(prop, tail, exp):
    lab = prop.classify(tail)
    if lab is not None:
        return lab
    for label, e in sorted(exp.items(), key=lambda kv: -len(kv[1])):
        if e and tail.startswith(e):
            return label
    return None


def main():
    args = parse_args()
    props = sorted(args.props)[args.shard_idx::args.shard_n]
    args.out_root.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(args.model_dir)

    for name in props:
        out_path = args.out_root / f"{name}.json"
        if out_path.exists():
            print(f"{name}: exists, skip", flush=True)
            continue
        prop = PROPS[name]
        items = build_items(name, tok)
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
            torch.manual_seed(zlib.crc32(f"{name}|prescreen|{bi}".encode()))
            with torch.no_grad():
                gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                     do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                     max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
            for r, i in enumerate(b):
                it = items[i]
                tail = tok.decode(gen[r, L:], skip_special_tokens=True)
                recs[i] = {"doc_id": it["doc_id"], "k": it["k"], "dist": it["dist"],
                           "pol": it["pol"], "label": classify(prop, tail, it["exp"]),
                           "tail": tail}
            if (bi + 1) % 50 == 0:
                print(f"{name}: batch {bi + 1}", flush=True)
        n = len(recs)
        sc = [r for r in recs if r["label"] is not None]
        cons = [r for r in sc if r["label"] == r["pol"]]
        print(f"{name}: items={n} scorable={len(sc)/max(n,1):.2f} "
              f"adherence={len(cons)/max(len(sc),1):.3f}", flush=True)
        json.dump({"property": name, "records": recs}, open(out_path, "w"))
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
