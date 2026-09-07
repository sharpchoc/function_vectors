#!/usr/bin/env python
"""Run one grid cell x direction through the 0-shot steering protocol (GPU).

Focused runner for the grid (grid.py); the older steer_adherence.py is kept for the legacy
cells. Modes:

  screen    16-token rollouts, few docs, layer x dose grid, unjudged -> locates (L, alpha)
  headline  sentence rollouts, many docs, chosen (L, alpha) + counterfactual control
  bylayer   sentence rollouts, all layers at each layer's best screen dose
  refs      shared arms, no cell needed: unsteered baseline on the 0-shot items and the
            k>=REFERENCE_K in-context reference, both directions

Items are 0-SHOT: the first cue token of each document (no prior manifestation of either
convention in the prefix). sentence_caps / all_caps have no true k=0 site; their earliest
cue (k=1) is used and treated as 0-shot (user decision 2026-09-06).

Sentence rollout: generate up to ROLLOUT_MAX_NEW tokens, then cut at the first sentence
boundary; if none is reached the rollout is marked `capped` and the judge is told, so
truncation is not scored as incoherent.

Outputs: artifacts/style_properties/steering/grid/<mode>/<cell>__<direction>/<prop>.json
(refs -> .../grid/refs/<direction>/<prop>.json)
"""
import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
from src.sandbox.ext_styleprops.properties import PROPS
from src.sandbox.ext_styleprops.prescreen_adherence import load_model, classify
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len
from src.sandbox.ext_steerability.steer_read_dir_methods import Injector
from src.sandbox.ext_styleprops.grid import (BY_NAME, CELLS, DIRECTIONS, SCREEN_LAYERS,
                                             SCREEN_ALPHAS, REFERENCE_K, ROLLOUT_MAX_NEW,
                                             SCREEN_DOCS, HEADLINE_DOCS, BYLAYER_DOCS)

PROPS_DIR = REPO_ROOT / "dataset_files" / "style_properties" / "props"
POOL = REPO_ROOT / "task_splits" / "style_properties_pool.json"
VEC = ARTIFACTS_ROOT / "style_properties" / "steering_vectors_grid"
OUT = ARTIFACTS_ROOT / "style_properties" / "steering" / "grid"
# raw activation means have residual-scale norms, so their working doses sit far below the
# shared grid; the shared doses are still all run (comparability), these are extra.
MEANACT_EXTRA_ALPHAS = (0.125, 0.25)
SENT_END = re.compile(r"[.!?][\"')\]]?(\s|$)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("screen", "headline", "bylayer", "refs"), required=True)
    p.add_argument("--cell", default=None)
    p.add_argument("--direction", choices=DIRECTIONS, default=None)
    p.add_argument("--props", nargs="*", default=None)
    p.add_argument("--docs", type=int, default=None)
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--token_budget", type=int, default=14000)
    p.add_argument("--batch_cap", type=int, default=24)
    return p.parse_args()


def zeroshot_items(prop, tok, n_docs, pol):
    """0-shot context: a cue whose prefix contains NO manifestation of either convention.

    k==0 sites only. A document whose k=0 site was dropped at dataset-build time (cue too
    close to the document start) is skipped rather than silently contributing a k=1 item.
    Two properties (sentence_caps, all_caps) have no k=0 site anywhere - any text already
    shows its capitalisation - so their global minimum k (=1) is used and treated as 0-shot
    per the user decision of 2026-09-06."""
    data = json.load(open(PROPS_DIR / f"{prop}.json"))
    kmin = min(s["k"] for d in data["docs"] for s in d["sites"])
    items = []
    for d in data["docs"]:
        if len(items) >= n_docs:
            break
        cands = [s for s in d["sites"] if s["k"] == kmin]
        if not cands:
            continue
        s = cands[0]
        ids = tok(d[f"text_{pol}"]).input_ids
        cue = s["cue_idx"][pol]
        assert ids[cue] == s["cue_tok_id"], f"{prop}/{d['doc_id']}: cue id mismatch"
        items.append({"ids": ids[:cue + 1], "cue_pos": cue, "doc_id": d["doc_id"],
                      "k": s["k"], "exp": s["exp"][f"{pol}_ctx"],
                      "ctx": tok.decode(ids[max(0, cue - 60):cue + 1])})
    assert items and len({it["k"] for it in items}) == 1, \
        f"{prop}: 0-shot items must all share the minimum k, got {sorted({i['k'] for i in items})}"
    return items


def reference_items(prop, tok, n_docs, pol):
    """A cue with >= REFERENCE_K prior manifestations of `pol`'s convention in context."""
    data = json.load(open(PROPS_DIR / f"{prop}.json"))
    items = []
    for d in data["docs"][:n_docs]:
        cands = [s for s in d["sites"] if s["k"] >= REFERENCE_K]
        if not cands:
            continue
        s = min(cands, key=lambda s: s["k"])
        ids = tok(d[f"text_{pol}"]).input_ids
        cue = s["cue_idx"][pol]
        assert ids[cue] == s["cue_tok_id"]
        items.append({"ids": ids[:cue + 1], "cue_pos": cue, "doc_id": d["doc_id"],
                      "k": s["k"], "exp": s["exp"][f"{pol}_ctx"],
                      "ctx": tok.decode(ids[max(0, cue - 60):cue + 1])})
    return items


def cut_sentence(text):
    """-> (sentence, capped). Cut at the first sentence boundary; capped if none found."""
    m = SENT_END.search(text)
    if m:
        return text[:m.end()].rstrip(), False
    return text, True


def run(model, tok, inj, prop, items, vec, alpha, layer, tgt_pol, seed_tag,
        sentence, max_new, token_budget, batch_cap):
    inj.vec = None if vec is None else (alpha * torch.tensor(vec, dtype=torch.float32)).cuda()
    tails, capped = [None] * len(items), [False] * len(items)
    for bi, b in enumerate(batches_by_len(items, token_budget, batch_cap)):
        lens = [len(items[i]["ids"]) for i in b]
        L = max(lens)
        ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
        att = torch.zeros(len(b), L, dtype=torch.long)
        mask = torch.zeros(len(b), L, dtype=torch.bool)
        for r, i in enumerate(b):
            off = L - lens[r]
            ids[r, off:] = torch.tensor(items[i]["ids"])
            att[r, off:] = 1
            mask[r, off + items[i]["cue_pos"]] = True
        inj.mask = mask.cuda() if vec is not None else None
        torch.manual_seed(zlib.crc32(f"{prop}|{seed_tag}|{bi}".encode()))
        with torch.no_grad():
            gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                 do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
        inj.mask = None
        for r, i in enumerate(b):
            raw = tok.decode(gen[r, L:], skip_special_tokens=True)
            if sentence:
                tails[i], capped[i] = cut_sentence(raw)
            else:
                tails[i], capped[i] = raw, True
    inj.vec = None
    labels = [classify(PROPS[prop], t, it["exp"]) for t, it in zip(tails, items)]
    sc = [l for l in labels if l is not None]
    return dict(adherence_tgt=round(float(np.mean([l == tgt_pol for l in sc])), 4) if sc else float("nan"),
                scorable=round(len(sc) / max(len(labels), 1), 3), n=len(labels),
                tails=tails, capped=capped, ks=[it["k"] for it in items])


def main():
    args = parse_args()
    pool = sorted(json.load(open(POOL))["pass"])
    props = sorted(args.props) if args.props else pool
    cf_of = {p: pool[(i + 5) % len(pool)] for i, p in enumerate(pool)}
    model, tok = load_model(args.model_dir)
    inj = None

    def set_layer(block):
        nonlocal inj
        if inj is not None:
            inj.remove()
        inj = Injector(model, [block])

    if args.mode == "refs":
        for direction in DIRECTIONS:
            d = OUT / "refs" / direction
            d.mkdir(parents=True, exist_ok=True)
            for prop in props:
                f = d / f"{prop}.json"
                if f.exists():
                    print(f"{prop}/{direction}: exists", flush=True)
                    continue
                pol = direction
                zs = zeroshot_items(prop, tok, args.docs or HEADLINE_DOCS, "nat" if direction == "alt" else "alt")
                rf = reference_items(prop, tok, args.docs or HEADLINE_DOCS, pol)
                set_layer(0)
                res = {"property": prop, "direction": direction, "conditions": {}}
                res["ctx"] = [it["ctx"] for it in zs]
                res["ctx_ref"] = [it["ctx"] for it in rf]
                res["conditions"]["unsteered_zeroshot"] = run(
                    model, tok, inj, prop, zs, None, 0, 0, pol, f"unst_{direction}",
                    True, ROLLOUT_MAX_NEW, args.token_budget, args.batch_cap)
                if rf:
                    res["conditions"][f"reference_k{REFERENCE_K}"] = run(
                        model, tok, inj, prop, rf, None, 0, 0, pol, f"ref_{direction}",
                        True, ROLLOUT_MAX_NEW, args.token_budget, args.batch_cap)
                json.dump(res, open(f, "w"), indent=1)
                print(f"{prop}/{direction}: unsteered={res['conditions']['unsteered_zeroshot']['adherence_tgt']:.3f} "
                      f"ref={res['conditions'].get(f'reference_k{REFERENCE_K}', {}).get('adherence_tgt', float('nan')):.3f}",
                      flush=True)
        print("refs done", flush=True)
        return

    cell = BY_NAME[args.cell]
    direction = args.direction
    ctx_pol = "nat" if direction == "alt" else "alt"     # steer AWAY from this
    tag = f"{cell.name}__{direction}"
    out_dir = OUT / args.mode / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = SCREEN_ALPHAS + (MEANACT_EXTRA_ALPHAS if cell.technique == "meanact" else ())

    for prop in props:
        f = out_dir / f"{prop}.json"
        if f.exists():
            print(f"{tag}/{prop}: exists", flush=True)
            continue
        vz = np.load(VEC / cell.name / f"{prop}.npz")
        v = vz[f"v_{direction}"]
        n_docs = args.docs or {"screen": SCREEN_DOCS, "headline": HEADLINE_DOCS,
                               "bylayer": BYLAYER_DOCS}[args.mode]
        items = zeroshot_items(prop, tok, n_docs, ctx_pol)
        res = {"property": prop, "cell": cell.name, "direction": direction,
               "mode": args.mode, "ctx": [it["ctx"] for it in items],
               "n_sites_alt": int(vz["n_alt"]), "n_sites_nat": int(vz["n_nat"]),
               "conditions": {}}
        sentence = args.mode != "screen"
        max_new = ROLLOUT_MAX_NEW if sentence else 16

        if args.mode == "screen":
            for L in SCREEN_LAYERS:
                set_layer(L - 1)
                for a in alphas:
                    res["conditions"][f"L{L}_a{a}"] = run(
                        model, tok, inj, prop, items, v[L], a, L, direction,
                        f"scr{L}{a}", sentence, max_new, args.token_budget, args.batch_cap)
        else:
            pick = json.load(open(OUT / "screen_picks.json"))[tag][prop]
            layers = [pick["L"]] if args.mode == "headline" else list(SCREEN_LAYERS)
            for L in layers:
                a = pick["alpha"] if args.mode == "headline" else pick["per_layer"][str(L)]
                set_layer(L - 1)
                res["conditions"][f"steered_L{L}_a{a}"] = run(
                    model, tok, inj, prop, items, v[L], a, L, direction,
                    f"{args.mode}{L}{a}", sentence, max_new, args.token_budget, args.batch_cap)
            if args.mode == "headline":
                cfz = np.load(VEC / cell.name / f"{cf_of[prop]}.npz")
                set_layer(pick["L"] - 1)
                res["conditions"]["cfprop"] = run(
                    model, tok, inj, prop, items, cfz[f"v_{direction}"][pick["L"]],
                    pick["alpha"], pick["L"], direction, "cf", sentence, max_new,
                    args.token_budget, args.batch_cap)
            res["pick"] = pick
        json.dump(res, open(f, "w"), indent=1)
        best = max((c["adherence_tgt"] for c in res["conditions"].values()
                    if not np.isnan(c["adherence_tgt"])), default=float("nan"))
        print(f"{tag}/{prop}: {len(res['conditions'])} conds, best raw adherence={best:.3f}", flush=True)
    print(f"{args.mode} done: {tag}", flush=True)


if __name__ == "__main__":
    main()
