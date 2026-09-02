#!/usr/bin/env python
"""Style-property steering eval: sampled adherence at cue tokens under injection.

Injection: z <- z + alpha * v[L] additively at chosen positions of the prefix (Injector,
prefill-only), where v comes from build_steering_vectors.py (meandiff / rawalt) or the
sparse head-sum vectors (train_sparse_heads_props.py). Capture-layer L indexes
hidden_states (0=emb, l>=1 = block l-1 output), so the hook goes on block L-1.

Positions: 'evid' = evidence-token spans of all PRIOR manifestation sites in the prefix
(k>=1 sites only), i.e. read-side steering at the places the property would have been
read from; 'cue' = the cue token itself (write-side).

Readout (the only readout, per user decision): ONE T=1 seeded sample per (site,
condition), classified nat/alt/unscorable by the property classifier; report
adherence-to-target among scorable.

Modes:
  sweep — meandiff vector, nat->alt, evid injection, layer x alpha grid, subsampled docs.
  full  — best (L, alpha) per property from the sweep results, then: baseline both
          directions, meandiff evid/cue both directions, counterfactual-property control,
          rawalt arm, and (if present) the sparse head-sum vector arm.

Outputs: artifacts/style_properties/steering/{sweep,full}/<prop>.json (resumable).
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
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT
from src.sandbox.ext_styleprops.properties import PROPS
from src.sandbox.ext_styleprops.prescreen_adherence import load_model, classify
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len
from src.sandbox.ext_steerability.steer_read_dir_methods import Injector

PROPS_DIR = REPO_ROOT / "dataset_files" / "style_properties" / "props"
POOL_PATH = REPO_ROOT / "task_splits" / "style_properties_pool.json"
VEC_DIR = ARTIFACTS_ROOT / "style_properties" / "steering_vectors"
HEADSUM_DIR = ARTIFACTS_ROOT / "style_properties" / "sparse_heads"
OUT_ROOT = ARTIFACTS_ROOT / "style_properties" / "steering"

SWEEP_LAYERS = (2, 4, 6, 8, 10, 12, 16, 20, 24)   # capture-layer indices
SWEEP_ALPHAS = (2.0, 4.0, 8.0, 16.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("sweep", "full"), required=True)
    p.add_argument("--props", nargs="*", default=None)
    p.add_argument("--model_dir", type=Path, default=None)
    p.add_argument("--docs", type=int, default=None,
                   help="doc cap (default: 30 sweep / 80 full)")
    p.add_argument("--max_sites", type=int, default=None,
                   help="k>=1 sites per doc cap (default: 4 sweep / 8 full)")
    p.add_argument("--token_budget", type=int, default=16000)
    p.add_argument("--batch_cap", type=int, default=32)
    p.add_argument("--site", choices=("evid", "cue"), default="evid",
                   help="injection site: evidence tokens (read-side) or the cue token "
                        "(write-side; the function-vector analog). cue includes k=0 sites.")
    return p.parse_args()


def build_items(prop_name, tok, ctx_pol, n_docs, max_sites, require_evid=True):
    """Items = sites of the ctx-polarity twin: prefix through the cue token, with the PRIOR
    sites' evidence spans as the evid-injection mask. require_evid=True keeps k>=1 sites
    with a non-empty mask (evidence-site steering); False keeps every site incl. k=0
    (cue-site steering: the zero-shot analog)."""
    data = json.load(open(PROPS_DIR / f"{prop_name}.json"))
    items = []
    for d in data["docs"][:n_docs]:
        ids = tok(d[f"text_{ctx_pol}"]).input_ids
        kept = 0
        for si, s in enumerate(d["sites"]):
            if (require_evid and s["k"] == 0) or kept >= max_sites:
                continue
            cue = s["cue_idx"][ctx_pol]
            assert ids[cue] == s["cue_tok_id"]
            evid_pos = []
            for s2 in d["sites"][:si]:
                if s2["k"] < s["k"]:
                    e0, e1 = s2["evid_idx"][ctx_pol]
                    evid_pos.extend(range(e0, min(e1, cue - 1) + 1))
            if require_evid and not evid_pos:
                continue
            kept += 1
            items.append({"ids": ids[:cue + 1], "evid_pos": evid_pos, "cue_pos": cue,
                          "max_new": s["max_new"], "doc_id": d["doc_id"], "k": s["k"],
                          "exp": s["exp"][f"{ctx_pol}_ctx"]})
    return items


def run_condition(model, tok, inj, prop, items, vec, block_layer, alpha, pos_mode,
                  tgt_pol, seed_tag, token_budget, batch_cap):
    """One condition -> (adherence_to_tgt, scorable_frac, n)."""
    inj.vec = None if vec is None else (alpha * vec.float()).cuda()
    labels, tails = [None] * len(items), [None] * len(items)
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
            pos = items[i]["evid_pos"] if pos_mode == "evid" else [items[i]["cue_pos"]]
            for p_ in pos:
                mask[r, off + p_] = True
        inj.mask = mask.cuda() if vec is not None else None
        max_new = max(items[i]["max_new"] for i in b)
        torch.manual_seed(zlib.crc32(f"{prop.name}|{seed_tag}|{bi}".encode()))
        with torch.no_grad():
            gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(),
                                 do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
        inj.mask = None
        for r, i in enumerate(b):
            tail = tok.decode(gen[r, L:], skip_special_tokens=True)
            labels[i] = classify(prop, tail, items[i]["exp"])
            tails[i] = tail
    inj.vec = None
    sc = [l for l in labels if l is not None]
    adh = float(np.mean([l == tgt_pol for l in sc])) if sc else float("nan")
    return adh, len(sc) / max(len(labels), 1), len(labels), tails


def main():
    args = parse_args()
    pool = sorted(json.load(open(POOL_PATH))["pass"])
    props = sorted(args.props) if args.props else pool
    n_docs = args.docs or (30 if args.mode == "sweep" else 80)
    max_sites = args.max_sites or (4 if args.mode == "sweep" else 8)
    out_dir = OUT_ROOT / (args.mode if args.site == "evid" else f"{args.mode}_cue")
    out_dir.mkdir(parents=True, exist_ok=True)
    req_evid = args.site == "evid"
    alphas = SWEEP_ALPHAS if args.site == "evid" else SWEEP_ALPHAS + (32.0,)
    vec_srcs = ("meandiff",) if args.site == "evid" else ("cuediff", "meandiff")
    # counterfactual pairing: rotate the pool by 5 (fixed derangement)
    cf_of = {p: pool[(i + 5) % len(pool)] for i, p in enumerate(pool)}

    model, tok = load_model(args.model_dir)
    inj = None

    def set_layer(block_layer):
        nonlocal inj
        if inj is not None:
            inj.remove()
        inj = Injector(model, [block_layer])

    for name in props:
        out_path = out_dir / f"{name}.json"
        if out_path.exists():
            print(f"{name}: exists, skip", flush=True)
            continue
        prop = PROPS[name]
        vz = np.load(VEC_DIR / f"{name}.npz")
        res = {"property": name, "mode": args.mode, "n_docs": n_docs, "conditions": {}}

        res["site"] = args.site

        def record(cname, adh, scf, n, tails=None, ks=None):
            # tails (+ per-item k) are stored so a classifier fix / k-split is a rescoring
            res["conditions"][cname] = {"adherence_tgt": round(adh, 4),
                                        "scorable": round(scf, 3), "n": n, "tails": tails,
                                        "ks": ks}
            print(f"{name} | {cname}: adh={adh:.3f} scorable={scf:.2f} n={n}", flush=True)

        if args.mode == "sweep":
            items = build_items(name, tok, "nat", n_docs, max_sites, require_evid=req_evid)
            ks = [it["k"] for it in items]
            set_layer(SWEEP_LAYERS[0] - 1)
            record("baseline_nat2alt", *run_condition(model, tok, inj, prop, items, None,
                   None, 0, args.site, "alt", "base", args.token_budget, args.batch_cap), ks)
            for vsrc in vec_srcs:
                for L in SWEEP_LAYERS:
                    set_layer(L - 1)
                    v = torch.tensor(vz[vsrc][L], dtype=torch.float16)
                    for a in alphas:
                        record(f"{vsrc}_{args.site}_L{L}_a{a}", *run_condition(model, tok, inj,
                               prop, items, v, L - 1, a, args.site, "alt", f"{vsrc}L{L}a{a}",
                               args.token_budget, args.batch_cap), ks)
        else:
            sweep = json.load(open(out_dir.parent / ("sweep" if args.site == "evid" else "sweep_cue")
                                   / f"{name}.json"))["conditions"]
            valid = [c for c in sweep if any(c.startswith(v) for v in vec_srcs)
                     and not np.isnan(sweep[c]["adherence_tgt"])]
            if valid:
                best = max(valid, key=lambda c: sweep[c]["adherence_tgt"])
                vsrc = best.split("_")[0]
                L = int(best.split("_L")[1].split("_")[0])
                a = float(best.split("_a")[1])
            else:
                best, vsrc, L, a = "fallback", vec_srcs[0], 10, 4.0
            res["best_from_sweep"] = {"cond": best, "vector": vsrc, "L": L, "alpha": a}
            items_n = build_items(name, tok, "nat", n_docs, max_sites, require_evid=req_evid)
            items_a = build_items(name, tok, "alt", n_docs, max_sites, require_evid=req_evid)
            ks_n, ks_a = [it["k"] for it in items_n], [it["k"] for it in items_a]
            set_layer(L - 1)
            v = torch.tensor(vz[vsrc][L], dtype=torch.float16)
            cfz = np.load(VEC_DIR / f"{cf_of[name]}.npz")
            vcf = torch.tensor(cfz[vsrc][L], dtype=torch.float16)
            res["cf_property"] = cf_of[name]
            S = args.site
            R = lambda *aa, **kw: run_condition(model, tok, inj, prop, *aa, **kw)

            record("baseline_nat2alt", *R(items_n, None, None, 0, S, "alt", "bn",
                   args.token_budget, args.batch_cap), ks_n)
            record(f"{vsrc}_{S}_nat2alt", *R(items_n, v, L - 1, a, S, "alt", "mn",
                   args.token_budget, args.batch_cap), ks_n)
            if S == "evid":   # write-side probe of the read-side vector (legacy condition)
                record(f"{vsrc}_cue_nat2alt", *R(items_n, v, L - 1, a, "cue", "alt", "md",
                       args.token_budget, args.batch_cap), ks_n)
            record(f"cfprop_{S}_nat2alt", *R(items_n, vcf, L - 1, a, S, "alt", "cf",
                   args.token_budget, args.batch_cap), ks_n)
            record("baseline_alt2nat", *R(items_a, None, None, 0, S, "nat", "ba",
                   args.token_budget, args.batch_cap), ks_a)
            record(f"{vsrc}_{S}_alt2nat", *R(items_a, -v, L - 1, a, S, "nat", "ma",
                   args.token_budget, args.batch_cap), ks_a)
            # the other vector source at ITS best sweep setting (comparison)
            for other in vec_srcs:
                if other == vsrc:
                    continue
                ov = [c for c in valid if c.startswith(other)]
                if ov:
                    ob = max(ov, key=lambda c: sweep[c]["adherence_tgt"])
                    oL = int(ob.split("_L")[1].split("_")[0]); oa = float(ob.split("_a")[1])
                    set_layer(oL - 1)
                    record(f"{other}_{S}_nat2alt_best", *R(items_n,
                           torch.tensor(vz[other][oL], dtype=torch.float16), oL - 1, oa, S,
                           "alt", "ob", args.token_budget, args.batch_cap), ks_n)
                    set_layer(L - 1)
            raw_key = "rawalt" if vsrc == "meandiff" else "rawalt_cue"
            vraw = torch.tensor(vz[raw_key][L], dtype=torch.float16)
            for ar in (0.5, 1.0, 2.0):
                record(f"{raw_key}_{S}_nat2alt_a{ar}", *R(items_n, vraw, L - 1, ar, S, "alt",
                       f"rw{ar}", args.token_budget, args.batch_cap), ks_n)
            hs_dir = HEADSUM_DIR if S == "evid" else HEADSUM_DIR.parent / "sparse_heads_cue"
            hs_path = hs_dir / f"{name}.npz"
            if hs_path.exists():
                hz = np.load(hs_path)
                vh = torch.tensor(hz["v_headsum"], dtype=torch.float16)
                hL = int(hz["cap_layer"]) if "cap_layer" in hz else L
                set_layer(hL - 1)
                for ah in (1.0, 2.0, 4.0, 8.0):
                    record(f"headsum_{S}_nat2alt_a{ah}", *R(items_n, vh, hL - 1, ah, S, "alt",
                           f"hs{ah}", args.token_budget, args.batch_cap), ks_n)
        json.dump(res, open(out_path, "w"), indent=1)
    print(f"{args.mode} done", flush=True)


if __name__ == "__main__":
    main()
