#!/usr/bin/env python
"""Style-property steering eval: sampled adherence at decision points under injection.

Injection: z <- z + alpha * v[L] additively at chosen positions of the prefix (Injector,
prefill-only), where v comes from build_steering_vectors.py (meandiff / rawalt) or the
sparse head-sum vectors (train_sparse_heads_props.py). Capture-layer L indexes
hidden_states (0=emb, l>=1 = block l-1 output), so the hook goes on block L-1.

Positions: 'evid' = evidence-token spans of all PRIOR manifestation sites in the prefix
(k>=1 sites only), i.e. read-side steering at the places the property would have been
read from; 'dec' = the decision token itself (write-side).

Readout (the only readout, per user decision): ONE T=1 seeded sample per (site,
condition), classified nat/alt/unscorable by the property classifier; report
adherence-to-target among scorable.

Modes:
  sweep — meandiff vector, nat->alt, evid injection, layer x alpha grid, subsampled docs.
  full  — best (L, alpha) per property from the sweep results, then: baseline both
          directions, meandiff evid/dec both directions, counterfactual-property control,
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
    return p.parse_args()


def build_items(prop_name, tok, ctx_pol, n_docs, max_sites):
    """Items = k>=1 sites of the ctx-polarity twin: prefix through the decision token,
    with the PRIOR sites' evidence spans as the evid-injection mask."""
    data = json.load(open(PROPS_DIR / f"{prop_name}.json"))
    items = []
    for d in data["docs"][:n_docs]:
        ids = tok(d[f"text_{ctx_pol}"]).input_ids
        kept = 0
        for si, s in enumerate(d["sites"]):
            if s["k"] == 0 or kept >= max_sites:
                continue
            dec = s["dec_idx"][ctx_pol]
            assert ids[dec] == s["dec_tok_id"]
            evid_pos = []
            for s2 in d["sites"][:si]:
                if s2["k"] < s["k"]:
                    e0, e1 = s2["evid_idx"][ctx_pol]
                    evid_pos.extend(range(e0, min(e1, dec - 1) + 1))
            if not evid_pos:
                continue
            kept += 1
            items.append({"ids": ids[:dec + 1], "evid_pos": evid_pos, "dec_pos": dec,
                          "max_new": s["max_new"], "doc_id": d["doc_id"], "k": s["k"],
                          "exp": s["exp"][f"{ctx_pol}_ctx"]})
    return items


def run_condition(model, tok, inj, prop, items, vec, block_layer, alpha, pos_mode,
                  tgt_pol, seed_tag, token_budget, batch_cap):
    """One condition -> (adherence_to_tgt, scorable_frac, n)."""
    inj.vec = None if vec is None else (alpha * vec.float()).cuda()
    labels = [None] * len(items)
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
            pos = items[i]["evid_pos"] if pos_mode == "evid" else [items[i]["dec_pos"]]
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
    inj.vec = None
    sc = [l for l in labels if l is not None]
    adh = float(np.mean([l == tgt_pol for l in sc])) if sc else float("nan")
    return adh, len(sc) / max(len(labels), 1), len(labels)


def main():
    args = parse_args()
    pool = sorted(json.load(open(POOL_PATH))["pass"])
    props = sorted(args.props) if args.props else pool
    n_docs = args.docs or (30 if args.mode == "sweep" else 80)
    max_sites = args.max_sites or (4 if args.mode == "sweep" else 8)
    out_dir = OUT_ROOT / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
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

        def record(cname, adh, scf, n):
            res["conditions"][cname] = {"adherence_tgt": round(adh, 4),
                                        "scorable": round(scf, 3), "n": n}
            print(f"{name} | {cname}: adh={adh:.3f} scorable={scf:.2f} n={n}", flush=True)

        if args.mode == "sweep":
            items = build_items(name, tok, "nat", n_docs, max_sites)
            set_layer(SWEEP_LAYERS[0] - 1)
            adh, scf, n = run_condition(model, tok, inj, prop, items, None, None, 0,
                                        "evid", "alt", "base", args.token_budget,
                                        args.batch_cap)
            record("baseline_nat2alt", adh, scf, n)
            for L in SWEEP_LAYERS:
                set_layer(L - 1)
                v = torch.tensor(vz["meandiff"][L], dtype=torch.float16)
                for a in SWEEP_ALPHAS:
                    adh, scf, n = run_condition(model, tok, inj, prop, items, v, L - 1,
                                                a, "evid", "alt", f"L{L}a{a}",
                                                args.token_budget, args.batch_cap)
                    record(f"meandiff_evid_L{L}_a{a}", adh, scf, n)
        else:
            sweep = json.load(open(OUT_ROOT / "sweep" / f"{name}.json"))["conditions"]
            best = max((c for c in sweep if c.startswith("meandiff")),
                       key=lambda c: sweep[c]["adherence_tgt"])
            L = int(best.split("_L")[1].split("_")[0])
            a = float(best.split("_a")[1])
            res["best_from_sweep"] = {"cond": best, "L": L, "alpha": a}
            items_n = build_items(name, tok, "nat", n_docs, max_sites)
            items_a = build_items(name, tok, "alt", n_docs, max_sites)
            set_layer(L - 1)
            v = torch.tensor(vz["meandiff"][L], dtype=torch.float16)
            cfz = np.load(VEC_DIR / f"{cf_of[name]}.npz")
            vcf = torch.tensor(cfz["meandiff"][L], dtype=torch.float16)
            res["cf_property"] = cf_of[name]

            record("baseline_nat2alt", *run_condition(model, tok, inj, prop, items_n,
                   None, None, 0, "evid", "alt", "bn", args.token_budget, args.batch_cap))
            record("meandiff_evid_nat2alt", *run_condition(model, tok, inj, prop, items_n,
                   v, L - 1, a, "evid", "alt", "mn", args.token_budget, args.batch_cap))
            record("meandiff_dec_nat2alt", *run_condition(model, tok, inj, prop, items_n,
                   v, L - 1, a, "dec", "alt", "md", args.token_budget, args.batch_cap))
            record("cfprop_evid_nat2alt", *run_condition(model, tok, inj, prop, items_n,
                   vcf, L - 1, a, "evid", "alt", "cf", args.token_budget, args.batch_cap))
            record("baseline_alt2nat", *run_condition(model, tok, inj, prop, items_a,
                   None, None, 0, "evid", "nat", "ba", args.token_budget, args.batch_cap))
            record("meandiff_evid_alt2nat", *run_condition(model, tok, inj, prop, items_a,
                   -v, L - 1, a, "evid", "nat", "ma", args.token_budget, args.batch_cap))
            vraw = torch.tensor(vz["rawalt"][L], dtype=torch.float16)
            for ar in (0.5, 1.0, 2.0):
                record(f"rawalt_evid_nat2alt_a{ar}", *run_condition(model, tok, inj,
                       prop, items_n, vraw, L - 1, ar, "evid", "alt", f"rw{ar}",
                       args.token_budget, args.batch_cap))
            hs_path = HEADSUM_DIR / f"{name}.npz"
            if hs_path.exists():
                hz = np.load(hs_path)
                vh = torch.tensor(hz["v_headsum"], dtype=torch.float16)
                for ah in (1.0, 2.0, 4.0, 8.0):
                    record(f"headsum_evid_nat2alt_a{ah}", *run_condition(model, tok, inj,
                           prop, items_n, vh, L - 1, ah, "evid", "alt", f"hs{ah}",
                           args.token_budget, args.batch_cap))
        json.dump(res, open(out_path, "w"), indent=1)
    print(f"{args.mode} done", flush=True)


if __name__ == "__main__":
    main()
