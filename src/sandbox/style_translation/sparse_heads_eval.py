#!/usr/bin/env python
"""Step 5c — evaluate the sparse-head style vectors on the 80 held-out texts per family (GPU).

Prompt = k = 0 item (Spanish + English up to the first cue token), texts 120..199 (doc_id order).
Arms (alt direction, injected at the selected layer L* with steer_hooks.CueSteer, alpha = 1):
  base        no steering
  sparse_w    v_f = C_alt[f]^T c           (learned coefficients)
  sparse_unw  v_f = sum of C_alt[f, h] over heads with c > 0.2   (canonical unweighted FV construction)
  sparse_cf   the NEXT family's sparse_w vector (cyclic in --families order) — counterfactual control
  sparse_unw_hi  (optional) indicator sum over heads with c > 0.8
One seeded T=1 sample, 48 new tokens, cut at the sentence end (capped flag), style by scoring.decide.
Records (step-4 confirm schema + arm/layer/lam/n_heads/cf_family) -> sparse_heads/eval/<family>.json;
grade with judge_rollouts.py --dir <that dir>.
--lambda_curve: sparse_w at every lambda of the selected layer -> sparse_heads/eval_lambda/<family>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.rollout import load_model
from src.sandbox.style_translation.scoring import decide, cut_sentence
from src.sandbox.style_translation.steer_hooks import unit_test
from src.sandbox.style_translation.steer_screen import k0_items, sample

ROOT = ARTIFACTS_ROOT / "style_translation" / "sparse_heads"
HEADS, TRAIN = ROOT / "heads", ROOT / "train"
N_FIT, MAX_NEW, BATCH = 120, 48, 16


def vectors(fams, c, heads_02, heads_08):
    V = {}
    for fam in fams:
        C = torch.from_numpy(np.load(HEADS / f"{fam}.npz")["C_alt"])           # (448, 4096)
        V[fam] = {"w": (C * c[:, None]).sum(0), "unw": C[heads_02].sum(0), "unw_hi": C[heads_08].sum(0)}
    return V


def run_arm(model, tok, items, layer, vec, alpha, fam, arm, extra):
    recs = []
    for bi in range(0, len(items), BATCH):
        chunk = items[bi: bi + BATCH]
        tails = sample(model, tok, chunk, layer, vec, alpha, f"{fam}|sparse|{arm}|{bi}", MAX_NEW)
        for it, raw in zip(chunk, tails):
            cut, capped = cut_sentence(raw)
            dec = decide(fam, it["seg_prefix"], cut, it["next_nat"], it["next_alt"])
            recs.append({**{k: it[k] for k in ("doc_id", "family", "k", "cue_tok", "seg_prefix", "next_nat", "next_alt", "ref_sentence", "context_tail", "es_text")},
                         "style": "alt", "arm": arm, "layer": layer, "alpha": alpha, **extra,
                         "tail_raw": raw, "tail": cut, "capped": capped, "decision": dec, "style_ok": dec == "alt", "judge": None})
    return recs


def summary(recs, arm):
    sel = [r for r in recs if r["arm"] == arm]
    return f"alt {np.mean([r['decision'] == 'alt' for r in sel]):.2f} nat {np.mean([r['decision'] == 'nat' for r in sel]):.2f} unsc {np.mean([r['decision'] is None for r in sel]):.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--with_hi", action="store_true")
    ap.add_argument("--lambda_curve", action="store_true")
    args = ap.parse_args()
    sel = json.load(open(ROOT / "selection.json"))
    L, lam = sel["layer"], sel["lambda"]
    c = torch.tensor(sel["c"], dtype=torch.float32)
    model, tok = load_model()
    assert unit_test(model, tok, layer=L), "hook unit test failed"
    print(f"hook unit test passed | L*={L} lambda*={lam} heads c>0.2: {len(sel['heads_02'])}", flush=True)
    fams = args.families
    if args.lambda_curve:
        out = ROOT / "eval_lambda"; out.mkdir(parents=True, exist_ok=True)
        lams = sorted(float(f.stem[len("lambda"):]) for f in (TRAIN / f"L{L}").glob("lambda*.pt"))
        cs = {l: torch.load(TRAIN / f"L{L}" / f"lambda{l:g}.pt", map_location="cpu", weights_only=False)["c"] for l in lams}
        for fam in fams:
            if (out / f"{fam}.json").exists():
                print(f"{fam}: exists, skip", flush=True); continue
            items = k0_items(fam)[N_FIT:]
            C = torch.from_numpy(np.load(HEADS / f"{fam}.npz")["C_alt"])
            recs = []
            for l in lams:
                v = (C * cs[l][:, None]).sum(0)
                recs += run_arm(model, tok, items, L, v, 1.0, fam, f"sparse_w_lam{l:g}", {"lam": l, "n_heads": int((cs[l] > 0.2).sum()), "cf_family": None})
                print(f"{fam}: lambda {l:g} heads {(cs[l] > 0.2).sum().item():3d} | {summary(recs, f'sparse_w_lam{l:g}')}", flush=True)
            json.dump(recs, open(out / f"{fam}.json", "w"), ensure_ascii=False)
        print("lambda curve done", flush=True); return

    out = ROOT / "eval"; out.mkdir(parents=True, exist_ok=True)
    V = vectors(fams, c, sel["heads_02"], sel["heads_08"])
    for i, fam in enumerate(fams):
        if (out / f"{fam}.json").exists():
            print(f"{fam}: exists, skip", flush=True); continue
        items = k0_items(fam)[N_FIT:]
        assert len(items) == 80, (fam, len(items))
        cf = fams[(i + 1) % len(fams)]
        recs = run_arm(model, tok, items, L, None, 0.0, fam, "base", {"lam": None, "n_heads": 0, "cf_family": None})
        recs += run_arm(model, tok, items, L, V[fam]["w"], 1.0, fam, "sparse_w", {"lam": lam, "n_heads": len(sel["heads_02"]), "cf_family": None})
        recs += run_arm(model, tok, items, L, V[fam]["unw"], 1.0, fam, "sparse_unw", {"lam": lam, "n_heads": len(sel["heads_02"]), "cf_family": None})
        recs += run_arm(model, tok, items, L, V[cf]["w"], 1.0, fam, "sparse_cf", {"lam": lam, "n_heads": len(sel["heads_02"]), "cf_family": cf})
        if args.with_hi:
            recs += run_arm(model, tok, items, L, V[fam]["unw_hi"], 1.0, fam, "sparse_unw_hi", {"lam": lam, "n_heads": len(sel["heads_08"]), "cf_family": None})
        json.dump(recs, open(out / f"{fam}.json", "w"), ensure_ascii=False)
        for arm in ("base", "sparse_w", "sparse_unw", "sparse_cf") + (("sparse_unw_hi",) if args.with_hi else ()):
            print(f"{fam}: arm {arm:13s} | {summary(recs, arm)}", flush=True)
    print("eval done", flush=True)


if __name__ == "__main__":
    main()
