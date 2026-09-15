#!/usr/bin/env python
"""Step 7b — confirm evidence-token steering on all k-shot prompts of a family (GPU).

Arms per family (all prompts, one seeded T=1 sample, 48 new tokens, sentence cut for text / code cut for code, capped):
  base_nat, base_alt         alpha = 0 (unsteered k-shot prompt, both context poles)
  nat2alt_top1/2             nat-context prompts, -u_nat at the evidence tokens, top-2 screen settings
  alt2nat_top1/2             alt-context prompts, +u_nat, top-2 screen settings
  nat2alt_cf, alt2nat_cf     ANOTHER family's read vector (cyclic within the shard) at the same positions
                             and the family's own top-1 (L, alpha) — skipped with --no_control
--k_ctx selects the prompt (3 = the text-study default; 1 = single in-context instance); the screen is read from
read_steer/<screen_tag> (default screen / screen_k<k>) over every (layer, alpha) cell it contains, records go to
read_steer/<out_tag> (default confirm / confirm_k<k>). The cue position is never steered (read_steer_screen.k_items).
Grading = steps 3/4 (scoring.decide at the next decision; judge_rollouts.py --dir for faithfulness / correctness).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.rollout import load_model
from src.sandbox.style_translation.scoring import decide, cut_sentence, cut_code
from src.sandbox.style_translation.ml_families import ML_FAMILY
from src.sandbox.style_translation.read_steer_screen import k_items, read_vectors, sample_positions, DIRECTIONS, K_CTX
import src.sandbox.style_translation.read_steer_screen as rss
from src.sandbox.style_translation.models import paths as model_paths, arch

MAX_NEW = 48
FIELDS = ("doc_id", "family", "k", "cue_tok", "seg_prefix", "next_nat", "next_alt", "ref_sentence", "context_tail", "es_text")


def top_settings(fam, direction, screen_tag="screen", n=2):
    recs = json.load(open(rss.ROOT / screen_tag / f"{fam}.json"))
    target = DIRECTIONS[direction][1]
    cells = sorted({(r["layer"], r["alpha"]) for r in recs if r["direction"] == direction})
    rates = []
    for layer, a in cells:
        sel = [r for r in recs if r["direction"] == direction and r["layer"] == layer and r["alpha"] == a]
        rates.append((np.mean([r["decision"] == target for r in sel]), -a, -layer, layer, a))
    rates.sort(reverse=True)
    return [(layer, a, rate) for rate, _, _, layer, a in rates[:n]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES], help="this shard's families; cf cycles within")
    ap.add_argument("--model", default="gptj", help="models.MODELS key (weights + artifact/results folders)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k_ctx", type=int, default=K_CTX, help="in-context instances in the steered prompt (prompt k)")
    ap.add_argument("--screen_tag", default=None, help="screen subdir under read_steer/ (default screen / screen_k<k>)")
    ap.add_argument("--out_tag", default=None, help="output subdir under read_steer/ (default confirm / confirm_k<k>)")
    ap.add_argument("--no_control", action="store_true", help="skip the counterfactual (other family's read vector) arms")
    args = ap.parse_args()
    rss.configure(args.model)
    screen_tag = args.screen_tag or ("screen" if args.k_ctx == K_CTX else f"screen_k{args.k_ctx}")
    out_tag = args.out_tag or ("confirm" if args.k_ctx == K_CTX else f"confirm_k{args.k_ctx}")
    (rss.ROOT / out_tag).mkdir(parents=True, exist_ok=True)
    fams = args.families
    model, tok = load_model(model=args.model)
    for fi, fam in enumerate(fams):
        out_path = rss.ROOT / out_tag / f"{fam}.json"
        if out_path.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        code = getattr(ML_FAMILY.get(fam), "domain", "text") == "code"
        u = read_vectors(fam); cf_fam = fams[(fi + 1) % len(fams)]; u_cf = None if args.no_control else read_vectors(cf_fam)
        items = {pole: k_items(fam, pole, args.k_ctx) for pole in ("nat", "alt")}
        arms = [("base_nat", "nat", None, 0, 0.0, None, None), ("base_alt", "alt", None, 0, 0.0, None, None)]
        for direction, (ctx_pole, target) in DIRECTIONS.items():
            sign = 1 if target == "nat" else -1
            tops = top_settings(fam, direction, screen_tag, 2)
            for rank, (layer, a, _) in enumerate(tops, 1):
                arms.append((f"{direction}_top{rank}", ctx_pole, target, layer, a, u[layer] * sign, None))
            if not args.no_control:
                layer, a, _ = tops[0]
                arms.append((f"{direction}_cf", ctx_pole, target, layer, a, u_cf[layer] * sign, cf_fam))
        recs = []
        for arm, ctx_pole, target, layer, alpha, v, cf in arms:
            its = items[ctx_pole]
            for bi in range(0, len(its), args.batch):
                chunk = its[bi:bi + args.batch]
                tails = sample_positions(model, tok, chunk, layer, v, alpha, f"{fam}|rconfirm{args.k_ctx}|{arm}|{bi}", MAX_NEW)
                for it, raw in zip(chunk, tails):
                    cut, capped = cut_code(raw, fam) if code else cut_sentence(raw)
                    d = decide(fam, it["seg_prefix"], cut, it["next_nat"], it["next_alt"])
                    recs.append({k: it[k] for k in FIELDS} | {"style": target or ctx_pole, "context": ctx_pole, "k_ctx": args.k_ctx, "arm": arm,
                                 "direction": arm.split("_")[0] if target else None,
                                 "target": target, "layer": layer, "alpha": alpha, "cf_family": cf, "n_positions": len(it["positions"]),
                                 "tail_raw": raw, "tail": cut, "capped": capped, "decision": d,
                                 "style_ok": (d == target) if target else (d == ctx_pole), "judge": None})
            sel = [r for r in recs if r["arm"] == arm]
            print(f"{fam}: arm {arm:12s} L={layer:2d} a={alpha:<4} | nat {np.mean([r['decision']=='nat' for r in sel]):.2f} "
                  f"alt {np.mean([r['decision']=='alt' for r in sel]):.2f} unscorable {np.mean([r['decision'] is None for r in sel]):.2f} "
                  f"capped {np.mean([r['capped'] for r in sel]):.2f}", flush=True)
        json.dump(recs, open(out_path, "w"), ensure_ascii=False)
    print("read confirm done", flush=True)


if __name__ == "__main__":
    main()
