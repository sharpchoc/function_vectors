#!/usr/bin/env python
"""Subspace patch with the document's OWN counterpart activation as target (GPU), towards the ALTERNATIVE convention.

Same subspace as write_subspace_patch.py (top-r uncentered right singular vectors of the k = 4 pair differences, training documents), but the
0-shot cue token's coordinates are replaced by the coordinates of the SAME held-out document's k = 4 alternative prompt (cue_pairs_layers/<fam>_k4.npz),
not by the training mean. Arms per layer: own1, own2, own10 (rank 1 / 2 / 10) and full (replace the whole cue activation by the document's k = 4
alternative activation); base = unsteered. Held-out documents without a stored pair (cue token differs) are dropped from every arm.
Records -> <out_dir>/<family>.json (judge afterwards: judge_rollouts.py --dir <out_dir>).
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
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.rollout import load_model
from src.sandbox.style_translation.steer_hooks import CueSubspacePatch, CueReplace, unit_test, unit_test_subspace_patch
from src.sandbox.style_translation.logprob_margin import candidates
from src.sandbox.style_translation.write_sweep import held_out_items
from src.sandbox.style_translation.write_subspace_patch import run_arm, _Null

RANKS = (1, 2, 10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="*", required=True)
    ap.add_argument("--layers", nargs="*", type=int, default=list(range(20, 29))); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out_dir", type=Path, required=True); ap.add_argument("--batch", type=int, default=20); ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    MP = model_paths(args.model); SRC = MP["prompt_pairs"].parent / "cue_pairs_layers"; args.out_dir.mkdir(parents=True, exist_ok=True)
    lex_p = MP["prompts"].parent / "scoring_lexicon.json"; lexicon = json.load(open(lex_p))["lexicon"] if lex_p.exists() else {}
    model, tok = load_model(model=args.model)
    assert unit_test(model, tok, layer=6) and unit_test_subspace_patch(model, tok, layer=24), "hook unit test failed"
    for fam in args.families:
        out = args.out_dir / f"{fam}.json"
        if out.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        z = np.load(SRC / f"{fam}_k{args.k}.npz"); docs = list(z["doc_id"]); layers = list(z["layers"]); tr = ~z["heldout"]
        items = [it for it in candidates(fam, tok, held_out_items(MP["prompts"], fam)) if it["doc_id"] in set(docs)][: args.limit]
        texts = {it["doc_id"]: tok.decode(it["prompt_ids"]) for it in items}; di = [docs.index(it["doc_id"]) for it in items]
        recs = run_arm(model, tok, fam, items, texts, lexicon, "base", "base", None, 0, _Null, args.batch)
        for layer in args.layers:
            j = layers.index(layer)
            hn, ha = (z["nat_last_prenorm"], z["alt_last_prenorm"]) if layer == 28 else (z["act_nat"][:, j], z["act_alt"][:, j])
            Dm = (hn - ha)[tr].astype(np.float64); _, _, Vt = np.linalg.svd(Dm, full_matrices=False); V = Vt[: max(RANKS)].astype(np.float32)
            T = ha[di].astype(np.float32)                                   # each held-out doc's own k = 4 alternative cue activation [n, D]
            C = T @ V.T                                                     # own coordinates [n, r_max]
            for r in RANKS:
                fn = (lambda bi, Vr=V[:r], Cr=C[:, :r]: CueSubspacePatch(model, layer, Vr, Cr[bi:bi + args.batch]))
                recs += run_arm(model, tok, fam, items, texts, lexicon, f"alt_L{layer}_own{r}", f"own{r}", "alt", layer, fn, args.batch)
            recs += run_arm(model, tok, fam, items, texts, lexicon, f"alt_L{layer}_full", "full", "alt", layer, lambda bi: CueReplace(model, layer, T[bi:bi + args.batch]), args.batch)
            print(f"{fam}: layer {layer} done", flush=True)
            json.dump(recs, open(str(out) + ".partial", "w"), ensure_ascii=False)
        json.dump(recs, open(out, "w"), ensure_ascii=False); Path(str(out) + ".partial").unlink()
        print(f"{fam}: {len(items)} held-out docs, {len(recs)} records", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
