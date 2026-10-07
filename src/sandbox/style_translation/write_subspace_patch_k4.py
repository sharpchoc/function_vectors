#!/usr/bin/env python
"""Twin patch at the k = 4 cue (GPU): natural-context k = 4 prompts of held-out documents are patched at the cue token with the coordinates of the
SAME document's k = 4 ALTERNATIVE-context twin (exact twin: same demonstrations and cue token, only the convention differs), and we count how many
completions flip to the alternative convention. Documents are kept only when the alternative twin's own k = 4 completion was correct (convention
followed AND judge OK). Subspace per layer 20..28 = top-r uncentered right singular vectors of the training documents' k = 4 pair differences.
Arms per layer: own1 / own2 / own10 (twin coordinates, rank 1 / 2 / 10), full (whole cue activation = the twin's), mean1 / mean2 (training-mean
alternative coordinates, for comparison); base = unpatched natural prompt. Success = alternative convention AND judge OK.
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
from src.sandbox.style_translation.write_split import heldout
from src.sandbox.style_translation.write_subspace_patch import run_arm, _Null

RANKS = (1, 2, 10)


def twin_items(MP, fam, k, docs_with_pairs):
    """k-shot NATURAL-context prompts of held-out documents whose ALTERNATIVE twin was completed correctly at k."""
    recs = json.load(open(MP["rollouts"] / f"{fam}.json"))
    ok_alt = {r["doc_id"] for r in recs if r["k"] == k and r["style"] == "alt" and r["style_ok"] and (r.get("judge") or {}).get("ok")}
    P = {(p["doc_id"], p["style"]): p for p in json.load(open(MP["prompts"] / f"{fam}.json")) if p["k"] == k}
    items = []
    for d in sorted({d for d, _ in P}):
        if not heldout(d) or d not in ok_alt or d not in docs_with_pairs:
            continue
        it = dict(P[(d, "nat")]); it["ref_nat"], it["ref_alt"] = P[(d, "nat")]["ref_sentence"], P[(d, "alt")]["ref_sentence"]; items.append(it)
    return items


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
        items = candidates(fam, tok, twin_items(MP, fam, args.k, set(docs)))[: args.limit]
        if not items:
            print(f"{fam}: no qualifying documents", flush=True); json.dump([], open(out, "w")); continue
        texts = {it["doc_id"]: tok.decode(it["prompt_ids"]) for it in items}; di = [docs.index(it["doc_id"]) for it in items]
        recs = run_arm(model, tok, fam, items, texts, lexicon, "base", "base", None, 0, _Null, args.batch)
        for layer in args.layers:
            j = layers.index(layer)
            hn, ha = (z["nat_last_prenorm"], z["alt_last_prenorm"]) if layer == 28 else (z["act_nat"][:, j], z["act_alt"][:, j])
            Dm = (hn - ha)[tr].astype(np.float64); _, _, Vt = np.linalg.svd(Dm, full_matrices=False); V = Vt[: max(RANKS)].astype(np.float32)
            if V[0] @ Dm.mean(0) < 0:
                V[0] *= -1
            T = ha[di].astype(np.float32); C = T @ V.T                        # the twin's cue activation and its coordinates
            c_mean = (ha[tr].astype(np.float32) @ V.T).mean(0)               # training-mean alternative coordinates
            for r in RANKS:
                fn = (lambda bi, Vr=V[:r], Cr=C[:, :r]: CueSubspacePatch(model, layer, Vr, Cr[bi:bi + args.batch]))
                recs += run_arm(model, tok, fam, items, texts, lexicon, f"alt_L{layer}_own{r}", f"own{r}", "alt", layer, fn, args.batch)
            for r in (1, 2):
                fn = (lambda bi, Vr=V[:r], cr=c_mean[:r]: CueSubspacePatch(model, layer, Vr, cr))   # bi unused: shared target for every row
                recs += run_arm(model, tok, fam, items, texts, lexicon, f"alt_L{layer}_mean{r}", f"mean{r}", "alt", layer, fn, args.batch)
            recs += run_arm(model, tok, fam, items, texts, lexicon, f"alt_L{layer}_full", "full", "alt", layer, lambda bi: CueReplace(model, layer, T[bi:bi + args.batch]), args.batch)
            print(f"{fam}: layer {layer} done", flush=True)
            json.dump(recs, open(str(out) + ".partial", "w"), ensure_ascii=False)
        json.dump(recs, open(out, "w"), ensure_ascii=False); Path(str(out) + ".partial").unlink()
        print(f"{fam}: {len(items)} qualifying held-out docs, {len(recs)} records", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
