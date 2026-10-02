#!/usr/bin/env python
"""Subspace-patch steering at the cue token of 0-shot prompts (GPU), one layer at a time, code-convention families.

Subspace (per layer L, from cue_pairs_layers/<family>_k4.npz, TRAINING documents only): the pair differences d_i = h_nat,i - h_alt,i of the
k = 4 twin prompts are stacked [n_train, D]; V_L = its top-r right singular vectors, UNCENTERED (so PC1 ~ the mean difference). Target
coordinates: c_pole = mean over the training documents' k = 4 `pole` prompts of h V_L^T. Layer 28 uses the un-normed output of the last block.
Arms, per layer and target pole, on the held-out documents' 0-shot prompts (first --limit), graded like write_sweep.py:
  patch2 : replace the cue token's coordinates in span(PC1, PC2) by c_target (CueSubspacePatch)
  patch1 : the same with PC1 only
  add2   : reference, add 2 * (mean difference of the same training pairs) towards the target (CueSteer)
  base   : unsteered.
Records -> <out_dir>/<family>.json (judge afterwards: judge_rollouts.py --dir <out_dir>); subspace -> cue_pairs_layers/<family>_k4_subspace.npz.
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
from src.sandbox.style_translation.models import paths as model_paths
from src.sandbox.style_translation.rollout import load_model, MAX_NEW
from src.sandbox.style_translation.scoring import cut_code
from src.sandbox.style_translation.code_scoring import decide_any
from src.sandbox.style_translation.steer_hooks import CueSteer, CueSubspacePatch, unit_test, unit_test_subspace_patch
from src.sandbox.style_translation.logprob_margin import candidates
from src.sandbox.style_translation.write_sweep import held_out_items, pad, KEEP


def build_subspace(src, rank=2):
    z = np.load(src); tr = ~z["heldout"]; layers = [int(l) for l in z["layers"]]; out = {}
    for j, l in enumerate(layers):
        hn, ha = (z["nat_last_prenorm"], z["alt_last_prenorm"]) if l == 28 else (z["act_nat"][:, j], z["act_alt"][:, j])
        hn = hn[tr].astype(np.float64); ha = ha[tr].astype(np.float64); Dm = hn - ha
        _, s, Vt = np.linalg.svd(Dm, full_matrices=False); V = Vt[:rank]; md = Dm.mean(0)
        if V[0] @ md < 0:
            V[0] *= -1
        out[l] = dict(V=V.astype(np.float32), c_nat=(hn @ V.T).mean(0).astype(np.float32), c_alt=(ha @ V.T).mean(0).astype(np.float32), mean_diff=md.astype(np.float32),
                      sv=s[:10].astype(np.float32), energy=(s[:rank] ** 2 / (s ** 2).sum()).astype(np.float32), cos_pc1_meandiff=float(V[0] @ md / np.linalg.norm(md)))
    return out, int(tr.sum())


def run_arm(model, tok, fam, items, texts, lexicon, arm, variant, target, layer, hook_fn, batch):
    recs = []
    for bi in range(0, len(items), batch):
        b = items[bi:bi + batch]; ids, att, L = pad(b, tok)
        torch.manual_seed(zlib.crc32(f"{fam}|{layer}|{variant}|{target}|{bi}".encode()))
        with torch.no_grad(), hook_fn() as hk:
            lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, -1, :].float(), -1); top1 = lp.argmax(-1)
            before = None if getattr(hk, "before", None) is None else hk.before.numpy().copy()
            gen = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id)
        for r, it in enumerate(b):
            raw = tok.decode(gen[r, L:], skip_special_tokens=True); cut, capped = cut_code(raw, fam)
            d = decide_any(fam, texts[it["doc_id"]], it["seg_prefix"], cut, it["next_nat"], it["next_alt"], lexicon)
            fn, fa = it["first_ctx"], it["first_other"]
            t, o = (fn, fa) if target != "alt" else (fa, fn)
            recs.append({k: it[k] for k in KEEP} | {"style": target or "none", "arm": arm, "variant": variant, "target": target, "layer": layer,
                         "ref_sentence": it["ref_alt"] if target == "alt" else it["ref_nat"], "tail_raw": raw, "tail": cut, "capped": capped,
                         "decision": d, "style_ok": (d == target) if target else None,
                         "lp_nat": float(lp[r, fn]), "lp_alt": float(lp[r, fa]), "margin_target": float(lp[r, t] - lp[r, o]),
                         "top1": "nat" if top1[r].item() == fn else ("alt" if top1[r].item() == fa else None),
                         "coords_before": None if before is None else [float(x) for x in before[r]], "judge": None})
    return recs


class _Null:
    def __enter__(self): return self
    def __exit__(self, *exc): pass


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
    print("hook unit tests passed", flush=True)
    for fam in args.families:
        out = args.out_dir / f"{fam}.json"
        if out.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        S, n_train = build_subspace(SRC / f"{fam}_k{args.k}.npz")
        np.savez_compressed(SRC / f"{fam}_k{args.k}_subspace.npz", layers=np.array(sorted(S)), n_train=n_train,
                            **{f"{key}_L{l}": S[l][key] for l in S for key in ("V", "c_nat", "c_alt", "mean_diff", "sv", "energy", "cos_pc1_meandiff")})
        items = candidates(fam, tok, held_out_items(MP["prompts"], fam))[: args.limit]
        texts = {it["doc_id"]: tok.decode(it["prompt_ids"]) for it in items}
        recs = run_arm(model, tok, fam, items, texts, lexicon, "base", "base", None, 0, _Null, args.batch)
        for layer in args.layers:
            s = S[layer]
            for target in ("nat", "alt"):
                c = s[f"c_{target}"]; sign = 1.0 if target == "nat" else -1.0
                arms = {"patch2": lambda: CueSubspacePatch(model, layer, s["V"], c), "patch1": lambda: CueSubspacePatch(model, layer, s["V"][:1], c[:1]),
                        "add2": lambda: CueSteer(model, layer, sign * s["mean_diff"], 2.0)}
                for variant, fn in arms.items():
                    recs += run_arm(model, tok, fam, items, texts, lexicon, f"{target}_L{layer}_{variant}", variant, target, layer, fn, args.batch)
            print(f"{fam}: layer {layer} done (c_nat {np.round(s['c_nat'], 1)}, c_alt {np.round(s['c_alt'], 1)})", flush=True)
            json.dump(recs, open(str(out) + ".partial", "w"), ensure_ascii=False)
        json.dump(recs, open(out, "w"), ensure_ascii=False); Path(str(out) + ".partial").unlink()
        print(f"{fam}: {len(items)} held-out docs, {len(recs)} records", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
