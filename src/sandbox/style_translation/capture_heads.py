#!/usr/bin/env python
"""Step 5a — per-attention-head mean outputs at the cue token, alt style (GPU).

Selection = step 4's (rollout records with `style_ok and judge.ok`, all k pooled) restricted to the
ALT style and to the FIT texts (doc_id-sorted index < 120), so the 80 test texts never enter the
head vectors. For each prompt a forward pass with baukit.TraceDict on every block's
attn.out_proj (retain_input) gives the per-head inputs a_{l,h} (256-d) at the cue = last position
(left padding). Means (28, 16, 256) are lifted to residual space per head:
C_alt[l*16+h] = W_O^{l,h} @ mean a_{l,h}  (448, 4096)   (`build_contributions_single`; no bias).
Linearity gate on the first batch of every family: sum_h W_O^h a_h == attention output.
Also stores the mean attention output per layer at the cue (28, 4096) for the indicator check.

Saves artifacts/style_translation/sparse_heads/heads/<family>.npz:
  head_means_alt (28,16,256) fp32, C_alt (448,4096) fp32, attn_out_mean (28,4096), n_alt, linearity_dev
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from baukit import TraceDict

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT
from src.utils.model_utils import get_attn_out_proj
from src.utils.varicl_utils import split_activations_by_head
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.rollout import load_model
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len
from src.sandbox.isolation_upper_bound.run_task import build_contributions_single

ROLL = ARTIFACTS_ROOT / "style_translation" / "rollouts"
PROMPTS = ARTIFACTS_ROOT / "style_translation" / "prompts"
OUT = ARTIFACTS_ROOT / "style_translation" / "sparse_heads" / "heads"
N_FIT = 120


def gptj_config(model):
    return {"n_layers": model.config.n_layer, "n_heads": model.config.n_head, "resid_dim": model.config.n_embd,
            "attn_hook_names": [f"transformer.h.{l}.attn.out_proj" for l in range(model.config.n_layer)]}


def fit_doc_ids(fam):
    """doc_id-sorted texts 0..119 (same order as steer_screen.k0_items)."""
    ids = sorted({p["doc_id"] for p in json.load(open(PROMPTS / f"{fam}.json"))})
    assert len(ids) == 200, (fam, len(ids))
    return set(ids[:N_FIT])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--token_budget", type=int, default=8000)
    ap.add_argument("--batch_cap", type=int, default=16)
    ap.add_argument("--all_texts", action="store_true", help="pool all 200 texts (default: fit texts only)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load_model()
    cfg = gptj_config(model)
    nl, nh, D = cfg["n_layers"], cfg["n_heads"], cfg["resid_dim"]
    hd = D // nh
    for fam in args.families:
        if (OUT / f"{fam}.npz").exists():
            print(f"{fam}: exists, skip", flush=True); continue
        keep = None if args.all_texts else fit_doc_ids(fam)
        recs = json.load(open(ROLL / f"{fam}.json"))
        prompts = {(p["doc_id"], p["style"], p["k"]): p["prompt_ids"] for p in json.load(open(PROMPTS / f"{fam}.json"))}
        items = [{"ids": prompts[(r["doc_id"], r["style"], r["k"])]} for r in recs
                 if r["style"] == "alt" and r["style_ok"] and r.get("judge") and r["judge"]["ok"]
                 and (keep is None or r["doc_id"] in keep)]
        assert len(items) >= 40, f"{fam}: only {len(items)} correct alt prompts"
        head_sum = torch.zeros(nl, nh, hd, dtype=torch.float64)
        attn_sum = torch.zeros(nl, D, dtype=torch.float64)
        n_seen, dev_max = 0, 0.0
        for bi, b in enumerate(batches_by_len(items, args.token_budget, args.batch_cap)):
            lens = [len(items[i]["ids"]) for i in b]; L = max(lens)
            ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros(len(b), L, dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, L - lens[r]:] = torch.tensor(items[i]["ids"]); att[r, L - lens[r]:] = 1
            with torch.no_grad(), TraceDict(model, layers=cfg["attn_hook_names"], retain_input=True, retain_output=True) as td:
                model.transformer(input_ids=ids.cuda(), attention_mask=att.cuda())
            for li, name in enumerate(cfg["attn_hook_names"]):
                inp = td[name].input; inp = inp[0] if isinstance(inp, tuple) else inp
                cue = split_activations_by_head(inp, cfg)[:, -1]                       # (B, H, hd), left padding -> cue = last
                head_sum[li] += cue.double().sum(0).cpu()
                outp = td[name].output; outp = outp[0] if isinstance(outp, tuple) else outp
                ref = outp[:, -1]
                attn_sum[li] += ref.double().sum(0).cpu()
                if bi == 0:                                                              # linearity gate
                    w = get_attn_out_proj(model, li).weight.detach()
                    rebuilt = torch.einsum("bhd,ehd->be", cue.to(w.dtype), w.view(D, nh, hd))
                    dev = (rebuilt - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
                    assert dev < 5e-2, f"{fam}: linearity gate failed at layer {li}: rel dev {dev:.3e}"
                    dev_max = max(dev_max, dev)
            n_seen += len(b)
        head_means = (head_sum / n_seen).float()
        C = build_contributions_single(head_means, model, cfg).cpu()
        attn_mean = (attn_sum / n_seen).float()
        # indicator check: sum of a layer's 16 head contributions == that layer's mean attention output
        rel = max((C[l * nh:(l + 1) * nh].sum(0) - attn_mean[l]).norm().item() / attn_mean[l].norm().item() for l in range(nl))
        assert rel < 1e-2, f"{fam}: indicator check failed rel {rel:.2e}"
        np.savez_compressed(OUT / f"{fam}.npz", head_means_alt=head_means.numpy(), C_alt=C.numpy(), attn_out_mean=attn_mean.numpy(),
                            n_alt=n_seen, linearity_dev=dev_max)
        norms = C.norm(dim=1).view(nl, nh)
        print(f"{fam}: n_alt={n_seen} linearity dev {dev_max:.2e} indicator rel {rel:.1e} | "
              f"|C| per head median {norms.median():.2f} max {norms.max():.2f} (L{int(norms.argmax()) // nh + 1})", flush=True)
    print("capture done", flush=True)


if __name__ == "__main__":
    main()
