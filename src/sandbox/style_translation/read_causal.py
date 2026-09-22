#!/usr/bin/env python
"""Are the read features causal for the write features? (GPU; code-convention families; 2026-09-22, user request — the analogue of the
function-vector read->write experiment.)

For every held-out k = 3 prompt whose demonstrations are in the NATURAL convention (first --limit documents, as read_sweep.py), the read
feature is injected at layer L_READ at EVERY evidence token (PositionSteer, prefill only): h += -alpha * r, r = mean_nat - mean_alt of the
read feature (capture_read.py, training documents), i.e. towards the ALTERNATIVE convention, for several alpha. Recorded at the query cue
token (last prompt position), at layers --from_layer..28 (default 20): the cue residual h_alpha; the unsteered residual h_0 of the same prompt; and the
COUNTERFACTUAL residual h_cf = the same document's k = 3 prompt with ALTERNATIVE-convention demonstrations (identical query, same cue).
Metrics per prompt, layer and alpha (the FV metric, user decision 2026-07-14 = dircos):
    dircos     = cos(h_alpha - h_0, h_cf - h_0)                 direction alignment of the steering displacement with the real context effect
    proj_frac  = ((h_alpha - h_0) . w) / ((h_cf - h_0) . w)     fraction of the counterfactual shift recovered along the WRITE feature w
    cos_w      = cos(h_alpha - h_0, w)                          w = unit(mean_alt - mean_nat) of the cue-token write feature at that layer
    norm_ratio = |h_alpha - h_0| / |h_cf - h_0|
    cf_cos_w   = cos(h_cf - h_0, w)                             how well the real context effect itself aligns with w (reference)
Controls (--controls only; off by default, user decision 2026-09-22) with the same alphas: (a) the read feature of ANOTHER pool family (seeded choice, same language when possible; as-is, norms
recorded); (b) a random Gaussian direction with the norm of r. Next-token log-prob margin towards the alternative first token is recorded
for every arm (candidates() as in read_sweep). Output <out_dir>/<family>.npz (metrics [n_prompts, n_arms, 28] + L24 cue vectors fp16).
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
from src.sandbox.style_translation.models import paths as model_paths, arch
from src.sandbox.style_translation.rollout import load_model
from src.sandbox.style_translation.steer_hooks import PositionSteer, unit_test_positions
from src.sandbox.style_translation.logprob_margin import candidates
from src.sandbox.style_translation.read_sweep import held_out_k3
from src.sandbox.style_translation.write_sweep import pad
from src.sandbox.style_translation.code_families import CODE_FAMILY

K = 3


def cue_states(model, trunk, tok, items, layer, vec, alpha, positions_of, batch, NL, L0=1):
    """[n, NL, D] float32 cue residuals (hidden_states[1..NL] at the last position) and [n, 2] log-probs of (first_ctx, first_other)."""
    H = np.zeros((len(items), NL - L0 + 1, arch(model)["hidden"]), np.float32); LP = np.zeros((len(items), 2), np.float32)
    for bi in range(0, len(items), batch):
        b = items[bi:bi + batch]; ids, att, L = pad(b, tok)
        positions = [[L - len(it["prompt_ids"]) + j for j in positions_of(it)] for it in b] if alpha else [[] for _ in b]
        with torch.no_grad(), PositionSteer(model, layer, vec, alpha, positions):
            out = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
            hs = torch.stack([out.hidden_states[l][:, -1, :] for l in range(L0, NL + 1)], 1).float()
            lp = torch.log_softmax(out.logits[:, -1, :].float(), -1)
        for r, it in enumerate(b):
            H[bi + r] = hs[r].cpu().numpy(); LP[bi + r] = [float(lp[r, it["first_ctx"]]), float(lp[r, it["first_other"]])]
    return H, LP


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25_code"); ap.add_argument("--families", nargs="*", required=True)
    ap.add_argument("--read_layer", type=int, default=8); ap.add_argument("--alphas", nargs="*", type=float, default=[0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--vec_tag", default="k3_train"); ap.add_argument("--out_dir", type=Path, required=True); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=40); ap.add_argument("--controls", action="store_true", help="also run the other-family and random-direction controls"); ap.add_argument("--from_layer", type=int, default=20, help="record cue residuals for layers from_layer..28 only"); ap.add_argument("--pool", default="results/code_styles/code_pool.json")
    args = ap.parse_args()
    MP = model_paths(args.model); RV = MP["read_features"] / f"vectors_{args.vec_tag}"; WV = MP["steering"] / f"vectors_{args.vec_tag}"
    args.out_dir.mkdir(parents=True, exist_ok=True); pool = json.load(open(_BOOT / args.pool))["pool"]
    model, tok = load_model(model=args.model); A = arch(model); NL, D, trunk = A["n_layers"], A["hidden"], None
    assert unit_test_positions(model, tok, layer=6), "position steering hook unit test failed"
    Lr = args.read_layer
    for fam in args.families:
        out = args.out_dir / f"{fam}.npz"
        if out.exists():
            print(f"{fam}: exists, skip", flush=True); continue
        items = candidates(fam, tok, held_out_k3(MP, fam))
        nat = [it for it in items if it["style"] == "nat"][: args.limit]
        alt_by_doc = {it["doc_id"]: it for it in items if it["style"] == "alt"}
        nat = [it for it in nat if it["doc_id"] in alt_by_doc]; cf = [alt_by_doc[it["doc_id"]] for it in nat]
        r = np.load(RV / f"{fam}.npz")["diff"][Lr - 1].astype(np.float32)                    # mean_nat - mean_alt at the read layer
        w_all = -np.load(WV / f"{fam}.npz")["v_nat"].astype(np.float32)                        # [NL, D]: mean_alt - mean_nat (write feature) per layer
        w_all = w_all[args.from_layer - 1:]; w_all /= np.linalg.norm(w_all, axis=1, keepdims=True) + 1e-8; i24 = 24 - args.from_layer
        lang = CODE_FAMILY[fam].tgt_lang; others = [f for f in pool if f != fam and CODE_FAMILY[f].tgt_lang == lang] or [f for f in pool if f != fam]
        rng = np.random.default_rng(zlib.crc32(f"read_causal|{fam}".encode())); ctrl_fam = others[rng.integers(len(others))]
        r_ctrl = np.load(RV / f"{ctrl_fam}.npz")["diff"][Lr - 1].astype(np.float32)
        r_rand = rng.standard_normal(D).astype(np.float32); r_rand *= np.linalg.norm(r) / np.linalg.norm(r_rand)
        ev = lambda it: it["evidence"]
        H0, LP0 = cue_states(model, trunk, tok, nat, Lr, np.zeros(D, np.float32), 0.0, ev, args.batch, NL, args.from_layer)
        Hcf, LPcf = cue_states(model, trunk, tok, cf, Lr, np.zeros(D, np.float32), 0.0, ev, args.batch, NL, args.from_layer)
        arms = [("read", -r)] + ([("ctrl_family", -r_ctrl), ("ctrl_random", -r_rand)] if args.controls else [])
        names = []; M = {k: [] for k in ("dircos", "proj_frac", "cos_w", "norm_ratio")}; L24 = []; LPs = []
        dcf = Hcf - H0                                                                          # [n, NL, D]
        for aname, vec in arms:
            for alpha in args.alphas:
                Ha, LPa = cue_states(model, trunk, tok, nat, Lr, vec, alpha, ev, args.batch, NL, args.from_layer); ds = Ha - H0
                num = (ds * dcf).sum(-1); den = np.linalg.norm(ds, axis=-1) * np.linalg.norm(dcf, axis=-1) + 1e-8
                pw = (ds * w_all[None]).sum(-1); pcf = (dcf * w_all[None]).sum(-1)
                M["dircos"].append(num / den); M["proj_frac"].append(pw / (pcf + 1e-8 * np.sign(pcf + 1e-12))); M["cos_w"].append(pw / (np.linalg.norm(ds, axis=-1) + 1e-8))
                M["norm_ratio"].append(np.linalg.norm(ds, axis=-1) / (np.linalg.norm(dcf, axis=-1) + 1e-8))
                names.append(f"{aname}_a{alpha:g}"); L24.append(Ha[:, i24, :].astype(np.float16)); LPs.append(LPa)
                print(f"{fam}: {aname} alpha {alpha:g} | L24 dircos {np.mean(M['dircos'][-1][:, i24]):.3f} proj_frac {np.mean(M['proj_frac'][-1][:, i24]):.3f} "
                      f"cos_w {np.mean(M['cos_w'][-1][:, i24]):.3f} norm_ratio {np.mean(M['norm_ratio'][-1][:, i24]):.2f}", flush=True)
        cf_cos_w = (dcf * w_all[None]).sum(-1) / (np.linalg.norm(dcf, axis=-1) + 1e-8)
        np.savez_compressed(out, arms=np.array(names), layers=np.arange(args.from_layer, NL + 1), doc_id=np.array([it["doc_id"] for it in nat]), read_layer=Lr, alphas=np.array(args.alphas), ctrl_family=ctrl_fam,
                            norm_r=float(np.linalg.norm(r)), norm_r_ctrl=float(np.linalg.norm(r_ctrl)), n_evidence=np.array([len(it["evidence"]) for it in nat]),
                            **{k: np.stack(v, 1).astype(np.float32) for k, v in M.items()}, cf_cos_w=cf_cos_w.astype(np.float32),
                            cf_norm=np.linalg.norm(dcf, axis=-1).astype(np.float32), h0_norm=np.linalg.norm(H0, axis=-1).astype(np.float32),
                            h0_L24=H0[:, i24, :].astype(np.float16), hcf_L24=Hcf[:, i24, :].astype(np.float16), h_L24=np.stack(L24, 1),
                            lp0=LP0, lpcf=LPcf, lp=np.stack(LPs, 1), first_is_ctx_nat=True)
        print(f"{fam}: {len(nat)} prompts, control family {ctrl_fam}, |r| {np.linalg.norm(r):.1f} |r_ctrl| {np.linalg.norm(r_ctrl):.1f} | cf cos_w L24 {np.mean(cf_cos_w[:, i24]):.3f}", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
