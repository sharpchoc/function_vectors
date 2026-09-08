#!/usr/bin/env python
"""Step 5b — sparse head selection for style steering (Hu et al. 2505.05145 §3.1 recipe, reusing
src/sandbox/sparse_head_selection/train_sparse_heads.py: train_c / batch_label_logprobs / evaluate_points).

One coefficient vector c in [0,1]^448 SHARED across the 17 families. For family f the injected
vector is v_f = sum_h c_h * C_alt[f, h] (C_alt = that family's per-head alt means lifted to residual
space, capture_heads.py). v_f is added to the residual stream at the output of block L (style
convention L = hidden_states[L] = transformer.h[L-1]) at the cue token (last prompt position) of
the k = 0 prompt (identical for both styles). Loss per point = mean over LABEL tokens of
-log p(token) (teacher forced) + lambda * ||c||_1, where the label = the text_alt twin's tokens
from cue+1 through the end of the alt style span (opps[opp_index].alt_span) — the golden
completion restricted to the style decision. Per-token MEAN (not sum) so that all_caps /
sentence_caps (whole-first-sentence spans) do not dominate the pooled objective.

Split per family (doc_id-sorted): 0..119 fit -> of which a stratified 20 = early-stop/validation
slice ("es"), 100 train; 120..199 test (never used here except to report NLL).

Modes:
  points  - build & verify the (prompt_ids, label_ids) points (CPU) -> points/<family>.json
  train   - for each --layers L and --lambdas lam: fit c (resumable, one .pt per cell)
  select  - lambda* per layer (largest lam within --tol nats of best es NLL), L* by es NLL,
            heads c>0.2 / c>0.8, overlap with the 37-head FV set -> selection.json, layer_lambda.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT, STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.families import FAMILIES
from src.sandbox.style_translation.cue_tokens import HEADER, TOKENIZER
import src.sandbox.sparse_head_selection.train_sparse_heads as TSH
from src.sandbox.isolation_upper_bound.run_task import select_heads_nonempty

FAMS = [f.name for f in FAMILIES]
PAIRS = STYLE_TRANSLATION_DATA / "pairs"
PROMPTS = ARTIFACTS_ROOT / "style_translation" / "prompts"
ROOT = ARTIFACTS_ROOT / "style_translation" / "sparse_heads"
HEADS, POINTS, TRAIN = ROOT / "heads", ROOT / "points", ROOT / "train"
FV37 = ARTIFACTS_ROOT / "sandbox" / "ext_steerability" / "prunedfail_seed43" / "pooled_sparse" / "selection.json"
N_FIT, N_ES = 120, 20
LAYERS = [6, 9, 12, 16, 20]
LAMBDAS = [0.001, 0.003, 0.01, 0.03, 0.1]
SEED = 43

# ---- per-token mean loss: wrap the sandbox's summed -log p(label) -------------------------------
_sum_logprobs = TSH.batch_label_logprobs


def _mean_logprobs(model, model_config, tokenizer, batch, v=None, inject_layer=9):
    nll, accs = _sum_logprobs(model, model_config, tokenizer, batch, v=v, inject_layer=inject_layer)
    n = torch.tensor([len(b["label_ids"]) for b in batch], device=nll.device, dtype=nll.dtype)
    return nll / n, accs


TSH.batch_label_logprobs = _mean_logprobs          # train_c / evaluate_points now use the per-token mean


# ---- points -----------------------------------------------------------------------------------
def build_points(fam, tok):
    recs = sorted(json.load(open(PAIRS / f"{fam}.json")), key=lambda r: r["doc_id"])
    k0 = {p["doc_id"]: p["prompt_ids"] for p in json.load(open(PROMPTS / f"{fam}.json")) if p["style"] == "alt" and p["k"] == 0}
    pts = []
    for idx, r in enumerate(recs):
        header = HEADER.format(es=r["text_es"]); h = len(header)
        c = r["cues"]["alt"][0]; opp = r["opps"][c["opp_index"]]
        enc = tok(header + r["text_alt"], return_offsets_mapping=True)
        ids, offs = enc.input_ids, enc.offset_mapping
        ci = c["cue_idx"]
        assert ids[: ci + 1] == k0[r["doc_id"]], (fam, r["doc_id"], "prompt mismatch")
        span_end = h + opp["alt_span"][1]
        j = ci + 1
        while j < len(ids) and offs[j][0] < span_end:
            j += 1
        label = ids[ci + 1: j]
        assert len(label) >= 1, (fam, r["doc_id"], "empty label")
        expected = (header + r["text_alt"])[offs[ci][1]: span_end]
        dec = tok.decode(label)
        assert dec.startswith(expected), (fam, r["doc_id"], repr(dec), repr(expected))
        pts.append({"task": fam, "doc_id": r["doc_id"], "idx": idx, "split": "fit" if idx < N_FIT else "test",
                    "prompt_ids": ids[: ci + 1], "label_ids": label, "cue_idx": ci, "label_text": dec,
                    "alt": opp["alt"], "nat": opp["nat"]})
    return pts


def load_points():
    pts = {fam: json.load(open(POINTS / f"{fam}.json")) for fam in FAMS}
    fit = [p for fam in FAMS for p in pts[fam] if p["split"] == "fit"]
    test = [p for fam in FAMS for p in pts[fam] if p["split"] == "test"]
    train, es = TSH.split_earlystop(fit, N_ES / N_FIT, SEED)      # stratified by family: 20 es / 100 train each
    return train, es, test


def tc_args(L, micro):
    return SimpleNamespace(init_c=0.5, lr=0.01, batch_size=64, micro_batch_size=micro, max_epochs=30, patience=3,
                           inject_layer=L - 1, threshold=0.2)


def per_family_nll(model, cfg, tok, points, C, task_index, c, args):
    out = {}
    for fam in task_index:                                   # families present in this run (smoke runs use a subset)
        sel = [p for p in points if p["task"] == fam]
        if not sel:
            continue
        nll, acc = TSH.evaluate_points(model, cfg, tok, sel, C, task_index, c, args)
        out[fam] = {"nll": nll, "acc": acc, "n": len(sel)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["points", "train", "select"])
    ap.add_argument("--layers", type=int, nargs="*", default=LAYERS)
    ap.add_argument("--lambdas", type=float, nargs="*", default=LAMBDAS)
    ap.add_argument("--micro", type=int, default=0, help="micro-batch; 0 = per-layer auto (saved activations scale with 28-L)")
    ap.add_argument("--families", nargs="*", default=FAMS, help="train: restrict pooled families (smoke)")
    ap.add_argument("--max_epochs", type=int, default=15)
    ap.add_argument("--tol", type=float, default=0.02)
    args = ap.parse_args()

    if args.mode == "points":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        POINTS.mkdir(parents=True, exist_ok=True)
        print(f"{'family':14s} {'n':>4s} {'label toks: med':>15s} {'max':>4s}  examples (prompt tail -> label)")
        for fam in FAMS:
            pts = build_points(fam, tok)
            json.dump(pts, open(POINTS / f"{fam}.json", "w"))
            ls = [len(p["label_ids"]) for p in pts]
            ex = " | ".join(f"...{tok.decode(p['prompt_ids'][-3:])!r} -> {p['label_text']!r}" for p in pts[:3])
            print(f"{fam:14s} {len(pts):4d} {int(np.median(ls)):15d} {max(ls):4d}  {ex}", flush=True)
        return

    if args.mode == "select":
        rows, per_layer = [], {}
        for L in args.layers:
            base = json.load(open(TRAIN / f"L{L}" / "baseline_c0.json"))
            cells = {}
            for lam in args.lambdas:
                f = TRAIN / f"L{L}" / f"lambda{lam:g}.pt"
                if not f.exists():
                    continue
                d = torch.load(f, map_location="cpu", weights_only=False)
                c = d["c"]
                cells[lam] = d
                rows.append(dict(layer=L, lam=lam, es_nll=round(d["es_nll"], 4), test_nll=round(d["test_nll"], 4), es_nll_c0=round(base["es_nll"], 4),
                                 test_nll_c0=round(base["test_nll"], 4), n_heads_02=int((c > 0.2).sum()), n_heads_08=int((c > 0.8).sum()),
                                 l1=round(float(c.sum()), 3), best_epoch=d["best_epoch"]))
            if not cells:
                continue
            best = min(v["es_nll"] for v in cells.values())
            lam_star = max(lam for lam, v in cells.items() if v["es_nll"] <= best + args.tol)
            per_layer[L] = (lam_star, cells[lam_star]["es_nll"])
        L_star = min(per_layer, key=lambda L: per_layer[L][1])
        lam_star = per_layer[L_star][0]
        d = torch.load(TRAIN / f"L{L_star}" / f"lambda{lam_star:g}.pt", map_location="cpu", weights_only=False)
        c = d["c"]
        h02, fb02 = select_heads_nonempty(c, 0.2); h08, fb08 = select_heads_nonempty(c, 0.8)
        fv37 = {l * 16 + h for l, h, _ in json.load(open(FV37))["selected_heads"]} if FV37.exists() else set()
        sel = {"layer": L_star, "lambda": lam_star, "tol": args.tol, "per_layer_lambda_star": {str(L): v[0] for L, v in per_layer.items()},
               "per_layer_es_nll": {str(L): v[1] for L, v in per_layer.items()}, "c": c.tolist(), "heads_02": h02, "heads_08": h08,
               "fallback_02": fb02, "fallback_08": fb08, "heads_02_lh": [[i // 16, i % 16, round(float(c[i]), 4)] for i in h02],
               "fv37_overlap_02": sorted(set(h02) & fv37), "fv37_n": len(fv37),
               "es_nll": d["es_nll"], "test_nll": d["test_nll"], "per_family_es": d["per_family_es"], "per_family_test": d["per_family_test"]}
        json.dump(sel, open(ROOT / "selection.json", "w"), indent=1)
        with open(ROOT / "layer_lambda.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"selected L={L_star} lambda={lam_star} | heads c>0.2: {len(h02)} (c>0.8: {len(h08)}) | overlap with FV-37: {len(sel['fv37_overlap_02'])}"
              f" | es nll {d['es_nll']:.3f} test nll {d['test_nll']:.3f}", flush=True)
        for r in rows:
            print(r)
        return

    # ---- train ----------------------------------------------------------------------------------
    from src.sandbox.style_translation.rollout import load_model
    from src.sandbox.style_translation.capture_heads import gptj_config
    model, tok = load_model()
    cfg = gptj_config(model)
    model.to(torch.bfloat16)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    fams = args.families
    C = torch.stack([torch.from_numpy(np.load(HEADS / f"{fam}.npz")["C_alt"]) for fam in fams]).to(model.device)   # (F, 448, 4096) fp32
    task_index = {fam: i for i, fam in enumerate(fams)}
    train, es, test = load_points()
    train = [p for p in train if p["task"] in fams]; es = [p for p in es if p["task"] in fams]; test = [p for p in test if p["task"] in fams]
    print(f"points: train {len(train)} es {len(es)} test {len(test)} | families {len(fams)} | C {tuple(C.shape)}", flush=True)
    for L in args.layers:
        out = TRAIN / f"L{L}"; out.mkdir(parents=True, exist_ok=True)
        micro = args.micro or {6: 8, 9: 8, 12: 12, 16: 12, 20: 16}.get(L, 4)     # measured: micro 16 OOMs at L7 (21 blocks of activations), fits at L20
        a = tc_args(L, micro); a.max_epochs = args.max_epochs
        print(f"L{L}: micro-batch {micro}", flush=True)
        bfile = out / "baseline_c0.json"
        if not bfile.exists():
            torch.set_grad_enabled(False)
            es0, _ = TSH.evaluate_points(model, cfg, tok, es, C, task_index, None, a)
            te0, _ = TSH.evaluate_points(model, cfg, tok, test, C, task_index, None, a)
            json.dump({"es_nll": es0, "test_nll": te0, "per_family_es": per_family_nll(model, cfg, tok, es, C, task_index, None, a),
                       "per_family_test": per_family_nll(model, cfg, tok, test, C, task_index, None, a)}, open(bfile, "w"), indent=1)
            print(f"L{L}: c=0 baseline es nll {es0:.4f} test nll {te0:.4f}", flush=True)
        for li, lam in enumerate(args.lambdas):
            f = out / f"lambda{lam:g}.pt"
            if f.exists():
                print(f"L{L} lambda={lam:g}: exists, skip", flush=True); continue
            torch.set_grad_enabled(True)
            c, hist, best_epoch = TSH.train_c(model, cfg, tok, train, es, C, task_index, lam, a, run_seed=SEED + 1000 * L + li, desc=f"L{L} lam={lam:g}")
            torch.set_grad_enabled(False)
            es_nll, es_acc = TSH.evaluate_points(model, cfg, tok, es, C, task_index, c, a)
            te_nll, te_acc = TSH.evaluate_points(model, cfg, tok, test, C, task_index, c, a)
            torch.save({"c": c.cpu(), "history": hist, "best_epoch": best_epoch, "es_nll": es_nll, "es_acc": es_acc, "test_nll": te_nll, "test_acc": te_acc,
                        "per_family_es": per_family_nll(model, cfg, tok, es, C, task_index, c, a),
                        "per_family_test": per_family_nll(model, cfg, tok, test, C, task_index, c, a),
                        "layer": L, "lambda": lam, "families": fams}, f)
            print(f"L{L} lambda={lam:g}: es nll {es_nll:.4f} test nll {te_nll:.4f} | heads c>0.2 {(c > 0.2).sum().item()} c>0.8 {(c > 0.8).sum().item()} | best epoch {best_epoch}", flush=True)
    print("train done", flush=True)


if __name__ == "__main__":
    main()
