#!/usr/bin/env python
"""Per-prompt read / write activations for the read→write linear-map test (GPU).

For every k = 4 prompt (200 texts × 2 poles per family) one forward pass collects
  read[L]  = mean residual activation over ALL evidence tokens of the 4 instances at layer L
             (L = 0 is the embedding output = hidden_states[0]; GPT-J adds no positional vector)
  write[L] = residual activation at the last position (the cue token) at layer L.
Layer convention as everywhere in this study: L = hidden_states[L] = output of block h[L-1].

Saves artifacts/style_translation/prompt_pairs/<family>.npz:
  doc_id[N] str, pole[N] str, style_ok[N] bool, judge_ok[N] bool (from the step-3 k = 4 rollouts),
  n_evidence[N], read_L{0,2,4,8,12,24}[N,4096] fp16, write_L{12,16,20,24}[N,4096] fp16.
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
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import batches_by_len

PROMPTS = ARTIFACTS_ROOT / "style_translation" / "prompts"
EVID = ARTIFACTS_ROOT / "style_translation" / "read_features" / "evidence"
ROLL = ARTIFACTS_ROOT / "style_translation" / "rollouts"
OUT = ARTIFACTS_ROOT / "style_translation" / "prompt_pairs"
READ_LAYERS = (0, 2, 4, 8, 12, 24)
WRITE_LAYERS = (12, 16, 20, 24)
K = 4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--token_budget", type=int, default=8000)
    ap.add_argument("--batch_cap", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load_model()
    D = model.config.n_embd
    for fam in args.families:
        if (OUT / f"{fam}.npz").exists():
            print(f"{fam}: exists, skip", flush=True); continue
        ev = {(r["doc_id"], r["pole"]): r for r in json.load(open(EVID / f"{fam}.json"))}
        ok = {(r["doc_id"], r["style"]): (bool(r.get("style_ok")), bool((r.get("judge") or {}).get("ok")))
              for r in json.load(open(ROLL / f"{fam}.json")) if r["k"] == K}
        items = []
        for p in json.load(open(PROMPTS / f"{fam}.json")):
            if p["k"] != K:
                continue
            r = ev[(p["doc_id"], p["style"])]
            assert r["prompt_len"] == len(p["prompt_ids"])
            pos = sorted({i for e in r["instances"] for i in e["idx"]})
            assert pos and max(pos) < len(p["prompt_ids"])
            items.append({"ids": p["prompt_ids"], "pole": p["style"], "doc": p["doc_id"], "pos": pos,
                          "ok": ok.get((p["doc_id"], p["style"]), (False, False))})
        N = len(items)
        read = {L: np.zeros((N, D), np.float16) for L in READ_LAYERS}
        write = {L: np.zeros((N, D), np.float16) for L in WRITE_LAYERS}
        for bidx in batches_by_len(items, args.token_budget, args.batch_cap):
            rows = [items[i] for i in bidx]
            T = max(len(r["ids"]) for r in rows)
            ids = torch.full((len(rows), T), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros((len(rows), T), dtype=torch.long)
            for j, r in enumerate(rows):                      # left padding, as in every capture of this study
                n = len(r["ids"]); ids[j, T - n:] = torch.tensor(r["ids"]); att[j, T - n:] = 1
            with torch.no_grad():
                out = model.transformer(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
            for j, (i, r) in enumerate(zip(bidx, rows)):
                off = T - len(r["ids"])
                pos = torch.tensor([off + q for q in r["pos"]], device="cuda")
                for L in READ_LAYERS:
                    read[L][i] = out.hidden_states[L][j, pos, :].float().mean(0).cpu().numpy().astype(np.float16)
                for L in WRITE_LAYERS:
                    write[L][i] = out.hidden_states[L][j, -1, :].float().cpu().numpy().astype(np.float16)
        np.savez_compressed(OUT / f"{fam}.npz",
                            doc_id=np.array([it["doc"] for it in items]), pole=np.array([it["pole"] for it in items]),
                            style_ok=np.array([it["ok"][0] for it in items]), judge_ok=np.array([it["ok"][1] for it in items]),
                            n_evidence=np.array([len(it["pos"]) for it in items]),
                            **{f"read_L{L}": read[L] for L in READ_LAYERS}, **{f"write_L{L}": write[L] for L in WRITE_LAYERS})
        # sanity: paired mean difference at the cue (L24) vs the stored steering vector (pooled over k, paired) — same sign expected
        sv = np.load(ARTIFACTS_ROOT / "style_translation" / "steering" / "vectors" / f"{fam}.npz")["v_nat"][23]
        pole = np.array([it["pole"] for it in items]); w = write[24].astype(np.float32)
        v = w[pole == "nat"].mean(0) - w[pole == "alt"].mean(0)
        print(f"{fam}: N={N} evidence tokens/prompt={np.mean([len(it['pos']) for it in items]):.1f} "
              f"cos(k4 cue diff, stored L24 steering vector)={float(v @ sv / np.linalg.norm(v) / np.linalg.norm(sv)):.2f}", flush=True)


if __name__ == "__main__":
    main()
