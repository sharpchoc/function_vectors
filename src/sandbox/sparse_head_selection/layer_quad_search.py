#!/usr/bin/env python
"""SANDBOX: k=4-layer mean-ablation search (§3.2 stage-1 extension).

Same construction as layer_mean_ablation.py (73-head H_sig, UNWEIGHTED Eq.-2 FVs, uniform grand
mean, injection @L9, 1720 zero-shot datapoints). Finds the best 4-layer kept-task-specific set:
  1. GREEDY forward selection: extend the cached best pair to the best triple, then best quad
     (evaluates ~2 * n_layers conditions).
  2. EXHAUSTIVE quads over a 12-layer pool = layers ranked by max(single acc, best pair acc
     containing the layer) from the cached stage-1 results (C(12,4) = 495 conditions).
All conditions cache into the same layer_ablation_results.json (keys set_L{a}_L{b}_...), so
reruns and later k values reuse everything.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.sandbox.sparse_head_selection.layer_mean_ablation import evaluate_fixed_v, parse_args
from src.sandbox.sparse_head_selection.train_sparse_heads import (
    build_contributions,
    build_task_datapoints,
    load_train_tasks,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed


def canonical_key(S):
    return "set_" + "_".join(f"L{l}" for l in sorted(S))


def main():
    args = parse_args()
    set_seed(args.seed)
    out_path = args.output_root / "layer_ablation_results.json"
    results = json.load(open(out_path))

    tasks = args.tasks or load_train_tasks(args)
    c = torch.load(args.coeffs_path, map_location="cpu", weights_only=False)["c"].float()
    n_heads = 16
    sig = [(i // n_heads, i % n_heads) for i in range(c.numel()) if c[i].item() > args.sig_threshold]
    sig_layers = sorted({l for l, _ in sig})

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    C = build_contributions(tasks, args, model, model_config)
    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sig_flat = torch.tensor([l * n_heads + h for l, h in sig], device=C.device)
    Csig = C[:, sig_flat, :]
    Cbar = Csig.mean(dim=0)
    layer_of = torch.tensor([l for l, _ in sig], device=C.device)
    task_index = {t: i for i, t in enumerate(tasks)}
    points = [p for t in tasks for p in build_task_datapoints(t, args, tokenizer, model_config)]
    print(f"{len(points)} datapoints; H_sig={len(sig)} heads over layers {sig_layers}")

    def lookup(S):
        """Find a cached acc for layer set S under any legacy key."""
        for k in (canonical_key(S),
                  f"single_L{next(iter(S))}" if len(S) == 1 else None,
                  f"pair_L{min(S)}_L{max(S)}" if len(S) == 2 else None):
            if k and k in results:
                return results[k]["acc"]
        return None

    def run(S):
        cached = lookup(S)
        if cached is not None:
            return cached
        keep = torch.isin(layer_of, torch.tensor(sorted(S), device=C.device))
        V = torch.where(keep[None, :, None], Csig, Cbar[None]).sum(dim=1)
        nll, acc = evaluate_fixed_v(model, model_config, tokenizer, points, V, task_index, args)
        results[canonical_key(S)] = {"layers_kept": sorted(S), "nll": nll, "acc": acc}
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"{canonical_key(S)}: acc={acc:.3f}", flush=True)
        return acc

    # --- 1. Greedy forward selection from the cached best pair ---
    pairs = sorted(((k, v) for k, v in results.items() if k.startswith("pair_")),
                   key=lambda kv: kv[1]["acc"], reverse=True)
    best_pair = set(pairs[0][1]["layers_kept"])
    greedy = set(best_pair)
    print(f"greedy start (best pair): {sorted(greedy)} acc={pairs[0][1]['acc']:.3f}")
    for step in range(2):
        scores = {l: run(greedy | {l}) for l in sig_layers if l not in greedy}
        best_l = max(scores, key=scores.get)
        greedy |= {best_l}
        print(f"greedy step {step + 1}: added L{best_l} -> {sorted(greedy)} acc={scores[best_l]:.3f}")
    greedy_acc = lookup(greedy)

    # --- 2. Exhaustive quads over a 12-layer pool ---
    def layer_score(l):
        s = results.get(f"single_L{l}", {}).get("acc", 0.0)
        pbest = max((v["acc"] for k, v in results.items()
                     if k.startswith("pair_") and l in v["layers_kept"]), default=0.0)
        return max(s, pbest)

    pool = sorted(sorted(sig_layers, key=layer_score, reverse=True)[:12])
    print(f"exhaustive pool (12 layers): {pool}")
    best_quad, best_quad_acc = None, -1.0
    for S in itertools.combinations(pool, 4):
        acc = run(set(S))
        if acc > best_quad_acc:
            best_quad, best_quad_acc = sorted(S), acc

    winner = best_quad if best_quad_acc >= (greedy_acc or 0) else sorted(greedy)
    winner_acc = max(best_quad_acc, greedy_acc or 0)
    heads_kept = [(l, h, round(float(c[l * n_heads + h].item()), 3)) for l, h in sig if l in winner]
    summary = {
        "greedy_quad": {"layers": sorted(greedy), "acc": greedy_acc},
        "exhaustive_pool": pool,
        "exhaustive_best_quad": {"layers": best_quad, "acc": best_quad_acc},
        "winner": {"layers": winner, "acc": winner_acc, "heads_kept_task_specific": heads_kept},
        "references": {k: results[k]["acc"] for k in ("no_intervention", "all_task_specific", "all_mean_ablated")},
    }
    with open(args.output_root / "layer_quad_best.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nGREEDY {sorted(greedy)}: {greedy_acc:.3f} | EXHAUSTIVE {best_quad}: {best_quad_acc:.3f}")
    print(f"WINNER {winner}: acc={winner_acc:.3f} ({len(heads_kept)} task-specific heads)")
    print(heads_kept)


if __name__ == "__main__":
    main()
