#!/usr/bin/env python
"""SANDBOX: stage-1 layer-wise mean-ablation refinement (Hu et al. 2025 §3.2) of the sparse head set.

Base set H_sig = the 73 heads with final c > 0.2 from the sparse-optimization run. FVs here are
UNWEIGHTED (paper Eq. 2, user-gated 2026-08-07): for a kept-layer set S,
    v_task = sum_{h in H_sig, layer(h) in S} h_task(h)  +  sum_{h in H_sig, layer(h) not in S} hbar(h)
where hbar = UNIFORM mean over the 20 train tasks of the out_proj-projected varicl mean head
outputs (linearity makes mean-of-projections = projection-of-mean). Injection and evaluation are
identical to train_sparse_heads.py: v added once at the cue token (block --inject_layer output)
of the same zero-shot datapoints; intervention accuracy = teacher-forced full-label accuracy.

Conditions evaluated: no intervention; all-task-specific (S = all); all-mean-ablated (S = {});
every single layer containing significant heads; every PAIR of such layers. Results cached per
condition in layer_ablation_results.json (resumable).
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.sandbox.sparse_head_selection.train_sparse_heads import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_RESULTS_ROOT,
    batch_label_logprobs,
    build_contributions,
    build_task_datapoints,
    load_train_tasks,
    make_batches,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_split_path", type=Path, default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--root_data_dir", type=str, default=str(REPO_ROOT / "dataset_files"))
    p.add_argument("--mean_acts_root", type=Path, default=DEFAULT_ARTIFACT_ROOT.parent.parent / "multitask_aie_heads_varicl")
    p.add_argument("--coeffs_path", type=Path, default=DEFAULT_ARTIFACT_ROOT / "coeffs_final.pt")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--inject_layer", type=int, default=9)
    p.add_argument("--max_queries", type=int, default=100)
    p.add_argument("--min_queries", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--sig_threshold", type=float, default=0.2)
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--output_root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    p.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return p.parse_args()


def evaluate_fixed_v(model, model_config, tokenizer, points, V, task_index, args):
    """V: (n_tasks, resid) fp32 per-task intervention vectors, or None for no intervention."""
    total_nll, n_correct = 0.0, 0
    with torch.no_grad():
        for batch in make_batches(points, args.batch_size):
            v = None
            if V is not None:
                t_idx = torch.tensor([task_index[b["task"]] for b in batch], device=V.device)
                v = V[t_idx]
            nll, accs = batch_label_logprobs(model, model_config, tokenizer, batch, v=v,
                                             inject_layer=args.inject_layer)
            total_nll += nll.sum().item()
            n_correct += sum(accs)
    return total_nll / len(points), n_correct / len(points)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / "layer_ablation_results.json"
    results = json.load(open(out_path)) if out_path.exists() else {}

    tasks = args.tasks or load_train_tasks(args)
    coeffs = torch.load(args.coeffs_path, map_location="cpu", weights_only=False)
    c = coeffs["c"].float()
    n_heads = 16
    sig = [(i // n_heads, i % n_heads) for i in range(c.numel()) if c[i].item() > args.sig_threshold]
    sig_layers = sorted({l for l, _ in sig})
    print(f"H_sig: {len(sig)} heads (c > {args.sig_threshold}) across layers {sig_layers}")

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    C = build_contributions(tasks, args, model, model_config)  # (T, 448, resid) fp32, fp16 weights
    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sig_flat = torch.tensor([l * n_heads + h for l, h in sig], device=C.device)
    Csig = C[:, sig_flat, :]                    # (T, 73, resid) task-specific contributions
    Cbar = Csig.mean(dim=0)                     # (73, resid) uniform grand mean over the 20 tasks
    layer_of = torch.tensor([l for l, _ in sig], device=C.device)
    task_index = {t: i for i, t in enumerate(tasks)}

    print("building datapoints ...")
    points = [p for t in tqdm(tasks) for p in build_task_datapoints(t, args, tokenizer, model_config)]
    print(f"{len(points)} datapoints over {len(tasks)} tasks")

    def v_for_layerset(S):
        keep = torch.isin(layer_of, torch.tensor(sorted(S), device=C.device)) if S else \
            torch.zeros(len(sig), dtype=torch.bool, device=C.device)
        V = torch.where(keep[None, :, None], Csig, Cbar[None]).sum(dim=1)  # (T, resid)
        return V

    def run(name, S):
        if name in results:
            print(f"skip {name} (cached): acc={results[name]['acc']:.3f}")
            return
        V = None if S == "none" else v_for_layerset(S)
        nll, acc = evaluate_fixed_v(model, model_config, tokenizer, points, V, task_index, args)
        results[name] = {"layers_kept": sorted(S) if S != "none" else None,
                         "nll": nll, "acc": acc}
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"{name}: acc={acc:.3f} nll={nll:.3f}", flush=True)

    run("no_intervention", "none")
    run("all_task_specific", set(sig_layers))
    run("all_mean_ablated", set())
    for l in sig_layers:
        run(f"single_L{l}", {l})
    for l1, l2 in itertools.combinations(sig_layers, 2):
        run(f"pair_L{l1}_L{l2}", {l1, l2})

    pairs = sorted(((k, v) for k, v in results.items() if k.startswith("pair_")),
                   key=lambda kv: kv[1]["acc"], reverse=True)
    singles = sorted(((k, v) for k, v in results.items() if k.startswith("single_")),
                     key=lambda kv: kv[1]["acc"], reverse=True)
    print("\n=== references ===")
    for k in ("no_intervention", "all_task_specific", "all_mean_ablated"):
        print(f"  {k}: acc={results[k]['acc']:.3f}")
    print("=== top 10 singles ==="); [print(f"  {k}: {v['acc']:.3f}") for k, v in singles[:10]]
    print("=== top 15 pairs ===");   [print(f"  {k}: {v['acc']:.3f}") for k, v in pairs[:15]]

    best_pair = pairs[0]
    kept = best_pair[1]["layers_kept"]
    heads_kept = [(l, h, round(float(c[l * n_heads + h].item()), 3)) for l, h in sig if l in kept]
    summary = {"best_pair": {"layers": kept, "acc": best_pair[1]["acc"], "nll": best_pair[1]["nll"],
                             "heads_kept_task_specific": heads_kept},
               "references": {k: results[k] for k in ("no_intervention", "all_task_specific", "all_mean_ablated")},
               "n_sig": len(sig), "sig_threshold": args.sig_threshold,
               "fv_construction": "unweighted (paper Eq. 2), uniform grand mean over 20 train tasks"}
    with open(args.output_root / "layer_ablation_best_pair.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest pair {kept}: acc={best_pair[1]['acc']:.3f}; "
          f"{len(heads_kept)} task-specific heads kept: {heads_kept}")


if __name__ == "__main__":
    main()
