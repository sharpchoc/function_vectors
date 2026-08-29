#!/usr/bin/env python
"""SANDBOX: per-task steering upper bound - task-specific isolation algorithms.

For each task (levers: write_up/isolation_methods_levers.md), fit the three isolation
algorithms using ONLY that task's 150 train prompts (dataset_files/isolation_prompts/<task>/),
crossing 1a (CIE top-10 / top-40) and 1b (sparse optimisation, K-fold-CV lambda) with the
three train metrics {zeroshot, sametask_shuffled10, mixedtask10}; 1c (per-layer cue-token
mean activation) has no train metric. Evaluate every product on the 30 paired test prompts
of each of the three settings, sweeping the injection layer 0..27 at alpha=1, full-label
teacher-forced accuracy, plus unsteered baselines.

Stages (resumable per task via artifact files): capture -> cie -> sparse -> eval.
Conventions: head/residual means from the 150 CLEAN train prompts; CIE evaluated on a
deterministic --cie_n-prompt subsample of the metric prompts; sparse trains with injection
at block --inject_layer (L9) output at the cue token; products are head-sum vectors
(sparse: heads with c > 0.8, unweighted).
"""
import argparse
import json
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from baukit import TraceDict

from src.sandbox.sparse_head_selection.train_sparse_heads import (
    batch_label_logprobs,
    split_earlystop,
    train_c,
)
from src.utils.model_utils import (
    get_attn_out_proj,
    load_gpt_model_and_tokenizer,
    set_seed,
    use_bos_literal,
)
from src.utils.prompt_utils import create_prompt, word_pairs_to_prompt_data
from src.utils.eval_utils import get_answer_id
from src.utils.varicl_utils import (
    batch_varicl_last_token_intervention,
    split_activations_by_head,
)
from src.utils.paths import ARTIFACTS_ROOT

DEFAULT_OUT = ARTIFACTS_ROOT / "sandbox" / "isolation_upper_bound"
TRAIN_METRICS = ["zeroshot", "sametask_shuffled10", "mixedtask10"]
TEST_SETTINGS = ["test_zeroshot", "test_sametask_shuffled10", "test_mixedtask10"]
LAMBDAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--stage", choices=["all", "capture", "cie", "sparse", "eval"], default="all")
    p.add_argument("--prompts_root", type=Path, default=REPO_ROOT / "dataset_files" / "isolation_prompts")
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inject_layer", type=int, default=9, help="Sparse-opt TRAINING injection layer.")
    p.add_argument("--cie_n", type=int, default=50, help="CIE eval-prompt subsample size.")
    p.add_argument("--cie_batch", type=int, default=8)
    p.add_argument("--capture_batch", type=int, default=8)
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.005, 0.01, 0.05, 0.2])
    p.add_argument("--c_high", type=float, default=0.8)
    p.add_argument("--metrics", nargs="+", default=None, choices=TRAIN_METRICS,
                   help="Subset of train metrics for the sparse stage (per-metric pod sharding).")
    # train_c hyperparameters, RESCALED for 150-datapoint task-specific runs: batch 128 gives
    # ~2 optimizer steps/epoch, so the pooled-run settings (lr .01, 30 epochs) truncated
    # training at ~58 steps with c_max stuck ~0.8 (2026-08-13 bug).
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--earlystop_frac", type=float, default=0.1)
    p.add_argument("--init_c", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.2)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Record adapters
# ---------------------------------------------------------------------------

def auto_batch(max_len, token_budget, cap):
    """Batch size bounded by a token-rows budget (long-prompt tasks like commonsense_qa
    reach ~1200 tokens per 10-shot prompt and OOM fixed batch sizes)."""
    return max(1, min(cap, token_budget // max(max_len, 1)))


def max_prompt_len(points):
    return max(len(p["prompt_ids"]) + len(p["label_ids"]) for p in points)


def record_to_prompt_data(rec, model_config):
    word_pairs = {"input": [d["input"] for d in rec["demos"]],
                  "output": [d["output"] for d in rec["demos"]]}
    return word_pairs_to_prompt_data(
        word_pairs, query_target_pair=dict(rec["query"]),
        prepend_bos_token=use_bos_literal(model_config), shuffle_labels=False)


def record_to_point(rec, tokenizer, model_config):
    prompt_data = record_to_prompt_data(rec, model_config)
    target = prompt_data["query_target"]["output"]
    target = target[0] if isinstance(target, list) else target
    sentence = create_prompt(prompt_data)
    assert sentence.rstrip().endswith("A:"), f"prompt does not end at cue: {sentence[-40:]!r}"
    prompt_ids = tokenizer(sentence, truncation=False, padding=False).input_ids
    label_ids = get_answer_id(sentence, target, tokenizer)
    if isinstance(label_ids, int):
        label_ids = [label_ids]
    return {"task": rec["task"], "query": rec["query"]["input"], "target": target,
            "prompt_ids": prompt_ids, "label_ids": list(label_ids),
            "cue_idx": len(prompt_ids) - 1, "prompt_index": rec["prompt_index"]}


def load_records(args, task, name):
    with open(args.prompts_root / task / f"{name}.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stage: capture (head means + per-layer residual means over 150 clean prompts)
# ---------------------------------------------------------------------------

def stage_capture(args, task, model, model_config, tokenizer):
    out = args.out_root / task / "means.pt"
    if out.exists():
        return torch.load(out, map_location="cpu", weights_only=False)
    recs = load_records(args, task, "train_prompts")
    # 150 in the standard prompt sets; small-pool tasks are capped by the generator
    # (qwen25 pool: next_in_group/next_in_period have 66 examples). Floor guards truncation.
    assert len(recs) >= 40, f"{task}: only {len(recs)} train prompts"
    n_layers, n_heads, resid = model_config["n_layers"], model_config["n_heads"], model_config["resid_dim"]
    head_dim = resid // n_heads

    head_sum = torch.zeros(n_layers, n_heads, head_dim, dtype=torch.float64)
    resid_sum = torch.zeros(n_layers, resid, dtype=torch.float64)
    n_seen = 0
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    linearity_checked = False
    try:
        sentences_all = [create_prompt(record_to_prompt_data(r, model_config)) for r in recs]
        max_tok = max(len(tokenizer(s).input_ids) for s in sentences_all)
        cap_batch = auto_batch(max_tok, 4000, args.capture_batch)
        for start in range(0, len(recs), cap_batch):
            batch = recs[start:start + cap_batch]
            sentences = [create_prompt(record_to_prompt_data(r, model_config)) for r in batch]
            inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
            prompt_lens = inputs.attention_mask.sum(dim=1) - 1
            bidx = torch.arange(len(sentences), device=model.device)
            with torch.no_grad(), TraceDict(model, layers=model_config["attn_hook_names"] +
                                            model_config["layer_hook_names"],
                                            retain_input=True, retain_output=True) as td:
                model(**inputs)
            for li, lname in enumerate(model_config["attn_hook_names"]):
                inp = td[lname].input
                inp = inp[0] if isinstance(inp, tuple) else inp
                heads = split_activations_by_head(inp, model_config)  # (B, seq, H, hd)
                cue = heads[bidx, prompt_lens]                        # (B, H, hd)
                head_sum[li] += cue.double().sum(dim=0).cpu()
                if not linearity_checked:
                    # linearity gate: sum over heads of W_O slices @ head acts == attn output
                    w = get_attn_out_proj(model, li).weight.detach()
                    rebuilt = torch.einsum("bhd,ehd->be", cue.to(w.dtype),
                                           w.view(resid, n_heads, head_dim))
                    outp = td[lname].output
                    outp = outp[0] if isinstance(outp, tuple) else outp
                    ref = outp[bidx, prompt_lens]
                    dev = (rebuilt - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
                    assert dev < 5e-2, f"linearity gate failed at layer {li}: rel dev {dev:.3e}"
            linearity_checked = True
            for li, lname in enumerate(model_config["layer_hook_names"]):
                outp = td[lname].output
                outp = outp[0] if isinstance(outp, tuple) else outp
                resid_sum[li] += outp[bidx, prompt_lens].double().sum(dim=0).cpu()
            n_seen += len(sentences)
    finally:
        tokenizer.padding_side = old_side
    assert n_seen == len(recs)
    means = {"head_means": (head_sum / n_seen).float(),
             "resid_means": (resid_sum / n_seen).float(), "n_prompts": n_seen}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(means, out)
    print(f"[{task}] capture done (linearity gate passed)", flush=True)
    return means


def build_contributions_single(head_means, model, model_config):
    """C[l*H+h] = W_O^{l,h} @ head_mean[l,h]  -> (448, resid) fp32 on device."""
    n_layers, n_heads, resid = model_config["n_layers"], model_config["n_heads"], model_config["resid_dim"]
    head_dim = resid // n_heads
    C = torch.zeros(n_layers * n_heads, resid, dtype=torch.float32, device=model.device)
    hm = head_means.to(model.device)
    for l in range(n_layers):
        w = get_attn_out_proj(model, l).weight.detach().float().view(resid, n_heads, head_dim)
        C[l * n_heads:(l + 1) * n_heads] = torch.einsum("ohd,hd->ho", w, hm[l].float())
    return C


# ---------------------------------------------------------------------------
# Stage: CIE per train metric
# ---------------------------------------------------------------------------

def stage_cie(args, task, model, model_config, tokenizer, head_means):
    results = {}
    for m in TRAIN_METRICS:
        out = args.out_root / task / f"cie_{m}.pt"
        if out.exists():
            results[m] = torch.load(out, map_location="cpu", weights_only=False)["cie"]
            continue
        recs = load_records(args, task, f"train_{m}")
        rng = np.random.RandomState(args.seed + 11_000_000 + (zlib.crc32(task.encode()) % 100000))
        idx = rng.choice(len(recs), size=min(args.cie_n, len(recs)), replace=False)
        sub = [recs[i] for i in idx]
        acc = torch.zeros(model_config["n_layers"], model_config["n_heads"], dtype=torch.float64)
        n = 0
        with torch.no_grad():
            pds = [record_to_prompt_data(r, model_config) for r in sub]
            max_tok = max(len(tokenizer(create_prompt(pd)).input_ids) for pd in pds)
            cie_batch = auto_batch(max_tok, 3000, args.cie_batch)
            for start in tqdm(range(0, len(pds), cie_batch), desc=f"{task} cie {m}"):
                batch = pds[start:start + cie_batch]
                ie = batch_varicl_last_token_intervention(batch, head_means, model, model_config, tokenizer)
                acc += ie.double().sum(dim=0).cpu()
                n += ie.shape[0]
        cie = (acc / n).float()
        torch.save({"cie": cie, "n_prompts": n, "prompt_indices": [int(i) for i in idx]}, out)
        results[m] = cie
        print(f"[{task}] cie {m} done (n={n})", flush=True)
    return results


def top_heads(cie, n):
    flat = torch.argsort(cie.flatten(), descending=True)[:n]
    return [(int(i) // cie.shape[1], int(i) % cie.shape[1]) for i in flat]


# ---------------------------------------------------------------------------
# Stage: sparse optimisation per train metric (K-fold CV for lambda)
# ---------------------------------------------------------------------------

def make_tc_args(args, metric, points):
    # micro-batch bounded by token budget (grad activations above inject_layer dominate)
    micro = auto_batch(max_prompt_len(points), 2400, 32)
    return types.SimpleNamespace(
        init_c=args.init_c, lr=args.lr, max_epochs=args.max_epochs,
        micro_batch_size=micro, batch_size=args.batch_size,
        inject_layer=args.inject_layer, patience=args.patience,
        threshold=args.threshold, earlystop_frac=args.earlystop_frac)


def eval_points_fixed_v(model, model_config, tokenizer, points, v, inject_layer, batch_size=None,
                        token_budget=6000, batch_cap=32):
    # logits are materialized [B, seq, vocab] and .float()ed - bound B by token budget
    # (large-vocab models like Qwen2.5, 152k, need ~2500/16 instead of the GPT-J 6000/32)
    if batch_size is None:
        batch_size = auto_batch(max_prompt_len(points), token_budget, batch_cap)
    n_correct = 0
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            vb = None if v is None else v.unsqueeze(0).expand(len(batch), -1)
            _, accs = batch_label_logprobs(model, model_config, tokenizer, batch,
                                           v=vb, inject_layer=inject_layer)
            n_correct += sum(accs)
    return n_correct / len(points)


def select_heads_nonempty(c, c_high):
    """c > c_high head indices; NEVER empty - falls back to the top-10 by coefficient."""
    sel = torch.nonzero(c > c_high).flatten().tolist()
    if sel:
        return sel, False
    return torch.argsort(c, descending=True)[:10].tolist(), True


def stage_sparse(args, task, model, model_config, tokenizer, C, metrics=None):
    task_index = {task: 0}
    C3 = C.unsqueeze(0)  # (1, 448, resid) for train_c/evaluate_points compatibility
    results = {}
    for m in (metrics or TRAIN_METRICS):
        final_path = args.out_root / task / f"sparse_{m}" / "final.pt"
        if final_path.exists():
            results[m] = torch.load(final_path, map_location="cpu", weights_only=False)
            continue
        recs = load_records(args, task, f"train_{m}")
        points = [record_to_point(r, tokenizer, model_config) for r in recs]
        tc_args = make_tc_args(args, m, points)
        (args.out_root / task / f"sparse_{m}").mkdir(parents=True, exist_ok=True)

        # K-fold CV over prompts for lambda; fold eval on the WEIGHTED c vector (no
        # threshold cliff - thresholding is applied only to the final deployed product).
        rng = np.random.RandomState(args.seed + 13_000_000 + (zlib.crc32((task + m).encode()) % 100000))
        order = rng.permutation(len(points))
        folds = np.array_split(order, args.kfolds)
        per_lambda, fold_table = {}, {}
        for lam in args.lambdas:
            accs = []
            for fi, fold in enumerate(folds):
                fold_path = args.out_root / task / f"sparse_{m}" / f"lambda{lam:g}_fold{fi}.pt"
                if fold_path.exists():
                    accs.append(torch.load(fold_path, map_location="cpu", weights_only=False)["fold_acc"])
                    continue
                heldout = [points[i] for i in fold]
                train_pool = [points[i] for i in order if i not in set(fold.tolist())]
                run_seed = args.seed + 1000 * args.lambdas.index(lam) + fi
                tr, es = split_earlystop(train_pool, tc_args.earlystop_frac, run_seed)
                c, history, best_epoch = train_c(model, model_config, tokenizer, tr, es, C3,
                                                 task_index, lam, tc_args, run_seed,
                                                 desc=f"{task} {m} lam={lam:g} fold{fi}")
                v = (c.unsqueeze(1) * C).sum(dim=0)  # weighted, no threshold
                fold_acc = eval_points_fixed_v(model, model_config, tokenizer, heldout, v,
                                               args.inject_layer)
                torch.save({"lambda": lam, "fold": fi, "c": c.cpu(), "fold_acc": fold_acc,
                            "best_epoch": best_epoch, "fold_eval": "weighted_c"}, fold_path)
                accs.append(fold_acc)
            per_lambda[lam] = float(np.mean(accs))
            fold_table[lam] = accs
        # strict best mean fold accuracy; ties -> smaller lambda
        best = max(per_lambda.values())
        chosen = min(l for l in args.lambdas if per_lambda[l] == best)

        run_seed = args.seed + 999
        tr, es = split_earlystop(points, tc_args.earlystop_frac, run_seed)
        c_final, history, best_epoch = train_c(model, model_config, tokenizer, tr, es, C3,
                                               task_index, chosen, tc_args, run_seed,
                                               desc=f"{task} {m} FINAL lam={chosen:g}")
        sel, fallback = select_heads_nonempty(c_final, args.c_high)
        res = {"c": c_final.cpu(), "chosen_lambda": chosen, "per_lambda": per_lambda,
               "fold_accs": fold_table, "selected_heads": sel, "n_selected": len(sel),
               "fallback_top10": fallback, "final_best_epoch": best_epoch,
               "hyperparams": {"lr": args.lr, "max_epochs": args.max_epochs,
                               "patience": args.patience, "batch_size": args.batch_size}}
        torch.save(res, final_path)
        results[m] = res
        print(f"[{task}] sparse {m} done: lam={chosen:g} n_sel={len(sel)}"
              f"{' (FALLBACK top-10)' if fallback else ''} best_epoch={best_epoch}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Stage: eval (products x test settings x layers)
# ---------------------------------------------------------------------------

def stage_eval(args, task, model, model_config, tokenizer, C, cie, sparse, resid_means):
    out = args.out_root / task / "eval_results.json"
    if out.exists():
        print(f"[{task}] eval exists, skip", flush=True)
        return
    n_layers = model_config["n_layers"]

    products = {}
    for m in TRAIN_METRICS:
        for nh, name in ((10, "cie10"), (40, "cie40")):
            heads = top_heads(cie[m], nh)
            idx = torch.tensor([l * model_config["n_heads"] + h for l, h in heads], device=C.device)
            products[f"{name}|{m}"] = {"v": C[idx].sum(dim=0), "heads": heads}
        sel = sparse[m]["selected_heads"]
        v = C[torch.tensor(sel, device=C.device)].sum(dim=0) if sel else torch.zeros_like(C[0])
        products[f"sparse|{m}"] = {"v": v, "heads": sel,
                                   "chosen_lambda": sparse[m]["chosen_lambda"]}

    resid_means = resid_means.to(model.device).float()
    results = {}
    for setting in TEST_SETTINGS:
        recs = load_records(args, task, setting)
        points = [record_to_point(r, tokenizer, model_config) for r in recs]
        baseline = eval_points_fixed_v(model, model_config, tokenizer, points, None, 9)
        entry = {"baseline": baseline, "products": {}}
        for pname, prod in products.items():
            accs = [eval_points_fixed_v(model, model_config, tokenizer, points,
                                        prod["v"].float(), L) for L in range(n_layers)]
            entry["products"][pname] = accs
        entry["products"]["mean_act"] = [
            eval_points_fixed_v(model, model_config, tokenizer, points, resid_means[L], L)
            for L in range(n_layers)]
        results[setting] = entry
        print(f"[{task}] eval {setting}: baseline={baseline:.3f} done", flush=True)

    meta = {"task": task, "alpha": 1.0, "readout": "full-label teacher-forced accuracy",
            "n_test_prompts": 30, "layers": list(range(n_layers)),
            "product_heads": {k: v.get("heads") for k, v in products.items()},
            "sparse_lambdas": {m: sparse[m]["chosen_lambda"] for m in TRAIN_METRICS},
            "settings": results}
    with open(out, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[{task}] eval done -> {out}", flush=True)


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name)
    model.eval()
    torch.set_grad_enabled(False)

    # Pass 1: captures + contributions for ALL tasks in the as-loaded (fp16) dtype, so
    # means/C are consistent across tasks and match repo capture conventions.
    means_by_task, C_by_task = {}, {}
    for task in args.tasks:
        (args.out_root / task).mkdir(parents=True, exist_ok=True)
        means_by_task[task] = stage_capture(args, task, model, model_config, tokenizer)
        C_by_task[task] = build_contributions_single(means_by_task[task]["head_means"],
                                                     model, model_config)
    if args.stage == "capture":
        return

    # Pass 2: everything else in bf16 (train_sparse_heads convention).
    model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    for task in args.tasks:
        means, C = means_by_task[task], C_by_task[task]
        cie = stage_cie(args, task, model, model_config, tokenizer, means["head_means"])
        if args.stage == "cie":
            continue
        torch.set_grad_enabled(True)  # train_c needs grad; its eval paths use no_grad
        stage_sparse(args, task, model, model_config, tokenizer, C, metrics=args.metrics)
        torch.set_grad_enabled(False)
        if args.stage == "sparse":
            continue
        # eval needs ALL 3 metrics' sparse results; other pods may own some metrics -
        # wait on the shared volume (per-metric pod sharding), then load.
        import time
        deadline = time.time() + 3 * 3600
        while True:
            missing = [m for m in TRAIN_METRICS
                       if not (args.out_root / task / f"sparse_{m}" / "final.pt").exists()]
            if not missing:
                break
            if time.time() > deadline:
                raise RuntimeError(f"{task}: sparse results still missing after wait: {missing}")
            print(f"[{task}] eval waiting for sparse: {missing}", flush=True)
            time.sleep(60)
        sparse = {m: torch.load(args.out_root / task / f"sparse_{m}" / "final.pt",
                                map_location="cpu", weights_only=False) for m in TRAIN_METRICS}
        stage_eval(args, task, model, model_config, tokenizer, C, cie, sparse,
                   means["resid_means"])
    print("ALL TASKS DONE")


if __name__ == "__main__":
    main()
