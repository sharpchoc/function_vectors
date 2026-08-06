#!/usr/bin/env python
"""SANDBOX: sparse-optimization head selection (Hu et al. 2025, arXiv:2505.05145 §3.1) on GPT-J.

NOT a repo default. Learns a single coefficient vector c in [0,1]^(n_layers*n_heads) over all
attention heads, shared across the 20 train tasks of the abstractive 29-task split, such that
adding v_task(c) = sum_h c_h * (out_proj-projected varicl mean head output of task) to the
residual stream (output of transformer block --inject_layer, final cue token only) of a
ZERO-SHOT prompt "Q: x\nA:" maximizes p(full label) under teacher forcing.

Loss per datapoint: raw -log p(full label) = summed token CE (greedy contextualized label
tokens via get_answer_id). The intervention vector is injected ONCE at the cue token; the
teacher-forced label positions receive no intervention.

Objective: L(c) = mean_datapoints[-log p(label)] + lambda * ||c||_1, AdamW(lr=0.01), batch 128,
c clamped to [0,1] after every step (paper defaults). Lambda selected by leave-one-TASK-out CV
over the 20 train tasks; final c retrained on all 20 tasks at the chosen lambda.

Modes:
  check  - FV-construction consistency check against stored train_varicl_top40 FVs (hard-stop on
           mismatch; the user adjudicates).
  smoke  - tiny end-to-end run (subset tasks/queries/epochs/lambdas).
  cv     - run LOTO folds for --lambdas (resumable; one .pt per (lambda, fold), skipped if done).
  reduce - aggregate folds, apply the selection rule, retrain final model, write artifacts,
           results tables and the summary figure, compute baselines.
  all    - cv then reduce.
"""
import argparse
import json
import math
import sys
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
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.utils.eval_utils import get_answer_id
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.prompt_utils import create_prompt, load_dataset, word_pairs_to_prompt_data
from src.utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT

DEFAULT_ARTIFACT_ROOT = ARTIFACTS_ROOT / "sandbox" / "sparse_head_selection"
DEFAULT_RESULTS_ROOT = RESULTS_ROOT / "sandbox" / "sparse_head_selection"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["check", "smoke", "cv", "reduce", "all"], default="all")
    p.add_argument("--task_split_path", type=Path, default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--root_data_dir", type=str, default=str(REPO_ROOT / "dataset_files"))
    p.add_argument("--mean_acts_root", type=Path, default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl")
    p.add_argument("--canonical_heads_path", type=Path,
                   default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl" / "multitask_top_aie_heads.pt")
    p.add_argument("--canonical_fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "function_vectors" / "gpt-j" / "train_varicl_top40")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                   help="Model dtype (repo loader gives fp16; bf16 is safer for backprop).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3,
                   help="load_dataset split fraction; with --seed 42 matches the varicl stage-1 splits.")
    p.add_argument("--inject_layer", type=int, default=9)
    p.add_argument("--max_queries", type=int, default=100)
    p.add_argument("--min_queries", type=int, default=80)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--micro_batch_size", type=int, default=32,
                   help="Forward/backward chunk size inside each optimizer batch (gradient "
                        "accumulation); the objective is identical to a full --batch_size step.")
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--earlystop_frac", type=float, default=0.1)
    p.add_argument("--init_c", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.2, help="Significant-head threshold on c (paper's 0.2).")
    p.add_argument("--accuracy_tolerance", type=float, default=0.01,
                   help="Lambda rule: largest lambda within this absolute mean-LOTO-accuracy of the best.")
    p.add_argument("--tasks", nargs="+", default=None, help="Override task list (smoke/debug only).")
    p.add_argument("--output_root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    p.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data building
# ---------------------------------------------------------------------------

def load_train_tasks(args):
    if args.tasks:
        return list(args.tasks)
    with open(args.task_split_path) as f:
        split = json.load(f)
    return list(split["train_tasks"])


def build_task_datapoints(task, args, tokenizer, model_config):
    """Zero-shot datapoints for one task: valid split capped at --max_queries; if the valid
    split has fewer than --min_queries examples, use all of it and top up from the train split
    (seeded) to exactly --min_queries. Returns a list of dicts."""
    dataset = load_dataset(task, root_data_dir=args.root_data_dir, test_size=args.test_split, seed=args.seed)
    rng = np.random.RandomState(args.seed + (zlib.crc32(task.encode()) % 100000))

    n_valid = len(dataset["valid"])
    picks = []  # (split_name, index)
    if n_valid >= args.max_queries:
        idx = rng.choice(n_valid, args.max_queries, replace=False)
        picks = [("valid", int(i)) for i in idx]
    else:
        picks = [("valid", i) for i in range(n_valid)]
        if n_valid < args.min_queries:
            n_top_up = args.min_queries - n_valid
            n_train = len(dataset["train"])
            if n_top_up > n_train:
                raise RuntimeError(f"{task}: cannot top up to {args.min_queries} queries "
                                   f"(valid={n_valid}, train={n_train})")
            idx = rng.choice(n_train, n_top_up, replace=False)
            picks += [("train", int(i)) for i in idx]

    prepend_bos = not model_config["prepend_bos"]
    points = []
    for split_name, i in picks:
        word_pairs = {"input": [], "output": []}
        word_pairs_test = dataset[split_name][i]
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test,
                                                prepend_bos_token=prepend_bos, shuffle_labels=False)
        target = prompt_data["query_target"]["output"]
        target = target[0] if isinstance(target, list) else target
        sentence = create_prompt(prompt_data)
        prompt_ids = tokenizer(sentence, truncation=False, padding=False).input_ids
        label_ids = get_answer_id(sentence, target, tokenizer)
        points.append({
            "task": task,
            "source_split": split_name,
            "query": prompt_data["query_target"]["input"],
            "target": target,
            "prompt_ids": prompt_ids,
            "label_ids": label_ids,
            "cue_idx": len(prompt_ids) - 1,
        })
    return points


# ---------------------------------------------------------------------------
# Head contributions
# ---------------------------------------------------------------------------

def build_contributions(tasks, args, model, model_config):
    """C[t, layer*n_heads + head, :] = out_proj_layer[:, head_block] @ mean_act[t, layer, head].
    GPT-J's out_proj has no bias, so summing per-head contributions is exact. fp32, on device."""
    n_layers, n_heads, resid = model_config["n_layers"], model_config["n_heads"], model_config["resid_dim"]
    head_dim = resid // n_heads
    device = model.device

    means = []
    for task in tasks:
        path = args.mean_acts_root / task / f"{task}_mean_head_activations_varicl.pt"
        m = torch.load(path, map_location="cpu", weights_only=False)
        assert m.shape == (n_layers, n_heads, head_dim), f"{path}: unexpected shape {tuple(m.shape)}"
        means.append(m.float())
    means = torch.stack(means).to(device)  # (T, L, H, d)

    C = torch.zeros(len(tasks), n_layers * n_heads, resid, device=device, dtype=torch.float32)
    for layer in range(n_layers):
        w = model.transformer.h[layer].attn.out_proj.weight.detach().float()  # (out, in)
        w = w.view(resid, n_heads, head_dim)  # (o, h, d)
        C[:, layer * n_heads:(layer + 1) * n_heads, :] = torch.einsum("ohd,thd->tho", w, means[:, layer])
    return C


def consistency_check(tasks, args, C, model_config):
    """v built from an indicator c over the canonical top-40 heads must match the stored
    train_varicl_top40 FVs. Mismatch = hard stop (user adjudicates data discrepancies)."""
    n_heads = model_config["n_heads"]
    canonical = load_canonical_heads(args)
    flat_idx = torch.tensor([l * n_heads + h for l, h, *_ in canonical], device=C.device)
    worst = {"task": None, "rel_err": 0.0, "cos": 1.0}
    for t, task in enumerate(tasks):
        fv_path = args.canonical_fv_root / task / f"{task}_function_vector.pt"
        if not fv_path.exists():
            raise FileNotFoundError(f"consistency check: missing stored FV {fv_path}")
        stored = torch.load(fv_path, map_location="cpu", weights_only=False)
        if isinstance(stored, dict):
            stored = stored.get("function_vector", stored)
        stored = torch.as_tensor(stored).float().flatten().to(C.device)
        built = C[t, flat_idx].sum(dim=0)
        rel_err = (built - stored).norm().item() / max(stored.norm().item(), 1e-12)
        cos = torch.nn.functional.cosine_similarity(built, stored, dim=0).item()
        if rel_err > worst["rel_err"]:
            worst = {"task": task, "rel_err": rel_err, "cos": cos}
        if rel_err > 2e-2 or cos < 0.9995:
            raise RuntimeError(
                f"CONSISTENCY CHECK FAILED for {task}: rel_err={rel_err:.4e}, cos={cos:.6f} vs stored "
                f"{fv_path}. HARD STOP - do not proceed; report to user for adjudication.")
    print(f"Consistency check PASSED on {len(tasks)} tasks. "
          f"Worst: {worst['task']} rel_err={worst['rel_err']:.3e} cos={worst['cos']:.6f}")
    return worst


def load_canonical_heads(args):
    obj = torch.load(args.canonical_heads_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        obj = obj["top_heads"]
    return [(int(l), int(h), float(s)) for l, h, s in obj]


# ---------------------------------------------------------------------------
# Forward pass with cue-token injection
# ---------------------------------------------------------------------------

def make_batches(points, batch_size, rng=None):
    order = np.arange(len(points))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [points[i] for i in order[start:start + batch_size]]


def batch_label_logprobs(model, model_config, tokenizer, batch, v=None, inject_layer=9):
    """One teacher-forced forward pass over [prompt || label] per sample; v (B, resid) fp32 or
    None is added to the output hidden state of block `inject_layer` at each sample's cue index
    only. Returns (per-sample -log p(full label) [fp32, differentiable wrt v],
    per-sample all-tokens-argmax-correct [bool])."""
    device = model.device
    pad_id = tokenizer.pad_token_id
    seqs = [b["prompt_ids"] + b["label_ids"] for b in batch]
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attention_mask[i, :len(s)] = 1
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    cue_idx = torch.tensor([b["cue_idx"] for b in batch], device=device)
    batch_arange = torch.arange(len(batch), device=device)

    handle = None
    if v is not None:
        block = model.transformer.h[inject_layer]

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            add = torch.zeros_like(hidden, dtype=torch.float32)
            add[batch_arange, cue_idx] = v
            hidden = hidden + add.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden

        handle = block.register_forward_hook(hook)
    try:
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    finally:
        if handle is not None:
            handle.remove()

    neg_logps, accs = [], []
    logits = logits.float()
    for i, b in enumerate(batch):
        n_prompt, n_label = len(b["prompt_ids"]), len(b["label_ids"])
        pred_positions = torch.arange(n_prompt - 1, n_prompt - 1 + n_label, device=device)
        label_tok = torch.tensor(b["label_ids"], device=device)
        lp = torch.log_softmax(logits[i, pred_positions], dim=-1)
        neg_logps.append(-lp[torch.arange(n_label, device=device), label_tok].sum())
        accs.append(bool((logits[i, pred_positions].argmax(dim=-1) == label_tok).all().item()))
    return torch.stack(neg_logps), accs


def evaluate_points(model, model_config, tokenizer, points, C, task_index, c, args):
    """Mean -log p(label) and full-label teacher-forced accuracy. c=None -> no intervention."""
    total_nll, n_correct = 0.0, 0
    with torch.no_grad():
        for batch in make_batches(points, args.batch_size):
            v = None
            if c is not None:
                t_idx = torch.tensor([task_index[b["task"]] for b in batch], device=C.device)
                v = torch.einsum("h,bhd->bd", c, C[t_idx])
            nll, accs = batch_label_logprobs(model, model_config, tokenizer, batch, v=v,
                                             inject_layer=args.inject_layer)
            total_nll += nll.sum().item()
            n_correct += sum(accs)
    return total_nll / len(points), n_correct / len(points)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def split_earlystop(points, frac, seed):
    """Stratified-by-task early-stop slice."""
    by_task = {}
    for p in points:
        by_task.setdefault(p["task"], []).append(p)
    train, es = [], []
    rng = np.random.RandomState(seed)
    for task in sorted(by_task):
        pts = by_task[task]
        order = rng.permutation(len(pts))
        n_es = max(1, int(round(frac * len(pts))))
        es += [pts[i] for i in order[:n_es]]
        train += [pts[i] for i in order[n_es:]]
    return train, es


def train_c(model, model_config, tokenizer, train_points, es_points, C, task_index, lam, args, run_seed, desc=""):
    device = C.device
    n_units = C.shape[1]
    c = torch.full((n_units,), args.init_c, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.AdamW([c], lr=args.lr)
    rng = np.random.RandomState(run_seed)

    best = {"es_loss": math.inf, "c": c.detach().clone(), "epoch": -1}
    history = []
    epochs_since_best = 0
    for epoch in range(args.max_epochs):
        epoch_loss, n_seen = 0.0, 0
        for batch in make_batches(train_points, args.batch_size, rng=rng):
            opt.zero_grad(set_to_none=True)
            batch_nll_sum = 0.0
            for mb_start in range(0, len(batch), args.micro_batch_size):
                micro = batch[mb_start:mb_start + args.micro_batch_size]
                t_idx = torch.tensor([task_index[b["task"]] for b in micro], device=device)
                v = torch.einsum("h,bhd->bd", c, C[t_idx])
                nll, _ = batch_label_logprobs(model, model_config, tokenizer, micro, v=v,
                                              inject_layer=args.inject_layer)
                (nll.sum() / len(batch)).backward()
                batch_nll_sum += nll.sum().item()
            (lam * c.abs().sum()).backward()
            if epoch == 0 and n_seen == 0:
                assert c.grad is not None and torch.isfinite(c.grad).all() and c.grad.abs().sum() > 0, \
                    "no/invalid gradient reached c - the injection hook is breaking grad flow"
            opt.step()
            with torch.no_grad():
                c.clamp_(0.0, 1.0)
            epoch_loss += batch_nll_sum
            n_seen += len(batch)

        es_loss, es_acc = evaluate_points(model, model_config, tokenizer, es_points, C, task_index,
                                          c.detach(), args)
        history.append({"epoch": epoch, "train_nll": epoch_loss / max(n_seen, 1),
                        "es_nll": es_loss, "es_acc": es_acc,
                        "n_active": int((c.detach() > args.threshold).sum().item()),
                        "l1": float(c.detach().sum().item())})
        print(f"  [{desc}] epoch {epoch}: train_nll={history[-1]['train_nll']:.4f} "
              f"es_nll={es_loss:.4f} es_acc={es_acc:.3f} active={history[-1]['n_active']}", flush=True)
        if es_loss < best["es_loss"] - 1e-4:
            best = {"es_loss": es_loss, "c": c.detach().clone(), "epoch": epoch}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                break
    return best["c"], history, best["epoch"]


# ---------------------------------------------------------------------------
# CV / reduce
# ---------------------------------------------------------------------------

def fold_path(args, lam, fold_task):
    return args.output_root / "fold_results" / f"lambda{lam:g}_fold_{fold_task}.pt"


def run_cv(tasks, points_by_task, model, model_config, tokenizer, C, task_index, args):
    (args.output_root / "fold_results").mkdir(parents=True, exist_ok=True)
    for lam in args.lambdas:
        for fi, fold_task in enumerate(tasks):
            out = fold_path(args, lam, fold_task)
            if out.exists():
                print(f"skip existing {out.name}")
                continue
            train_pool = [p for t in tasks if t != fold_task for p in points_by_task[t]]
            run_seed = args.seed + 1000 * (args.lambdas.index(lam) + 1) + fi
            train_points, es_points = split_earlystop(train_pool, args.earlystop_frac, run_seed)
            c, history, best_epoch = train_c(model, model_config, tokenizer, train_points, es_points,
                                             C, task_index, lam, args, run_seed,
                                             desc=f"lam={lam:g} fold={fold_task}")
            fold_nll, fold_acc = evaluate_points(model, model_config, tokenizer,
                                                 points_by_task[fold_task], C, task_index, c, args)
            torch.save({"lambda": lam, "fold_task": fold_task, "c": c.cpu(),
                        "fold_nll": fold_nll, "fold_acc": fold_acc, "best_epoch": best_epoch,
                        "history": history, "run_seed": run_seed}, out)
            print(f"[lam={lam:g} fold={fold_task}] heldout nll={fold_nll:.4f} acc={fold_acc:.3f} "
                  f"active={(c > args.threshold).sum().item()}", flush=True)


def select_lambda(tasks, args):
    per_lambda = {}
    for lam in args.lambdas:
        rows = []
        for fold_task in tasks:
            path = fold_path(args, lam, fold_task)
            if not path.exists():
                raise FileNotFoundError(f"reduce: missing fold result {path} - run --mode cv first")
            r = torch.load(path, map_location="cpu", weights_only=False)
            rows.append(r)
        per_lambda[lam] = {
            "mean_acc": float(np.mean([r["fold_acc"] for r in rows])),
            "mean_nll": float(np.mean([r["fold_nll"] for r in rows])),
            "mean_active": float(np.mean([(r["c"] > args.threshold).sum().item() for r in rows])),
            "per_fold": [{"fold_task": r["fold_task"], "acc": r["fold_acc"], "nll": r["fold_nll"],
                          "n_active": int((r["c"] > args.threshold).sum().item())} for r in rows],
        }
    best_acc = max(v["mean_acc"] for v in per_lambda.values())
    eligible = [lam for lam in args.lambdas if per_lambda[lam]["mean_acc"] >= best_acc - args.accuracy_tolerance]
    chosen = max(eligible)
    return chosen, per_lambda


def run_reduce(tasks, points_by_task, model, model_config, tokenizer, C, task_index, args, consistency):
    chosen, per_lambda = select_lambda(tasks, args)
    print(f"Chosen lambda={chosen:g} "
          f"(rule: largest within {args.accuracy_tolerance} of best mean LOTO accuracy)")

    # Final model on all 20 tasks.
    final_c_path = args.output_root / "coeffs_final.pt"
    if final_c_path.exists():
        saved = torch.load(final_c_path, map_location="cpu", weights_only=False)
        c_final, final_history = saved["c"].to(C.device), saved["history"]
        print("loaded existing final coefficients")
    else:
        all_points = [p for t in tasks for p in points_by_task[t]]
        run_seed = args.seed + 999
        train_points, es_points = split_earlystop(all_points, args.earlystop_frac, run_seed)
        c_final, final_history, _ = train_c(model, model_config, tokenizer, train_points, es_points,
                                            C, task_index, chosen, args, run_seed, desc=f"FINAL lam={chosen:g}")
        torch.save({"c": c_final.cpu(), "lambda": chosen, "history": final_history,
                    "run_seed": run_seed}, final_c_path)

    n_heads = model_config["n_heads"]
    selected = [(i // n_heads, i % n_heads, round(float(c_final[i].item()), 6))
                for i in torch.argsort(c_final, descending=True).tolist()
                if c_final[i].item() > args.threshold]
    canonical = load_canonical_heads(args)
    canonical_set = {(l, h) for l, h, _ in canonical}
    overlap = [lh for lh in [(l, h) for l, h, _ in selected] if lh in canonical_set]

    # Baselines per task on the same datapoints: no intervention / canonical top-40 at inject layer.
    baselines_path = args.output_root / "baselines.json"
    if baselines_path.exists():
        with open(baselines_path) as f:
            baselines = json.load(f)
    else:
        flat = torch.zeros(C.shape[1], device=C.device)
        for l, h, _ in canonical:
            flat[l * n_heads + h] = 1.0
        baselines = {}
        for task in tqdm(tasks, desc="baselines"):
            nll0, acc0 = evaluate_points(model, model_config, tokenizer, points_by_task[task],
                                         C, task_index, None, args)
            nll40, acc40 = evaluate_points(model, model_config, tokenizer, points_by_task[task],
                                           C, task_index, flat, args)
            nllf, accf = evaluate_points(model, model_config, tokenizer, points_by_task[task],
                                         C, task_index, c_final, args)
            baselines[task] = {"no_intervention": {"nll": nll0, "acc": acc0},
                               "canonical_top40_layer9": {"nll": nll40, "acc": acc40},
                               "final_sparse_c": {"nll": nllf, "acc": accf}}
        with open(baselines_path, "w") as f:
            json.dump(baselines, f, indent=2)

    selection = {
        "sandbox": True,
        "note": "SANDBOX sparse-optimization head selection - NOT the repo-default head set.",
        "method": "Hu et al. 2025 (arXiv:2505.05145) section 3.1, LOTO-CV lambda selection",
        "model_name": args.model_name,
        "inject_layer": args.inject_layer,
        "chosen_lambda": chosen,
        "lambda_grid": args.lambdas,
        "accuracy_tolerance": args.accuracy_tolerance,
        "threshold": args.threshold,
        "n_selected": len(selected),
        "selected_heads": selected,
        "overlap_with_canonical_top40": {"n": len(overlap), "heads": overlap},
        "per_lambda_cv": {str(k): {kk: vv for kk, vv in v.items() if kk != "per_fold"}
                          for k, v in per_lambda.items()},
        "consistency_check": consistency,
    }
    with open(args.output_root / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)

    metadata = {
        "sandbox": True,
        "task_split_path": str(args.task_split_path),
        "task_split_key": "train_tasks",
        "tasks": tasks,
        "n_tasks": len(tasks),
        "query_split": "valid (cap 100, min 80; short tasks topped up from train)",
        "n_datapoints": {t: len(points_by_task[t]) for t in tasks},
        "datapoint_sources": {t: {s: sum(1 for p in points_by_task[t] if p["source_split"] == s)
                                  for s in ("valid", "train")} for t in tasks},
        "mean_acts_root": str(args.mean_acts_root),
        "loss": "raw -log p(full label), teacher-forced, injection ONCE at cue token only",
        "optimizer": {"name": "AdamW", "lr": args.lr, "batch_size": args.batch_size,
                      "micro_batch_size": args.micro_batch_size,
                      "max_epochs": args.max_epochs, "patience": args.patience,
                      "earlystop_frac": args.earlystop_frac, "init_c": args.init_c,
                      "clip": [0.0, 1.0]},
        "cv": "leave-one-task-out over the 20 train tasks",
        "seed": args.seed,
        "dtype": args.dtype,
    }
    with open(args.output_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    write_results(tasks, per_lambda, chosen, c_final.cpu(), selected, canonical, baselines, args)
    print(f"Selected {len(selected)} heads (c > {args.threshold}), "
          f"overlap with canonical top-40: {len(overlap)}. Artifacts in {args.output_root}")


def write_results(tasks, per_lambda, chosen, c_final, selected, canonical, baselines, args):
    import csv
    args.results_root.mkdir(parents=True, exist_ok=True)

    rows = [{"lambda": lam, **{k: v for k, v in per_lambda[lam].items() if k != "per_fold"},
             "chosen": lam == chosen} for lam in args.lambdas]
    with open(args.results_root / "lambda_cv_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["# SANDBOX sparse head selection - summary", "",
             "**Sandbox trial only - NOT the repo-default head set.**", "",
             f"Chosen lambda = **{chosen:g}** (largest within {args.accuracy_tolerance} mean LOTO "
             f"accuracy of best). Final selection: **{len(selected)} heads** (c > {args.threshold}).", "",
             "| lambda | mean LOTO acc | mean LOTO nll | mean n_active | chosen |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['lambda']:g} | {r['mean_acc']:.3f} | {r['mean_nll']:.3f} | "
                     f"{r['mean_active']:.1f} | {'YES' if r['chosen'] else ''} |")
    lines += ["", "## Selected heads (layer, head, coeff)", "",
              ", ".join(f"({l},{h},{s:.2f})" for l, h, s in selected), "",
              "## Per-task accuracy (final c vs baselines, same datapoints)", "",
              "| task | no interv. | canonical top-40 @L9 | final sparse c |", "|---|---|---|---|"]
    for t in tasks:
        b = baselines[t]
        lines.append(f"| {t} | {b['no_intervention']['acc']:.3f} | "
                     f"{b['canonical_top40_layer9']['acc']:.3f} | {b['final_sparse_c']['acc']:.3f} |")
    with open(args.results_root / "summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    plot_summary(per_lambda, chosen, c_final, args)


def plot_summary(per_lambda, chosen, c_final, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_heads = 16
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    heat = c_final.view(-1, n_heads).numpy()
    im = axes[0].imshow(heat.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xlabel("layer"); axes[0].set_ylabel("head")
    axes[0].set_title(f"final c (lambda={chosen:g})")
    plt.colorbar(im, ax=axes[0])

    lams = list(args.lambdas)
    accs = [per_lambda[l]["mean_acc"] for l in lams]
    acts = [per_lambda[l]["mean_active"] for l in lams]
    axes[1].plot(lams, accs, "o-")
    axes[1].axvline(chosen, color="r", ls="--", label=f"chosen {chosen:g}")
    axes[1].set_xscale("log"); axes[1].set_xlabel("lambda"); axes[1].set_ylabel("mean LOTO accuracy")
    axes[1].legend(); axes[1].set_title("LOTO accuracy vs lambda")
    axes[2].plot(lams, acts, "o-")
    axes[2].axvline(chosen, color="r", ls="--")
    axes[2].set_xscale("log"); axes[2].set_xlabel("lambda"); axes[2].set_ylabel(f"mean heads > {args.threshold}")
    axes[2].set_title("sparsity vs lambda")
    fig.suptitle("SANDBOX sparse-optimization head selection (GPT-J, 20 train tasks)")
    fig.tight_layout()
    out = args.results_root / "sparse_head_selection_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        args.tasks = args.tasks or ["present-past", "country-capital"]  # train tasks only
        args.max_queries, args.min_queries = 10, 5
        args.max_epochs = 2
        args.lambdas = [0.05]

    tasks = load_train_tasks(args)
    print(f"tasks ({len(tasks)}): {tasks}")

    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)

    # Build head contributions from the as-loaded (fp16) weights so the consistency check compares
    # against the same weight values the stored FVs were built from, before any dtype cast.
    C = build_contributions(tasks, args, model, model_config)

    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    # No gradient checkpointing: only the injected v requires grad, so autograd stores
    # activations for blocks > inject_layer only, and micro-batching keeps that small.
    model.eval()
    task_index = {t: i for i, t in enumerate(tasks)}

    consistency = None
    if args.mode in ("check", "reduce", "all"):
        consistency = consistency_check(tasks, args, C, model_config)
        if args.mode == "check":
            return

    print("building datapoints ...")
    points_by_task = {t: build_task_datapoints(t, args, tokenizer, model_config) for t in tqdm(tasks)}
    for t in tasks:
        n_v = sum(1 for p in points_by_task[t] if p["source_split"] == "valid")
        n_t = len(points_by_task[t]) - n_v
        print(f"  {t}: {len(points_by_task[t])} points (valid={n_v}, train top-up={n_t})")

    if args.mode == "smoke":
        # end-to-end: one tiny fold + reduce-style eval, no artifacts beyond fold files
        lam = args.lambdas[0]
        fold_task = tasks[-1]
        train_pool = [p for t in tasks if t != fold_task for p in points_by_task[t]]
        train_points, es_points = split_earlystop(train_pool, args.earlystop_frac, args.seed)
        c, history, _ = train_c(model, model_config, tokenizer, train_points, es_points, C,
                                task_index, lam, args, args.seed, desc="smoke")
        nll, acc = evaluate_points(model, model_config, tokenizer, points_by_task[fold_task],
                                   C, task_index, c, args)
        nll0, acc0 = evaluate_points(model, model_config, tokenizer, points_by_task[fold_task],
                                     C, task_index, None, args)
        assert history[-1]["train_nll"] < history[0]["train_nll"] + 1e-6 or len(history) == 1, \
            "smoke: train loss did not decrease"
        assert c.min() >= 0 and c.max() <= 1, "smoke: c escaped [0,1]"
        print(f"SMOKE OK: fold={fold_task} heldout nll={nll:.3f} acc={acc:.3f} "
              f"(no-interv nll={nll0:.3f} acc={acc0:.3f}); c in [{c.min():.3f},{c.max():.3f}]")
        return

    if args.mode in ("cv", "all"):
        run_cv(tasks, points_by_task, model, model_config, tokenizer, C, task_index, args)
    if args.mode in ("reduce", "all"):
        run_reduce(tasks, points_by_task, model, model_config, tokenizer, C, task_index, args, consistency)


if __name__ == "__main__":
    main()
