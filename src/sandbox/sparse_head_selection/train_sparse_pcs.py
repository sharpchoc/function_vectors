#!/usr/bin/env python
"""SANDBOX: sparse-optimization selection over the 83 FV-stack PC directions (GPT-J).

PC analog of train_sparse_heads.py (Hu et al. 2025, arXiv:2505.05145 section 3.1): learn
c in [0,1]^83 over the uncentered PCs of the pooled 20-train-task fixed10 sparse23
per-prompt FV stack (see build_fv_pc_basis.py). Per task t the injected vector is

    v_t(c) = sum_i c_i * (v_t . u_i) * u_i      (v_t = fixed10-mean sparse23 FV)

added ONCE at the cue token of a zero-shot "Q: x\nA:" prompt at the output of block
--inject_layer (default 9). Loss = raw -log p(full label) (teacher-forced) + lambda * sum(c);
lambda chosen by leave-one-task-out CV over the 20 train tasks (largest lambda within
--accuracy_tolerance of the best mean LOTO full-label accuracy); c clamped to [0,1].

Reuses the generic machinery of train_sparse_heads.py (datapoints, injection forward,
training loop, fold layout, lambda selection). Fold run seeds depend on the lambda VALUE
(not its grid position) so the sweep can be sharded across pods with disjoint --lambdas.

SANDBOX per the 2026-08-06 decision - nothing here is repo standard.
"""
import argparse
import json
import sys
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

from src.sandbox.sparse_head_selection.train_sparse_heads import (
    batch_label_logprobs,
    build_task_datapoints,
    evaluate_points,
    fold_path,
    load_train_tasks,
    make_batches,
    select_lambda,
    split_earlystop,
    train_c,
)
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from src.utils.paths import ARTIFACTS_ROOT, RESULTS_ROOT

DEFAULT_ARTIFACT_ROOT = ARTIFACTS_ROOT / "sandbox" / "sparse_pc_selection"
DEFAULT_RESULTS_ROOT = RESULTS_ROOT / "sandbox" / "sparse_pc_selection"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["check", "smoke", "cv", "reduce", "all"], default="all")
    p.add_argument("--basis_path", type=Path, default=DEFAULT_ARTIFACT_ROOT / "pc_basis_83.pt")
    p.add_argument("--task_split_path", type=Path,
                   default=REPO_ROOT / "task_splits/abstractive_train_test_tasks_29.json")
    p.add_argument("--root_data_dir", type=str, default=str(REPO_ROOT / "dataset_files"))
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16",
                   help="Model forward dtype; c and the injected v stay fp32.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3,
                   help="Dataset valid/test split arg (must match the head run: 0.3).")
    p.add_argument("--inject_layer", type=int, default=9)
    p.add_argument("--max_queries", type=int, default=100)
    p.add_argument("--min_queries", type=int, default=80)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
                   help="Subset per pod for sharding; reduce must be run with the FULL grid.")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--micro_batch_size", type=int, default=32,
                   help="Gradient-accumulation micro-batch (raise to fill big-memory GPUs).")
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--earlystop_frac", type=float, default=0.1)
    p.add_argument("--init_c", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.2,
                   help="Significant-PC threshold on c (paper's 0.2).")
    p.add_argument("--accuracy_tolerance", type=float, default=0.01)
    p.add_argument("--topk", type=int, nargs="+", default=[1, 2, 3, 5, 8, 12, 20, 30, 50, 83],
                   help="k values for the post-hoc top-k-PCs accuracy curve (reduce mode).")
    p.add_argument("--tasks", nargs="+", default=None, help="Override task list (smoke/debug only).")
    p.add_argument("--output_root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    p.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return p.parse_args()


# ---------------------------------------------------------------------------
# PC contributions
# ---------------------------------------------------------------------------

def load_basis(args, tasks):
    basis = torch.load(args.basis_path, map_location="cpu", weights_only=False)
    missing = [t for t in tasks if t not in basis["v_means"]]
    assert not missing, f"basis {args.basis_path} lacks mean FVs for tasks: {missing}"
    return basis


def build_pc_contributions(basis, tasks, device):
    """C[t, i, :] = (v_t . u_i) * u_i, fp32 on device. Shape (T, n_pcs, resid)."""
    U = basis["U"]  # (n_pcs, resid) fp64, orthonormal rows
    C = torch.zeros(len(tasks), U.shape[0], U.shape[1], dtype=torch.float32, device=device)
    for t, task in enumerate(tasks):
        v = basis["v_means"][task]  # fp64
        C[t] = ((U @ v).unsqueeze(1) * U).float().to(device)
    return C


def consistency_check_pc(basis, tasks, C):
    """v(c=1) = sum_i C[t,i] must equal the fp64 orthogonal projection U^T U v_t for every
    task (rel err <= 1e-4). Mismatch = hard stop (user adjudicates data discrepancies)."""
    U = basis["U"]
    worst = {"task": None, "rel_err": 0.0, "retained_energy": None}
    retained = {}
    for t, task in enumerate(tasks):
        v = basis["v_means"][task]
        proj = (U.T @ (U @ v))  # fp64 reference
        built = C[t].sum(dim=0).double().cpu()
        rel_err = (built - proj).norm().item() / max(proj.norm().item(), 1e-12)
        retained[task] = float(proj.norm().item() ** 2 / v.norm().item() ** 2)
        if rel_err > worst["rel_err"]:
            worst = {"task": task, "rel_err": rel_err, "retained_energy": retained[task]}
        if rel_err > 1e-4:
            raise RuntimeError(
                f"PC CONSISTENCY CHECK FAILED for {task}: rel_err={rel_err:.4e} between "
                f"c=1 contribution sum and the fp64 subspace projection. HARD STOP - "
                f"report to user for adjudication.")
    print(f"PC consistency check PASSED on {len(tasks)} tasks. "
          f"Worst rel_err={worst['rel_err']:.3e} ({worst['task']}); "
          f"FV energy retained by subspace: min={min(retained.values()):.4f}")
    return {"worst_rel_err": worst["rel_err"], "worst_task": worst["task"],
            "fv_energy_retained": retained}


def evaluate_points_v(model, model_config, tokenizer, points, v_by_task, args):
    """Like evaluate_points but injecting a fixed per-task vector (fp32 device tensors)."""
    total_nll, n_correct = 0.0, 0
    with torch.no_grad():
        for batch in make_batches(points, args.batch_size):
            v = torch.stack([v_by_task[b["task"]] for b in batch])
            nll, accs = batch_label_logprobs(model, model_config, tokenizer, batch, v=v,
                                             inject_layer=args.inject_layer)
            total_nll += nll.sum().item()
            n_correct += sum(accs)
    return total_nll / len(points), n_correct / len(points)


# ---------------------------------------------------------------------------
# CV (lambda-value-keyed seeds so --lambdas sharding is reproducible)
# ---------------------------------------------------------------------------

def run_seed_for(args, lam, fold_index):
    return args.seed + 100000 + int(round(lam * 1e4)) * 100 + fold_index


def run_cv(tasks, points_by_task, model, model_config, tokenizer, C, task_index, args):
    (args.output_root / "fold_results").mkdir(parents=True, exist_ok=True)
    for lam in args.lambdas:
        for fi, fold_task in enumerate(tasks):
            out = fold_path(args, lam, fold_task)
            if out.exists():
                print(f"skip existing {out.name}")
                continue
            train_pool = [p for t in tasks if t != fold_task for p in points_by_task[t]]
            run_seed = run_seed_for(args, lam, fi)
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


# ---------------------------------------------------------------------------
# Reduce: final retrain + baselines + top-k curve + reporting
# ---------------------------------------------------------------------------

def run_reduce(tasks, points_by_task, model, model_config, tokenizer, C, task_index, args,
               basis, consistency):
    chosen, per_lambda = select_lambda(tasks, args)
    print(f"Chosen lambda={chosen:g} "
          f"(rule: largest within {args.accuracy_tolerance} of best mean LOTO accuracy)")

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
                                            C, task_index, chosen, args, run_seed,
                                            desc=f"FINAL lam={chosen:g}")
        torch.save({"c": c_final.cpu(), "lambda": chosen, "history": final_history,
                    "run_seed": run_seed}, final_c_path)

    n_pcs = C.shape[1]
    order = torch.argsort(c_final, descending=True).tolist()
    selected = [(i, round(float(c_final[i].item()), 6)) for i in order
                if c_final[i].item() > args.threshold]

    # Per-task references on the same datapoints.
    baselines_path = args.output_root / "baselines.json"
    if baselines_path.exists():
        with open(baselines_path) as f:
            baselines = json.load(f)
    else:
        v_full = {t: basis["v_means"][t].float().to(C.device) for t in tasks}
        ones = torch.ones(n_pcs, device=C.device)
        baselines = {}
        for task in tqdm(tasks, desc="baselines"):
            pts = points_by_task[task]
            nll0, acc0 = evaluate_points(model, model_config, tokenizer, pts, C, task_index, None, args)
            nllf, accf = evaluate_points_v(model, model_config, tokenizer, pts, v_full, args)
            nll1, acc1 = evaluate_points(model, model_config, tokenizer, pts, C, task_index, ones, args)
            nlls, accs = evaluate_points(model, model_config, tokenizer, pts, C, task_index, c_final, args)
            baselines[task] = {
                "no_intervention": {"nll": nll0, "acc": acc0},
                "full_fv_fixed10": {"nll": nllf, "acc": accf},
                "proj83_c1": {"nll": nll1, "acc": acc1},
                "final_sparse_c": {"nll": nlls, "acc": accs},
            }
        with open(baselines_path, "w") as f:
            json.dump(baselines, f, indent=2)

    # Post-hoc top-k PC curve (by final c, weighted and unweighted).
    topk_path = args.output_root / "topk_curve.json"
    if topk_path.exists():
        with open(topk_path) as f:
            topk_curve = json.load(f)
    else:
        all_points = [p for t in tasks for p in points_by_task[t]]
        topk_curve = []
        for k in args.topk:
            k = min(k, n_pcs)
            mask = torch.zeros(n_pcs, device=C.device)
            mask[order[:k]] = 1.0
            row = {"k": k, "pc_indices": [int(i) for i in order[:k]]}
            for name, c_eval in (("weighted", c_final * mask), ("unweighted", mask)):
                nll, acc = evaluate_points(model, model_config, tokenizer, all_points,
                                           C, task_index, c_eval, args)
                row[f"{name}_nll"], row[f"{name}_acc"] = nll, acc
            topk_curve.append(row)
            print(f"top-k={k:3d}: weighted acc={row['weighted_acc']:.3f} "
                  f"unweighted acc={row['unweighted_acc']:.3f}", flush=True)
        with open(topk_path, "w") as f:
            json.dump(topk_curve, f, indent=2)

    selection = {
        "sandbox": True,
        "note": "SANDBOX sparse-optimization PC selection over the sparse23 per-prompt FV "
                "stack basis - NOT repo standard.",
        "method": "Hu et al. 2025 (arXiv:2505.05145) section 3.1 adapted to PC coefficients, "
                  "LOTO-CV lambda selection",
        "basis_path": str(args.basis_path),
        "fv_definition": "sparse23 heads, fixed10 capture mean (matches the PCA population)",
        "model_name": args.model_name,
        "inject_layer": args.inject_layer,
        "chosen_lambda": chosen,
        "lambda_grid": args.lambdas,
        "accuracy_tolerance": args.accuracy_tolerance,
        "threshold": args.threshold,
        "n_pcs": n_pcs,
        "n_selected": len(selected),
        "n_near_one": int((c_final > 0.8).sum().item()),
        "selected_pcs": selected,
        "per_lambda_cv": {str(k): {kk: vv for kk, vv in v.items() if kk != "per_fold"}
                          for k, v in per_lambda.items()},
        "consistency_check": consistency,
    }
    with open(args.output_root / "selection.json", "w") as f:
        json.dump(selection, f, indent=2)

    metadata = {
        "sandbox": True,
        "task_split_path": str(args.task_split_path),
        "tasks": tasks,
        "n_tasks": len(tasks),
        "query_split": "valid (cap 100, min 80; short tasks topped up from train)",
        "n_datapoints": {t: len(points_by_task[t]) for t in tasks},
        "datapoint_sources": {t: {s: sum(1 for p in points_by_task[t] if p["source_split"] == s)
                                  for s in ("valid", "train")} for t in tasks},
        "loss": "raw -log p(full label), teacher-forced, injection ONCE at cue token only",
        "optimizer": {"name": "AdamW", "lr": args.lr, "batch_size": args.batch_size,
                      "micro_batch_size": args.micro_batch_size,
                      "max_epochs": args.max_epochs, "patience": args.patience,
                      "earlystop_frac": args.earlystop_frac, "init_c": args.init_c,
                      "clip": [0.0, 1.0]},
        "cv": "leave-one-task-out over the 20 train tasks; fold seeds keyed to lambda VALUE",
        "seed": args.seed,
        "dtype": args.dtype,
    }
    with open(args.output_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    write_results(tasks, per_lambda, chosen, c_final.cpu(), selected, baselines, topk_curve, args)
    print(f"Selected {len(selected)} PCs (c > {args.threshold}), "
          f"{selection['n_near_one']} near 1.0. Artifacts in {args.output_root}")


def write_results(tasks, per_lambda, chosen, c_final, selected, baselines, topk_curve, args):
    import csv
    args.results_root.mkdir(parents=True, exist_ok=True)

    rows = [{"lambda": lam, **{k: v for k, v in per_lambda[lam].items() if k != "per_fold"},
             "chosen": lam == chosen} for lam in args.lambdas]
    with open(args.results_root / "lambda_cv_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(args.results_root / "loto_per_fold.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "fold_task", "acc", "nll", "n_active"])
        for lam in args.lambdas:
            for r in per_lambda[lam]["per_fold"]:
                w.writerow([lam, r["fold_task"], f"{r['acc']:.4f}", f"{r['nll']:.4f}", r["n_active"]])

    lines = ["# SANDBOX sparse PC selection - summary", "",
             "**Sandbox trial only - builds on the SANDBOX sparse23 head set; NOT repo standard.**",
             "",
             "Units = the 83 uncentered PCs (>=90% pooled variance) of the 20 train tasks' "
             "fixed10 sparse23 per-prompt FV stack; task FV = fixed10 capture mean "
             "(NOT the canonical varicl mean - do not compare absolute numbers to the "
             "sparse_head_selection tables without noting this).", "",
             f"Chosen lambda = **{chosen:g}** (largest within {args.accuracy_tolerance} mean LOTO "
             f"accuracy of best). Final selection: **{len(selected)} PCs** (c > {args.threshold}).", "",
             "| lambda | mean LOTO acc | mean LOTO nll | mean n_active | chosen |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['lambda']:g} | {r['mean_acc']:.3f} | {r['mean_nll']:.3f} | "
                     f"{r['mean_active']:.1f} | {'YES' if r['chosen'] else ''} |")
    lines += ["", "## Selected PCs (index, coeff), by coeff desc", "",
              ", ".join(f"({i},{s:.2f})" for i, s in selected), "",
              "## Top-k PC curve (pooled over all datapoints)", "",
              "| k | weighted acc | unweighted acc |", "|---|---|---|"]
    for r in topk_curve:
        lines.append(f"| {r['k']} | {r['weighted_acc']:.3f} | {r['unweighted_acc']:.3f} |")
    lines += ["", "## Per-task accuracy (same datapoints)", "",
              "| task | no interv. | full FV (fixed10) | 83-PC proj (c=1) | final sparse c |",
              "|---|---|---|---|---|"]
    for t in tasks:
        b = baselines[t]
        lines.append(f"| {t} | {b['no_intervention']['acc']:.3f} | {b['full_fv_fixed10']['acc']:.3f} | "
                     f"{b['proj83_c1']['acc']:.3f} | {b['final_sparse_c']['acc']:.3f} |")
    with open(args.results_root / "summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    plot_summary(per_lambda, chosen, c_final, baselines, topk_curve, tasks, args)


def plot_summary(per_lambda, chosen, c_final, baselines, topk_curve, tasks, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    ax.bar(np.arange(len(c_final)), c_final.numpy(), width=0.9)
    ax.axhline(args.threshold, color="r", ls="--", lw=0.8, label=f"threshold {args.threshold}")
    ax.set_xlabel("PC index (variance order)"); ax.set_ylabel("final c")
    ax.set_title(f"final c per PC (lambda={chosen:g})"); ax.legend(fontsize=8)

    lams = list(args.lambdas)
    ax = axes[0, 1]
    ax.plot(lams, [per_lambda[l]["mean_acc"] for l in lams], "o-")
    ax.axvline(chosen, color="r", ls="--", label=f"chosen {chosen:g}")
    ax.set_xscale("log"); ax.set_xlabel("lambda"); ax.set_ylabel("mean LOTO accuracy")
    ax.legend(fontsize=8); ax.set_title("LOTO accuracy vs lambda")

    ax = axes[1, 0]
    ax.plot(lams, [per_lambda[l]["mean_active"] for l in lams], "o-")
    ax.axvline(chosen, color="r", ls="--")
    ax.set_xscale("log"); ax.set_xlabel("lambda")
    ax.set_ylabel(f"mean PCs > {args.threshold}"); ax.set_title("sparsity vs lambda")

    ax = axes[1, 1]
    ks = [r["k"] for r in topk_curve]
    ax.plot(ks, [r["weighted_acc"] for r in topk_curve], "o-", label="top-k, weighted c")
    ax.plot(ks, [r["unweighted_acc"] for r in topk_curve], "s-", label="top-k, unweighted (c=1)")
    mean_acc = {key: float(np.mean([baselines[t][key]["acc"] for t in tasks]))
                for key in ("proj83_c1", "full_fv_fixed10", "no_intervention")}
    ax.axhline(mean_acc["proj83_c1"], color="gray", ls="--", lw=0.8, label="83-PC proj (c=1)")
    ax.axhline(mean_acc["full_fv_fixed10"], color="k", ls=":", lw=0.8, label="full FV (fixed10)")
    ax.axhline(mean_acc["no_intervention"], color="r", ls=":", lw=0.8, label="no intervention")
    ax.set_xlabel("k (top PCs by final c)"); ax.set_ylabel("pooled accuracy")
    ax.legend(fontsize=8); ax.set_title("accuracy vs #PCs kept (train-task datapoints)")

    fig.suptitle("SANDBOX sparse PC selection over the sparse23 per-prompt FV basis "
                 "(GPT-J, 20 train tasks, zero-shot @L9)")
    fig.tight_layout()
    out = args.results_root / "sparse_pc_selection_summary.png"
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

    basis = load_basis(args, tasks)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    C = build_pc_contributions(basis, tasks, model.device)

    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    # No gradient checkpointing (incompatible with grad-carrying injected v, 2026-08-06 lesson);
    # micro-batch accumulation keeps stored activations (blocks > inject_layer only) small.
    model.eval()
    task_index = {t: i for i, t in enumerate(tasks)}

    consistency = None
    if args.mode in ("check", "reduce", "all"):
        consistency = consistency_check_pc(basis, tasks, C)
        if args.mode == "check":
            return

    print("building datapoints ...")
    points_by_task = {t: build_task_datapoints(t, args, tokenizer, model_config) for t in tqdm(tasks)}
    for t in tasks:
        n_v = sum(1 for p in points_by_task[t] if p["source_split"] == "valid")
        print(f"  {t}: {len(points_by_task[t])} points (valid={n_v}, "
              f"train top-up={len(points_by_task[t]) - n_v})")

    if args.mode == "smoke":
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
        run_reduce(tasks, points_by_task, model, model_config, tokenizer, C, task_index, args,
                   basis, consistency)


if __name__ == "__main__":
    main()
