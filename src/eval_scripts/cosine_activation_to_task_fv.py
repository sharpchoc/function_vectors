#!/usr/bin/env python
"""Stage 2: cosine similarity between residual activations and the task-specific FV, as a
heatmap over (token position x layer) -- the same grid as the full-dim ridge MSE study, but the
per-cell metric is raw cosine alignment instead of regression MSE.

For each task, its OWN task-specific function vector (artifacts/gptj_fv/<task>/...) is compared to
every captured residual activation: per cell (icl_example_index, token_role, layer) we compute the
per-example cosine(activation, task_FV) and average over examples (raw, no centering). Cell values
are then averaged over all 29 tasks.

Two heatmaps are produced using the per-prompt correctness from Stage 1
(compute_capture_prompt_correctness.py):
  * combined_cosine_heatmap_all.png      -- over all prompts
  * combined_cosine_heatmap_correct.png  -- over only prompts the model answered correctly
(correctness is one bit per (split, query_source_index), shared across all token positions of a
prompt). No model is loaded -- pure tensor math.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    LABEL_ROLES, QUERY_ICL_INDEX, FINAL_PROMPT_ROLE,
    torch_load_trusted, load_function_vector, load_json,
)
from src.eval_scripts.merge_fulldim_ridge_results import (
    ROLE_ORDER, position_key, position_label,
)
from src.utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    p.add_argument("--fv_root", type=Path, default=ARTIFACTS_ROOT / "gptj_fv",
                   help="Root holding each task's own task-specific FV (<task>/<task>_function_vector.pt).")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations" / "gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations" / "gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--output_dir", type=Path, default=FV_FORMATION_DIR / "activation_to_fv_decoding/cosine/activation_to_task_fv")
    p.add_argument("--correctness_dir", type=Path, default=None,
                   help="Dir with <task>.json from Stage 1 (default: <output_dir>/correctness).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eps", type=float, default=1e-8)
    return p.parse_args()


def load_split_roles_with_meta(activations_root, task, split):
    """Single pass over a (task, split): return {token_role: (acts[n,L,H] f32, [query_source_index])}.

    Each activation directory was captured for a single target ICL index, so a row's token_role
    uniquely identifies its per-prompt token position (label roles all carry that one icl index;
    last_prompt_token carries icl_example_index=None). Bucketing by role is therefore exact.
    """
    split_dir = activations_root / task / split
    index = load_json(split_dir / "index.json")
    buckets = {}  # role -> (list[Tensor], list[int])
    for shard in index["shards"]:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = split_dir / shard_path
        elif not shard_path.exists():
            # index.json may hold absolute paths from before the results->artifacts reorg.
            shard_path = split_dir / shard_path.name
        data = torch_load_trusted(shard_path, map_location="cpu")
        acts = data["activations"]
        meta = data["metadata"]
        for i, m in enumerate(meta):
            role = m.get("token_role")
            if role is None:
                continue
            a, q = buckets.setdefault(role, ([], []))
            a.append(acts[i])
            q.append(int(m["query_source_index"]))
    return {role: (torch.stack(a, dim=0).float(), q) for role, (a, q) in buckets.items()}


def load_task_roles(activations_root, task, splits):
    """Pool splits: {role: (acts[n,L,H] f32, [(split, query_source_index)])}."""
    pooled = {}
    for split in splits:
        per_role = load_split_roles_with_meta(activations_root, task, split)
        for role, (acts, qsi) in per_role.items():
            a_list, key_list = pooled.setdefault(role, ([], []))
            a_list.append(acts)
            key_list.extend((split, q) for q in qsi)
    return {role: (torch.cat(a, dim=0), keys) for role, (a, keys) in pooled.items()}


def load_correct_map(correctness_dir, task):
    """{(split, query_source_index): bool} from Stage 1, or None if absent."""
    path = correctness_dir / f"{task}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {(r["split"], int(r["query_source_index"])): bool(r["correct"]) for r in data["records"]}


def render_heatmap_diverging(pos_labels, layers, grid, title, out_path, vmax):
    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.32), max(5, len(pos_labels) * 0.3)))
    im = ax.imshow(np.array(grid, dtype=float), aspect="auto", cmap="coolwarm",
                   interpolation="nearest", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=6)
    ax.set_yticks(range(len(pos_labels)))
    ax.set_yticklabels(pos_labels, fontsize=6)
    ax.set_xlabel("layer (0 = embedding)")
    ax.set_ylabel("token position (icl/role)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label("mean cosine(activation, task FV)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    correctness_dir = args.correctness_dir or (args.output_dir / "correctness")
    manifest = load_json(args.task_manifest)
    tasks = list(manifest["train_tasks"]) + list(manifest["test_tasks"])

    # cell -> per-task lists of (mean_all, n_all, mean_correct, n_correct)
    cell_rows = {}  # (icl, role, layer) -> dict of accumulators
    n_layers = None

    for icl in range(1, QUERY_ICL_INDEX + 1):
        if icl == QUERY_ICL_INDEX:
            activations_root = args.query_activations_root
            roles = LABEL_ROLES + [FINAL_PROMPT_ROLE]
        else:
            activations_root = Path(args.icl_activations_root_template.format(icl=icl))
            roles = list(LABEL_ROLES)

        for task in tasks:
            fv = load_function_vector(args.fv_root, task).to(args.device)  # [H] f32
            fv_norm = fv.norm()
            correct_map = load_correct_map(correctness_dir, task)
            role_data = load_task_roles(activations_root, task, args.splits)
            for role in roles:
                if role not in role_data:
                    raise ValueError(f"{task}: no rows for role {role} in {activations_root}")
                acts, keys = role_data[role]          # acts [n, L, H], keys list of (split, qsi)
                acts = acts.to(args.device)
                if n_layers is None:
                    n_layers = acts.shape[1]
                if correct_map is not None:
                    correct_mask = torch.tensor([correct_map[k] for k in keys], device=args.device)
                else:
                    correct_mask = None
                for layer in range(acts.shape[1]):
                    A = acts[:, layer, :]                              # [n, H]
                    cos = (A @ fv) / (A.norm(dim=1) * fv_norm + args.eps)   # [n], raw signed
                    mean_all = float(cos.mean())
                    n_all = int(cos.numel())
                    if correct_mask is not None and bool(correct_mask.any()):
                        mean_correct = float(cos[correct_mask].mean())
                        n_correct = int(correct_mask.sum())
                    else:
                        mean_correct, n_correct = float("nan"), 0
                    acc = cell_rows.setdefault((icl, role, layer),
                                               {"all": [], "n_all": 0, "correct": [], "n_correct": 0})
                    acc["all"].append(mean_all)
                    acc["n_all"] += n_all
                    acc["n_correct"] += n_correct
                    if n_correct > 0:
                        acc["correct"].append(mean_correct)
        print(f"icl{icl:02d}: processed {len(tasks)} tasks ({activations_root.name})")

    # Aggregate over tasks (mean of per-task means).
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_cell = []
    for (icl, role, layer), acc in cell_rows.items():
        per_cell.append({
            "icl_example_index": icl, "token_role": role, "layer": layer,
            "mean_cosine_all": float(np.mean(acc["all"])),
            "n_tasks_all": len(acc["all"]), "n_examples_all": acc["n_all"],
            "mean_cosine_correct": float(np.mean(acc["correct"])) if acc["correct"] else float("nan"),
            "n_tasks_correct": len(acc["correct"]), "n_examples_correct": acc["n_correct"],
        })
    per_cell.sort(key=lambda r: (position_key(r["icl_example_index"], r["token_role"]), r["layer"]))

    combined_csv = args.output_dir / "combined_metrics.csv"
    fields = ["icl_example_index", "token_role", "layer", "mean_cosine_all", "n_tasks_all",
              "n_examples_all", "mean_cosine_correct", "n_tasks_correct", "n_examples_correct"]
    with open(combined_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_cell)
    print(f"Wrote {combined_csv} ({len(per_cell)} cells).")

    # Build grids.
    pos_set = sorted({(r["icl_example_index"], r["token_role"]) for r in per_cell},
                     key=lambda ir: position_key(*ir))
    layer_set = sorted({r["layer"] for r in per_cell})
    pos_index = {p: i for i, p in enumerate(pos_set)}
    layer_index = {l: j for j, l in enumerate(layer_set)}
    grid_all = np.full((len(pos_set), len(layer_set)), np.nan)
    grid_cor = np.full((len(pos_set), len(layer_set)), np.nan)
    for r in per_cell:
        i = pos_index[(r["icl_example_index"], r["token_role"])]
        j = layer_index[r["layer"]]
        grid_all[i, j] = r["mean_cosine_all"]
        grid_cor[i, j] = r["mean_cosine_correct"]

    pos_labels = [position_label(*p) for p in pos_set]
    vmax = float(np.nanmax(np.abs(np.concatenate([grid_all.ravel(), grid_cor.ravel()]))))
    render_heatmap_diverging(pos_labels, layer_set, grid_all,
                             "cosine(activation, task FV) -- all prompts",
                             args.output_dir / "combined_cosine_heatmap_all.png", vmax)
    render_heatmap_diverging(pos_labels, layer_set, grid_cor,
                             "cosine(activation, task FV) -- correct prompts only",
                             args.output_dir / "combined_cosine_heatmap_correct.png", vmax)
    print("Wrote combined_cosine_heatmap_{all,correct}.png (shared color scale +/-%.3f)" % vmax)

    # Summary: best/worst cell + layer profile for each variant.
    def summarize(key):
        finite = [r for r in per_cell if np.isfinite(r[key])]
        best = max(finite, key=lambda r: r[key])
        worst = min(finite, key=lambda r: r[key])
        prof = {}
        for layer in layer_set:
            vals = [r[key] for r in finite if r["layer"] == layer]
            prof[layer] = float(np.mean(vals)) if vals else float("nan")
        return {
            "best_cell": {"token_position": position_label(best["icl_example_index"], best["token_role"]),
                          "layer": best["layer"], key: best[key]},
            "worst_cell": {"token_position": position_label(worst["icl_example_index"], worst["token_role"]),
                           "layer": worst["layer"], key: worst[key]},
            "layer_profile_mean_over_positions": prof,
        }

    summary = {
        "n_cells": len(per_cell), "n_token_positions": len(pos_set), "n_layers": len(layer_set),
        "n_tasks": len(tasks), "metric": "per-example mean raw cosine(activation, task-specific FV)",
        "color_scale_vmax": vmax,
        "all_prompts": summarize("mean_cosine_all"),
        "correct_prompts": summarize("mean_cosine_correct"),
    }
    (args.output_dir / "combined_summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote combined_summary.json")
    print("  all  best:", summary["all_prompts"]["best_cell"])
    print("  corr best:", summary["correct_prompts"]["best_cell"])


if __name__ == "__main__":
    main()
