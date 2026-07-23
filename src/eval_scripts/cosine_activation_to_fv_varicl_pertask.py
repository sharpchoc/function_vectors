#!/usr/bin/env python
"""Per-task cosine(activation, varicl_top40 FV) over the (token position x layer) grid.

Companion to the pooled cosine_activation_to_task_fv.py study, but (a) the FV definition is the
canonical train_varicl_top40 set instead of the task-specific gptj_fv vectors, and (b) results are
reported PER TASK (default: the 4 tasks of the per-task ridge R^2 study) instead of averaged over
the 29-task manifest. Per cell (task, icl_example_index, token_role, layer) we compute the raw
per-prompt cosine(activation, task FV) and average over all prompts of both splits (130 train +
40 test). No centering; no model is loaded -- pure tensor math over the existing captures.

Outputs (per_task_cosine.csv, per-task diverging heatmaps + shared-scale panel, summary.json) go
under FV_FORMATION_DIR/cosine_activation_to_fv_varicl_top40_pertask/. The CSV feeds
plot_pertask_r2_best_lines.py via --value_column mean_cosine.
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
    LABEL_ROLES, QUERY_ICL_INDEX, FINAL_PROMPT_ROLE, load_function_vector,
)
from src.eval_scripts.cosine_activation_to_task_fv import load_task_roles
from src.eval_scripts.merge_fulldim_ridge_results import position_key, position_label
from src.utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+",
                   default=["antonym", "synonym", "prev_number_digits", "next_number_digits"])
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_varicl_top40")
    p.add_argument("--icl_activations_root_template", type=str,
                   default=str(ARTIFACTS_ROOT / "residual_activations" / "gptj_56tasks_170prompts_icl{icl}_3tokens"))
    p.add_argument("--query_activations_root", type=Path,
                   default=ARTIFACTS_ROOT / "residual_activations" / "gptj_56tasks_170prompts_4tokens")
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--output_dir", type=Path,
                   default=FV_FORMATION_DIR / "cosine_activation_to_fv_varicl_top40_pertask")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eps", type=float, default=1e-8)
    return p.parse_args()


def main():
    args = parse_args()
    fvs = {task: load_function_vector(args.fv_root, task).to(args.device) for task in args.tasks}

    rows = []  # one dict per (task, icl, role, layer)
    for icl in range(1, QUERY_ICL_INDEX + 1):
        if icl == QUERY_ICL_INDEX:
            activations_root = args.query_activations_root
            roles = LABEL_ROLES + [FINAL_PROMPT_ROLE]
        else:
            activations_root = Path(args.icl_activations_root_template.format(icl=icl))
            roles = list(LABEL_ROLES)

        for task in args.tasks:
            fv = fvs[task]
            fv_norm = fv.norm()
            role_data = load_task_roles(activations_root, task, args.splits)
            for role in roles:
                if role not in role_data:
                    raise ValueError(f"{task}: no rows for role {role} in {activations_root}")
                acts, _keys = role_data[role]          # [n, L, H] f32
                acts = acts.to(args.device)
                for layer in range(acts.shape[1]):
                    A = acts[:, layer, :]
                    cos = (A @ fv) / (A.norm(dim=1) * fv_norm + args.eps)
                    rows.append({
                        "task": task, "icl_example_index": icl, "token_role": role,
                        "layer": layer, "mean_cosine": float(cos.mean()),
                        "n_prompts": int(cos.numel()),
                    })
        print(f"icl{icl:02d}: processed {len(args.tasks)} tasks ({activations_root.name})")

    rows.sort(key=lambda r: (r["task"], position_key(r["icl_example_index"], r["token_role"]), r["layer"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "per_task_cosine.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "icl_example_index", "token_role", "layer",
                                          "mean_cosine", "n_prompts"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows).")

    # Grids: token position (31) x layer (29) per task, shared diverging scale centered at 0.
    pos_set = sorted({(r["icl_example_index"], r["token_role"]) for r in rows},
                     key=lambda ir: position_key(*ir))
    layer_set = sorted({r["layer"] for r in rows})
    pos_index = {p: i for i, p in enumerate(pos_set)}
    layer_index = {l: j for j, l in enumerate(layer_set)}
    pos_labels = [position_label(*p) for p in pos_set]
    grids = {task: np.full((len(pos_set), len(layer_set)), np.nan) for task in args.tasks}
    for r in rows:
        grids[r["task"]][pos_index[(r["icl_example_index"], r["token_role"])],
                         layer_index[r["layer"]]] = r["mean_cosine"]
    vmax = float(max(np.nanmax(np.abs(g)) for g in grids.values()))

    suptitle = "GPT-J raw cosine: activation vs train_varicl_top40 FV (per-prompt mean, train+test pooled)"
    summary = {"fv_root": str(args.fv_root), "splits": list(args.splits),
               "metric": "per-prompt raw cosine(activation, train_varicl_top40 task FV), "
                         "mean over all prompts of the pooled splits",
               "n_cells_per_task": len(pos_set) * len(layer_set),
               "color_scale_vmax": vmax, "tasks": {}}

    def draw(ax, task, tick_fontsize=6, label_step=1):
        im = ax.imshow(grids[task], aspect="auto", cmap="coolwarm", interpolation="nearest",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(0, len(layer_set), label_step))
        ax.set_xticklabels(layer_set[::label_step], fontsize=tick_fontsize)
        ax.set_yticks(range(len(pos_labels)))
        ax.set_yticklabels(pos_labels, fontsize=tick_fontsize)
        return im

    for task in args.tasks:
        fig, ax = plt.subplots(figsize=(max(8, len(layer_set) * 0.32), max(5, len(pos_set) * 0.3)))
        im = draw(ax, task)
        ax.set_xlabel("layer (0 = embedding)")
        ax.set_ylabel("token position (icl/role)")
        ax.set_title(f"{task}: mean cosine(activation, varicl_top40 FV)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025)
        cbar.set_label("mean cosine", fontsize=8)
        fig.suptitle(suptitle, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        out_png = args.output_dir / f"cosine_heatmap_{task}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        g = grids[task]
        bi, bj = np.unravel_index(np.nanargmax(g), g.shape)
        summary["tasks"][task] = {
            "best_cosine": float(g[bi, bj]),
            "best_cell": {"position": pos_labels[bi], "layer": int(layer_set[bj])},
            "min_cosine": float(np.nanmin(g)),
            "median_cosine": float(np.nanmedian(g)),
        }
        s = summary["tasks"][task]
        print(f"{task}: best cosine {s['best_cosine']:.3f} at {s['best_cell']['position']} "
              f"L{s['best_cell']['layer']} | median {s['median_cosine']:.3f} -> {out_png.name}")

    # Combined panel, shared scale.
    n = len(args.tasks)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(ncols * max(7, len(layer_set) * 0.3),
                                      nrows * max(4.5, len(pos_set) * 0.16)))
    for k, task in enumerate(args.tasks):
        ax = axes[k // ncols][k % ncols]
        im = draw(ax, task, tick_fontsize=5, label_step=2)
        ax.set_title(task, fontsize=10)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"{suptitle}\nper-task mean cosine(activation, varicl_top40 FV) — shared scale",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=[a for row in axes for a in row], fraction=0.02)
    cbar.set_label("mean cosine", fontsize=8)
    panel_png = args.output_dir / "cosine_heatmap_panel.png"
    fig.savefig(panel_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {panel_png}")

    best_task = max(summary["tasks"], key=lambda t: summary["tasks"][t]["best_cosine"])
    summary["best_overall"] = {"task": best_task, **summary["tasks"][best_task]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
