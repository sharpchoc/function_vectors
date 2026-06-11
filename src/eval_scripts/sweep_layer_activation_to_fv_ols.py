#!/usr/bin/env python
"""Layer sweep for the activation->function-vector joint-PCA regression (OLS), full-dim MSE.

Picks the residual-stream layer for the joint-PCA regression by sweeping every layer and
reporting held-out test-task MSE in the original 4096-d function-vector space.

For each (ICL example, token role, layer):
  * fit a fresh activation PCA at that layer on the train tasks (act_k components),
  * reuse the shared (layer-independent) function-vector PCA fit inline on the train-task FVs,
  * build the joint feature/target space [activation PCs (act_k), FV PCs (fv_k)],
  * fit OLS on the train tasks,
  * reconstruct the predicted FV back to 4096-d from the FV-PC half of the output and score
    held-out test-task MSE there (NOT in the reduced joint-PCA space).

The activation-side k and FV-side k are decoupled: fv_k = min(k, --fv_k_cap). The FV PCA is
fit on the train-task FVs, so it has at most (#train tasks - 1) directions; --fv_k_cap (default
16) caps it further. The grey dashed "FV-PC reconstruction floor" is the MSE a perfect FV-PC
prediction still incurs (variance outside the top-fv_k FV PCs); it is layer-independent, so it
shows as a single horizontal line.

Produces one plot panel per token position; within each panel one test-MSE-vs-layer series
per ICL example. Defaults to ICL examples 1-5.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


DEFAULT_TOKEN_ROLES = [
    "pre_label_token",
    "first_label_token",
    "last_label_token",
]
TOKEN_TITLES = {
    "pre_label_token": "Pre-label token",
    "first_label_token": "First label token",
    "last_label_token": "Last label token",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep layers for the OLS joint-PCA activation->FV regression (full-dim MSE).")
    parser.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv"))
    parser.add_argument(
        "--activations_root_template",
        type=str,
        default="results/residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results/layer_sweep_activation_to_fv_ols"))
    parser.add_argument("--icl_example_indices", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--token_roles", nargs="+", default=DEFAULT_TOKEN_ROLES)
    parser.add_argument("--k", type=int, default=16, help="Activation-side PCA components.")
    parser.add_argument("--fv_k_cap", type=int, default=16, help="Maximum FV-side PCs (target rank cap); fv_k = min(k, cap).")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Layers to sweep (default: all available).")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prev_layer", type=int, default=11, help="Previously-chosen layer, marked on the plot.")
    parser.add_argument("--plot_min_layer", type=int, default=1,
                        help="Lowest layer shown on the plot (default 1 drops the embedding layer 0, whose MSE is huge).")
    parser.add_argument("--train_tasks", nargs="+", default=None, help="Optional override for smoke tests.")
    parser.add_argument("--test_tasks", nargs="+", default=None, help="Optional override for smoke tests.")
    return parser.parse_args()


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_task_manifest(args):
    manifest = load_json(args.task_manifest)
    train_tasks = args.train_tasks if args.train_tasks is not None else manifest["train_tasks"]
    test_tasks = args.test_tasks if args.test_tasks is not None else manifest["test_tasks"]
    overlap = sorted(set(train_tasks).intersection(test_tasks))
    if overlap:
        raise ValueError(f"Tasks cannot be both train and test: {overlap}")
    return manifest, list(train_tasks), list(test_tasks)


def load_function_vector(fv_root, task):
    fv_path = fv_root / task / f"{task}_function_vector.pt"
    if not fv_path.exists():
        raise FileNotFoundError(fv_path)
    data = torch_load_trusted(fv_path, map_location="cpu")
    fv = data["function_vector"] if isinstance(data, dict) else data
    return fv.detach().float().cpu().reshape(-1).numpy()


def load_role_activations_all_layers(activations_root, task, split, token_role, expected_icl_index):
    """Return (n_selected, n_layers, hidden) float32 activations for one task/role/ICL."""
    split_dir = activations_root / task / split
    index = load_json(split_dir / "index.json")
    chunks = []
    for shard in index["shards"]:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = split_dir / shard_path
        data = torch_load_trusted(shard_path, map_location="cpu")
        activations = data["activations"]
        shard_metadata = data["metadata"]
        if len(shard_metadata) != activations.shape[0]:
            raise ValueError(f"Metadata/activation mismatch in {shard_path}")
        selected = [
            i
            for i, meta in enumerate(shard_metadata)
            if meta.get("token_role") == token_role and meta.get("icl_example_index") == expected_icl_index
        ]
        if selected:
            chunks.append(activations[selected].float())
    if not chunks:
        raise ValueError(f"No {token_role} activations found for {task}/{split}/ICL {expected_icl_index}")
    return torch.cat(chunks, dim=0).numpy()


def joint_project(x, act_mean, act_comp, fv_mean, fv_comp, act_k, fv_k):
    act_part = (x - act_mean) @ act_comp[:act_k].T
    fv_part = (x - fv_mean) @ fv_comp[:fv_k].T
    return np.concatenate([act_part, fv_part], axis=1)


def build_matrices(tasks, acts_layer, fvs, act_mean, act_comp, fv_mean, fv_comp, act_k, fv_k):
    """Return (X joint features, Y joint target, Y_raw true 4096-d FV) per prompt."""
    x_chunks, y_chunks, yraw_chunks = [], [], []
    for task in tasks:
        x_task = joint_project(acts_layer[task], act_mean, act_comp, fv_mean, fv_comp, act_k, fv_k)
        fv_joint = joint_project(fvs[task].reshape(1, -1), act_mean, act_comp, fv_mean, fv_comp, act_k, fv_k).reshape(-1)
        n = x_task.shape[0]
        x_chunks.append(x_task)
        y_chunks.append(np.repeat(fv_joint.reshape(1, -1), n, axis=0))
        yraw_chunks.append(np.repeat(fvs[task].reshape(1, -1), n, axis=0))
    return (np.concatenate(x_chunks, axis=0), np.concatenate(y_chunks, axis=0),
            np.concatenate(yraw_chunks, axis=0))


def mse(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))


def reconstruct_fv(pred_joint, fv_mean, fv_comp, act_k, fv_k):
    """Reconstruct 4096-d FV from the FV-PC half (cols act_k:act_k+fv_k) of a joint prediction."""
    return fv_mean + pred_joint[:, act_k:act_k + fv_k] @ fv_comp[:fv_k]


def fv_pc_floor(y_raw_fv, fv_mean, fv_comp, fv_k):
    """MSE a perfect (true FV-PC) prediction still incurs: variance outside the top-fv_k FV-PCs."""
    proj = (y_raw_fv - fv_mean) @ fv_comp[:fv_k].T
    recon = fv_mean + proj @ fv_comp[:fv_k]
    return mse(y_raw_fv, recon)


def plot_sweep(rows, icl_indices, token_roles, act_k, fv_k, floor, prev_layer, output_dir, min_layer=1):
    rows = [r for r in rows if r["layer"] >= min_layer]
    series = {}
    for r in rows:
        series.setdefault((r["icl_example_index"], r["token_role"]), []).append(r)
    for key in series:
        series[key] = sorted(series[key], key=lambda r: r["layer"])

    cmap = plt.get_cmap("viridis")
    colors = {icl: cmap(i / max(len(icl_indices) - 1, 1)) for i, icl in enumerate(icl_indices)}

    fig, axes = plt.subplots(1, len(token_roles), figsize=(6.0 * len(token_roles), 5.0), sharey=True, squeeze=False)
    axes = axes.reshape(-1)
    for ax, role in zip(axes, token_roles):
        ax.axhline(floor, color="0.55", linestyle="--", linewidth=1.2, zorder=1,
                   label="FV-PC recon floor\n(perfect prediction)")
        for icl in icl_indices:
            s = series.get((icl, role))
            if not s:
                continue
            layers = [r["layer"] for r in s]
            test = [r["fv_test_mse"] for r in s]
            ax.plot(layers, test, marker="o", markersize=3, linewidth=1.4, color=colors[icl], label=f"ICL {icl}")
            best = min(s, key=lambda r: r["fv_test_mse"])
            ax.scatter([best["layer"]], [best["fv_test_mse"]], s=70, facecolors="none",
                       edgecolors=colors[icl], linewidths=1.6, zorder=5)
        if prev_layer is not None:
            ax.axvline(prev_layer, color="0.6", linestyle="--", linewidth=1.0)
            ax.text(prev_layer, ax.get_ylim()[1], f" prev={prev_layer}", color="0.4",
                    fontsize=8, va="top", ha="left")
        ax.set_title(TOKEN_TITLES.get(role, role))
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Held-out test-task MSE in FV space (4096-d, FV-PC reconstruction)")
    axes[0].legend(title="series (○ = best layer)", fontsize=8)
    note = f"  (layers <{min_layer} omitted: embedding layer MSE is off-scale)" if min_layer > 0 else ""
    fig.suptitle(
        f"Layer sweep: OLS activation → function-vector regression, full-dim test MSE vs layer "
        f"(activation PCs={act_k}, FV PCs={fv_k}; MSE in 4096-d FV space; fit on train tasks → eval on held-out test tasks){note}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = output_dir / "layer_sweep_test_mse.png"
    pdf = output_dir / "layer_sweep_test_mse.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, train_tasks, test_tasks = load_task_manifest(args)
    all_tasks = train_tasks + test_tasks
    fvs = {task: load_function_vector(args.fv_root, task) for task in all_tasks}
    n_features = len(fvs[train_tasks[0]])

    # FV PCA: fit once on the train-task FVs (layer-independent), capped at fv_k_cap and rank n_train-1.
    act_k = args.k
    max_fv_components = min(args.fv_k_cap, len(train_tasks) - 1, n_features)
    if max_fv_components < args.fv_k_cap:
        print(f"FV-side cap reduced from {args.fv_k_cap} to {max_fv_components} "
              f"(only {len(train_tasks)} train tasks -> rank {len(train_tasks) - 1}).")
    fv_k = min(act_k, max_fv_components)
    fv_fit = np.stack([fvs[t] for t in train_tasks], axis=0)
    fv_pca = PCA(n_components=max_fv_components).fit(fv_fit)
    fv_mean, fv_comp = fv_pca.mean_, fv_pca.components_
    print(f"activation PCs k={act_k}, FV PCs fv_k={fv_k} (cap {max_fv_components}); joint dim={act_k + fv_k}")

    # FV-PC reconstruction floor is layer/icl/role-independent: depends only on test FVs and fv_k.
    yraw_test_unique = np.stack([fvs[t] for t in test_tasks], axis=0)
    floor = fv_pc_floor(yraw_test_unique, fv_mean, fv_comp, fv_k)
    print(f"FV-PC reconstruction floor (test tasks, fv_k={fv_k}): {floor:.4f}")

    rows = []
    for icl in args.icl_example_indices:
        activations_root = Path(args.activations_root_template.format(icl=icl))
        if not activations_root.exists():
            raise FileNotFoundError(activations_root)
        for role in args.token_roles:
            # Load all-layer activations for every task once for this (icl, role).
            acts = {task: load_role_activations_all_layers(activations_root, task, args.split, role, icl)
                    for task in all_tasks}
            n_layers = next(iter(acts.values())).shape[1]
            layers = args.layers if args.layers is not None else list(range(n_layers))
            for layer in layers:
                if layer < 0 or layer >= n_layers:
                    raise IndexError(f"Layer {layer} outside [0, {n_layers})")
                acts_layer = {task: acts[task][:, layer, :] for task in all_tasks}
                # Fresh activation PCA at this layer, fit on train tasks.
                x_train_fit = np.concatenate([acts_layer[t] for t in train_tasks], axis=0)
                max_act = min(act_k, x_train_fit.shape[0], n_features)
                act_pca = PCA(n_components=max_act).fit(x_train_fit)
                act_mean, act_comp = act_pca.mean_, act_pca.components_

                x_tr, y_tr, yraw_tr = build_matrices(train_tasks, acts_layer, fvs, act_mean, act_comp, fv_mean, fv_comp, max_act, fv_k)
                x_te, y_te, yraw_te = build_matrices(test_tasks, acts_layer, fvs, act_mean, act_comp, fv_mean, fv_comp, max_act, fv_k)
                model = LinearRegression().fit(x_tr, y_tr)
                pred_tr, pred_te = model.predict(x_tr), model.predict(x_te)
                # Reconstruct predicted FV in original 4096-d space and score there.
                fvrec_tr = reconstruct_fv(pred_tr, fv_mean, fv_comp, max_act, fv_k)
                fvrec_te = reconstruct_fv(pred_te, fv_mean, fv_comp, max_act, fv_k)
                rows.append({
                    "icl_example_index": int(icl),
                    "token_role": role,
                    "layer": int(layer),
                    "activation_k": int(max_act),
                    "fv_k": int(fv_k),
                    "feature_dim": int(max_act + fv_k),
                    "train_sample_count": int(x_tr.shape[0]),
                    "test_sample_count": int(x_te.shape[0]),
                    "pca_train_mse": mse(y_tr, pred_tr),
                    "pca_test_mse": mse(y_te, pred_te),
                    "fv_train_mse": mse(yraw_tr, fvrec_tr),
                    "fv_test_mse": mse(yraw_te, fvrec_te),
                    "fv_recon_floor_test_mse": floor,
                })
            del acts
            print(f"icl{icl} {role}: swept {len(layers)} layers")

    # CSV + JSON
    csv_path = args.output_dir / "layer_sweep_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["icl_example_index", "token_role", "layer", "activation_k", "fv_k",
                                               "feature_dim", "train_sample_count", "test_sample_count",
                                               "pca_train_mse", "pca_test_mse",
                                               "fv_train_mse", "fv_test_mse", "fv_recon_floor_test_mse"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output_dir / "layer_sweep_metrics.json", rows)

    png, pdf = plot_sweep(rows, args.icl_example_indices, args.token_roles, act_k, fv_k, floor, args.prev_layer,
                          args.output_dir, min_layer=args.plot_min_layer)

    # Best layer per (icl, role) by full-dim FV test MSE.
    best = {}
    for r in rows:
        key = f"icl{r['icl_example_index']}/{r['token_role']}"
        if key not in best or r["fv_test_mse"] < best[key]["fv_test_mse"]:
            best[key] = {"layer": r["layer"], "fv_test_mse": r["fv_test_mse"]}
    write_json(args.output_dir / "run_config.json", {
        "task_manifest": str(args.task_manifest),
        "fv_root": str(args.fv_root),
        "activations_root_template": args.activations_root_template,
        "output_dir": str(args.output_dir),
        "model": "OLS (LinearRegression, fit_intercept=True)",
        "activation_k": act_k,
        "fv_k": int(fv_k),
        "fv_k_cap": int(args.fv_k_cap),
        "fv_pca_max_components": int(max_fv_components),
        "test_mse_metric": "fv_test_mse = MSE vs true 4096-d FV, reconstructed from the model's FV-PC outputs (fv_mean + pred[:,act_k:act_k+fv_k] @ FV-PCs[:fv_k])",
        "fv_recon_floor_test_mse": floor,
        "best_layer_metric": "fv_test_mse",
        "split": args.split,
        "icl_example_indices": args.icl_example_indices,
        "token_roles": args.token_roles,
        "layers": args.layers if args.layers is not None else "all",
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "best_layer_per_series": best,
        "metrics_csv": str(csv_path),
        "plot_png": str(png),
    })
    print(csv_path)
    print(png)
    print("best layer per series (by full-dim FV test MSE):")
    for key, v in sorted(best.items()):
        print(f"  {key:28s} layer={v['layer']:>2}  fv_test_mse={v['fv_test_mse']:.4f}")


if __name__ == "__main__":
    main()
