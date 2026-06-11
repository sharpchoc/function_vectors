#!/usr/bin/env python
"""Sweep the joint-PCA dimensionality k for the OLS activation->function-vector regression.

Companion to sweep_layer_activation_to_fv_ols.py, but fixes the layer and sweeps k (the
number of PCA components used from each of the activation and FV PCA spaces; joint feature
dim = 2k). Methodology otherwise matches: fit activation PCA on train tasks at the fixed
layer, fit the FV PCA on the train-task FVs, build the joint k-dim space, OLS fit on train
tasks, evaluate held-out test-task MSE.

Because PCA components are nested, both PCAs are fit once with max(k) components and then
sliced to [:k] for each k. k is capped at (#train tasks - 1) because the FV PCA, fit on the
train-task FVs, has at most that many independent directions.

Produces one panel per token position; within each, one test-MSE-vs-k series per ICL
example (default ICL 2-5). The "best" k per series is the one with lowest test MSE.
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


DEFAULT_TOKEN_ROLES = ["pre_label_token", "first_label_token", "last_label_token"]
TOKEN_TITLES = {
    "pre_label_token": "Pre-label token",
    "first_label_token": "First label token",
    "last_label_token": "Last label token",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep k for the OLS joint-PCA activation->FV regression.")
    parser.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv_multitask_top10"))
    parser.add_argument(
        "--activations_root_template",
        type=str,
        default="results/residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results/k_sweeps/activation_to_fv_ols_multitask_top10"))
    parser.add_argument("--icl_example_indices", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--token_roles", nargs="+", default=DEFAULT_TOKEN_ROLES)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--k_min", type=int, default=5)
    parser.add_argument("--k_max", type=int, default=19)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--train_tasks", nargs="+", default=None)
    parser.add_argument("--test_tasks", nargs="+", default=None)
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


def load_role_activations(activations_root, task, split, layer, token_role, expected_icl_index):
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
        if layer < 0 or layer >= activations.shape[1]:
            raise IndexError(f"Layer {layer} outside activation shape {tuple(activations.shape)}")
        selected = [
            i for i, meta in enumerate(shard_metadata)
            if meta.get("token_role") == token_role and meta.get("icl_example_index") == expected_icl_index
        ]
        if selected:
            chunks.append(activations[selected, layer, :].float())
    if not chunks:
        raise ValueError(f"No {token_role} activations for {task}/{split}/ICL {expected_icl_index}")
    return torch.cat(chunks, dim=0).numpy()


def joint_project(x, act_mean, act_comp, fv_mean, fv_comp, k):
    act_part = (x - act_mean) @ act_comp[:k].T
    fv_part = (x - fv_mean) @ fv_comp[:k].T
    return np.concatenate([act_part, fv_part], axis=1)


def build_matrices(tasks, acts_by_task, fvs, act_mean, act_comp, fv_mean, fv_comp, k):
    """Return (X joint features, Y joint target, Y_raw true 4096-d FV) per prompt."""
    x_chunks, y_chunks, yraw_chunks = [], [], []
    for task in tasks:
        x_task = joint_project(acts_by_task[task], act_mean, act_comp, fv_mean, fv_comp, k)
        fv_joint = joint_project(fvs[task].reshape(1, -1), act_mean, act_comp, fv_mean, fv_comp, k).reshape(-1)
        n = x_task.shape[0]
        x_chunks.append(x_task)
        y_chunks.append(np.repeat(fv_joint.reshape(1, -1), n, axis=0))
        yraw_chunks.append(np.repeat(fvs[task].reshape(1, -1), n, axis=0))
    return (np.concatenate(x_chunks, axis=0), np.concatenate(y_chunks, axis=0),
            np.concatenate(yraw_chunks, axis=0))


def mse(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))


def reconstruct_fv(pred_joint, fv_mean, fv_comp, k):
    """Reconstruct 4096-d FV from the FV-PC half (cols k:2k) of a joint prediction."""
    return fv_mean + pred_joint[:, k:] @ fv_comp[:k]


def fv_pc_floor(y_raw_fv, fv_mean, fv_comp, k):
    """MSE a perfect (true FV-PC) prediction still incurs: variance outside the top-k FV-PCs."""
    proj = (y_raw_fv - fv_mean) @ fv_comp[:k].T
    recon = fv_mean + proj @ fv_comp[:k]
    return mse(y_raw_fv, recon)


def plot_sweep(rows, icl_indices, token_roles, layer, output_dir):
    series = {}
    for r in rows:
        series.setdefault((r["icl_example_index"], r["token_role"]), []).append(r)
    for key in series:
        series[key] = sorted(series[key], key=lambda r: r["k"])

    cmap = plt.get_cmap("viridis")
    colors = {icl: cmap(i / max(len(icl_indices) - 1, 1)) for i, icl in enumerate(icl_indices)}

    fig, axes = plt.subplots(1, len(token_roles), figsize=(6.0 * len(token_roles), 5.0), sharey=True, squeeze=False)
    axes = axes.reshape(-1)
    # Reconstruction floor (perfect FV-PC prediction): depends only on k, same across series.
    floor_by_k = {}
    for r in rows:
        floor_by_k.setdefault(r["k"], []).append(r["fv_recon_floor_test_mse"])
    floor_ks = sorted(floor_by_k)
    floor_vals = [float(np.mean(floor_by_k[k])) for k in floor_ks]
    for ax, role in zip(axes, token_roles):
        ax.plot(floor_ks, floor_vals, color="0.55", linestyle="--", linewidth=1.2, zorder=1,
                label="FV-PC recon floor\n(perfect prediction)")
        for icl in icl_indices:
            s = series.get((icl, role))
            if not s:
                continue
            ks = [r["k"] for r in s]
            test = [r["fv_test_mse"] for r in s]
            ax.plot(ks, test, marker="o", markersize=3.5, linewidth=1.4, color=colors[icl], label=f"ICL {icl}")
            best = min(s, key=lambda r: r["fv_test_mse"])
            ax.scatter([best["k"]], [best["fv_test_mse"]], s=80, facecolors="none",
                       edgecolors=colors[icl], linewidths=1.8, zorder=5)
        ax.set_title(TOKEN_TITLES.get(role, role))
        ax.set_xlabel("k (PCA components per space)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Held-out test-task MSE in FV space (4096-d, FV-PC reconstruction)")
    axes[0].legend(title="series (○ = best k)", fontsize=8)
    fig.suptitle(
        f"k sweep: OLS activation → function-vector regression, test MSE vs k "
        f"(layer {layer}; MSE in original 4096-d FV space via FV-PC reconstruction; train→test tasks)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = output_dir / "k_sweep_test_mse.png"
    pdf = output_dir / "k_sweep_test_mse.pdf"
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

    # FV PCA: fit once on train-task FVs (rank <= n_train - 1).
    max_fv_components = min(len(train_tasks) - 1, len(fvs[train_tasks[0]]))
    k_max = min(args.k_max, max_fv_components)
    if k_max < args.k_max:
        print(f"Capping k_max from {args.k_max} to {k_max} (FV PCA has at most {max_fv_components} "
              f"components from {len(train_tasks)} train tasks).")
    k_values = list(range(args.k_min, k_max + 1))
    fv_fit = np.stack([fvs[t] for t in train_tasks], axis=0)
    fv_pca = PCA(n_components=k_max).fit(fv_fit)
    fv_mean, fv_comp = fv_pca.mean_, fv_pca.components_

    rows = []
    for icl in args.icl_example_indices:
        activations_root = Path(args.activations_root_template.format(icl=icl))
        if not activations_root.exists():
            raise FileNotFoundError(activations_root)
        for role in args.token_roles:
            acts = {t: load_role_activations(activations_root, t, args.split, args.layer, role, icl)
                    for t in all_tasks}
            # Activation PCA: fit once at this (icl, role, fixed layer) on train tasks.
            x_train_fit = np.concatenate([acts[t] for t in train_tasks], axis=0)
            act_pca = PCA(n_components=k_max).fit(x_train_fit)
            act_mean, act_comp = act_pca.mean_, act_pca.components_
            for k in k_values:
                x_tr, y_tr, yraw_tr = build_matrices(train_tasks, acts, fvs, act_mean, act_comp, fv_mean, fv_comp, k)
                x_te, y_te, yraw_te = build_matrices(test_tasks, acts, fvs, act_mean, act_comp, fv_mean, fv_comp, k)
                model = LinearRegression().fit(x_tr, y_tr)
                pred_tr, pred_te = model.predict(x_tr), model.predict(x_te)
                # Reconstruct predicted FV in original 4096-d space and score there.
                fvrec_tr = reconstruct_fv(pred_tr, fv_mean, fv_comp, k)
                fvrec_te = reconstruct_fv(pred_te, fv_mean, fv_comp, k)
                rows.append({
                    "icl_example_index": int(icl),
                    "token_role": role,
                    "k": int(k),
                    "layer": int(args.layer),
                    "train_sample_count": int(x_tr.shape[0]),
                    "test_sample_count": int(x_te.shape[0]),
                    "pca_train_mse": mse(y_tr, pred_tr),
                    "pca_test_mse": mse(y_te, pred_te),
                    "fv_train_mse": mse(yraw_tr, fvrec_tr),
                    "fv_test_mse": mse(yraw_te, fvrec_te),
                    "fv_recon_floor_test_mse": fv_pc_floor(yraw_te, fv_mean, fv_comp, k),
                })
            print(f"icl{icl} {role}: swept k={k_values[0]}..{k_values[-1]}")

    csv_path = args.output_dir / "k_sweep_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["icl_example_index", "token_role", "k", "layer",
                                               "train_sample_count", "test_sample_count",
                                               "pca_train_mse", "pca_test_mse",
                                               "fv_train_mse", "fv_test_mse", "fv_recon_floor_test_mse"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output_dir / "k_sweep_metrics.json", rows)

    png, pdf = plot_sweep(rows, args.icl_example_indices, args.token_roles, args.layer, args.output_dir)

    best = {}
    for r in rows:
        key = f"icl{r['icl_example_index']}/{r['token_role']}"
        if key not in best or r["fv_test_mse"] < best[key]["fv_test_mse"]:
            best[key] = {"k": r["k"], "fv_test_mse": r["fv_test_mse"]}
    write_json(args.output_dir / "run_config.json", {
        "task_manifest": str(args.task_manifest),
        "fv_root": str(args.fv_root),
        "activations_root_template": args.activations_root_template,
        "output_dir": str(args.output_dir),
        "model": "OLS (LinearRegression, fit_intercept=True)",
        "layer": args.layer,
        "k_values": k_values,
        "fv_pca_max_components": int(max_fv_components),
        "test_mse_metric": "fv_test_mse = MSE vs true 4096-d FV, reconstructed from the model's FV-PC outputs (fv_mean + pred[:,k:] @ FV-PCs[:k])",
        "best_k_metric": "fv_test_mse",
        "split": args.split,
        "icl_example_indices": args.icl_example_indices,
        "token_roles": args.token_roles,
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "best_k_per_series": best,
        "metrics_csv": str(csv_path),
        "plot_png": str(png),
    })
    print(csv_path)
    print(png)
    print("best k per series (by FV-space test MSE):")
    for key, v in sorted(best.items()):
        print(f"  {key:28s} k={v['k']:>2}  fv_test_mse={v['fv_test_mse']:.4f}")


if __name__ == "__main__":
    main()
