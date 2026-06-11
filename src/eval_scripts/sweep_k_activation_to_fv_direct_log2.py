#!/usr/bin/env python
"""Log-spaced k sweep for a DIRECT k_activations -> k_FV PCA regression.

This is the "direct" counterpart to sweep_k_activation_to_fv_ols_log2.py. The two scripts
differ ONLY in what space the OLS regression lives in:

  * joint-space (sweep_k_activation_to_fv_ols_log2.py): both the activation and the FV are
    projected onto the SAME concatenated [activation-PCA (k_act) | FV-PCA (k_fv)] basis, and
    OLS maps the (k_act + k_fv)-dim joint projection of the activation -> the (k_act + k_fv)-dim
    joint projection of the FV.

  * direct (THIS script): the activation is projected onto its own activation-PCA basis to
    k_activations components (the INPUT), the FV is projected onto its own FV-PCA basis to
    k_FV components (the TARGET), and OLS regresses R^{k_activations} -> R^{k_FV} directly.
    Predicted FV-PCs are reconstructed to 4096-d (fv_mean + pred @ FV-PCs[:k_FV]) and scored
    against the true FV. This is the standard "reduce both sides, regress between them" setup.

k is swept on a base-2 (doubling) grid -- k = 1, 2, 4, 8, 16, 32, ... up to --k_max. The
activation side can grow large; the FV side is capped at --fv_k_cap (default 16) and at the
FV-PCA rank limit (#train tasks - 1). By default the two are coupled on the diagonal:

    k_activations = k
    k_FV          = min(k, fv_k_cap)     # (1,1),(2,2),(4,4),(8,8),(16,16),(32,16),(64,16),...

Pass --fix_fv_k to instead hold k_FV = fv_k_cap for every k (sweep k_activations alone).

MSE is reported in the original 4096-d FV space. The grey dashed "FV-PC reconstruction floor"
is the error a perfect k_FV-PC prediction still incurs; it depends only on k_FV.

Fit on train tasks, evaluate on held-out test tasks. One panel per token role; within each,
one test-MSE-vs-k series per ICL example. The x axis is log-scaled and each tick is labelled
with the (k_activations / k_FV) pair used.
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
    parser = argparse.ArgumentParser(description="Log-spaced (doubling) k sweep for the DIRECT k_activations->k_FV PCA OLS regression.")
    parser.add_argument("--task_manifest", type=Path, default=Path("task_splits/abstractive_train_test_tasks_29.json"))
    parser.add_argument("--fv_root", type=Path, default=Path("results/gptj_fv_multitask_top10"))
    parser.add_argument(
        "--activations_root_template",
        type=str,
        default="results/residual_activations/gptj_56tasks_170prompts_icl{icl}_3tokens",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results/k_sweeps/activation_to_fv_direct_ols_multitask_top10_log2"))
    parser.add_argument("--icl_example_indices", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--token_roles", nargs="+", default=DEFAULT_TOKEN_ROLES)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--k_min", type=int, default=1, help="Smallest k (activation PCs); included if a power of 2 >= this.")
    parser.add_argument("--k_max", type=int, default=100, help="Largest k (activation PCs) to attempt; doubling stops here.")
    parser.add_argument("--fv_k_cap", type=int, default=16, help="Maximum FV-side PCs (target rank cap).")
    parser.add_argument("--fix_fv_k", action="store_true",
                        help="Hold the FV-side PCs fixed at fv_k_cap for EVERY k (sweep activation-k only) "
                             "instead of the default coupled diagonal k_FV = min(k, fv_k_cap).")
    parser.add_argument("--fix_act_k", type=int, default=None,
                        help="Hold k_activations fixed at this many act-PCs and sweep k_FV instead "
                             "(the doubling grid is interpreted as k_FV, capped at fv_k_cap). "
                             "Mutually exclusive with --fix_fv_k.")
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


def doubling_k_values(k_min, k_max):
    """Powers of two within [k_min, k_max]: 1, 2, 4, 8, ..."""
    vals = []
    k = 1
    while k <= k_max:
        if k >= k_min:
            vals.append(k)
        k *= 2
    return vals


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


def project(x, mean, comp, k):
    """Project rows of x onto the first k PCs: (x - mean) @ comp[:k].T -> dim k."""
    return (x - mean) @ comp[:k].T


def build_matrices(tasks, acts_by_task, fvs, act_mean, act_comp, fv_mean, fv_comp, act_k, fv_k):
    """Return (X = activation PCs (act_k), Y = FV PCs (fv_k), Y_raw = true 4096-d FV) per prompt."""
    x_chunks, y_chunks, yraw_chunks = [], [], []
    for task in tasks:
        x_task = project(acts_by_task[task], act_mean, act_comp, act_k)            # [n, act_k]
        fv_pcs = project(fvs[task].reshape(1, -1), fv_mean, fv_comp, fv_k).reshape(-1)  # [fv_k]
        n = x_task.shape[0]
        x_chunks.append(x_task)
        y_chunks.append(np.repeat(fv_pcs.reshape(1, -1), n, axis=0))
        yraw_chunks.append(np.repeat(fvs[task].reshape(1, -1), n, axis=0))
    return (np.concatenate(x_chunks, axis=0), np.concatenate(y_chunks, axis=0),
            np.concatenate(yraw_chunks, axis=0))


def mse(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))


def reconstruct_fv(pred_fv_pcs, fv_mean, fv_comp, fv_k):
    """Reconstruct 4096-d FV from predicted FV-PCs (dim fv_k)."""
    return fv_mean + pred_fv_pcs[:, :fv_k] @ fv_comp[:fv_k]


def fv_pc_floor(y_raw_fv, fv_mean, fv_comp, fv_k):
    """MSE a perfect (true FV-PC) prediction still incurs: variance outside the top-fv_k FV-PCs."""
    proj = (y_raw_fv - fv_mean) @ fv_comp[:fv_k].T
    recon = fv_mean + proj @ fv_comp[:fv_k]
    return mse(y_raw_fv, recon)


def plot_sweep(rows, icl_indices, token_roles, layer, k_values, fv_k_cap, output_dir, fix_fv_k=False, fix_act_k=None):
    series = {}
    for r in rows:
        series.setdefault((r["icl_example_index"], r["token_role"]), []).append(r)
    for key in series:
        series[key] = sorted(series[key], key=lambda r: r["k"])

    cmap = plt.get_cmap("viridis")
    colors = {icl: cmap(i / max(len(icl_indices) - 1, 1)) for i, icl in enumerate(icl_indices)}

    # Reconstruction floor (perfect FV-PC prediction): depends only on FV-side k, flat past the cap.
    floor_by_k = {}
    for r in rows:
        floor_by_k.setdefault(r["k"], []).append(r["fv_recon_floor_test_mse"])
    floor_ks = sorted(floor_by_k)
    floor_vals = [float(np.mean(floor_by_k[k])) for k in floor_ks]

    fv_k_by_k = {r["k"]: r["fv_k"] for r in rows}
    act_k_by_k = {r["k"]: r["activation_k"] for r in rows}
    tick_ks = sorted(k_values)
    if fix_act_k is not None:
        tick_labels = [f"{k}\n/ act {act_k_by_k.get(k, fix_act_k)}" for k in tick_ks]
    else:
        tick_labels = [f"{k}\n/ FV {fv_k_by_k.get(k, min(k, fv_k_cap))}" for k in tick_ks]

    fig, axes = plt.subplots(1, len(token_roles), figsize=(6.0 * len(token_roles), 5.0), sharey=True, squeeze=False)
    axes = axes.reshape(-1)
    for ax, role in zip(axes, token_roles):
        ax.plot(floor_ks, floor_vals, color="0.55", linestyle="--", linewidth=1.2, zorder=1,
                label="FV-PC recon floor\n(perfect prediction)")
        for icl in icl_indices:
            s = series.get((icl, role))
            if not s:
                continue
            ks = [r["k"] for r in s]
            test = [r["fv_test_mse"] for r in s]
            ax.plot(ks, test, marker="o", markersize=4.5, linewidth=1.4, color=colors[icl], label=f"ICL {icl}")
            best = min(s, key=lambda r: r["fv_test_mse"])
            ax.scatter([best["k"]], [best["fv_test_mse"]], s=90, facecolors="none",
                       edgecolors=colors[icl], linewidths=1.8, zorder=5)
        ax.axvline(fv_k_cap, color="0.7", linestyle=":", linewidth=1.0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(tick_ks)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.minorticks_off()
        ax.set_title(TOKEN_TITLES.get(role, role))
        ax.set_xlabel("k_FV  /  k_activations (fixed)" if fix_act_k is not None else "k_activations  /  k_FV")
        ax.grid(alpha=0.25, which="major")
    axes[0].set_ylabel("Held-out test-task MSE in FV space (4096-d, FV-PC reconstruction)")
    axes[0].legend(title="series (○ = best k)", fontsize=8)
    if fix_act_k is not None:
        rule = f"k_activations = {fix_act_k} (fixed), sweep k_FV (≤ {fv_k_cap})"
        sweep_var = "k_FV"
    else:
        rule = (f"k_activations = k, k_FV = {fv_k_cap} (fixed)" if fix_fv_k
                else f"k_activations = k, k_FV = min(k, {fv_k_cap})")
        sweep_var = "k"
    fig.suptitle(
        f"k sweep (log2): DIRECT k_activations → k_FV OLS regression, test MSE vs {sweep_var} "
        f"(layer {layer}; {rule}; MSE in 4096-d FV space; train→test tasks)",
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
    n_features = len(fvs[train_tasks[0]])

    # FV PCA: fit once on train-task FVs, capped at fv_k_cap (and at the rank limit n_train - 1).
    max_fv_components = min(args.fv_k_cap, len(train_tasks) - 1, n_features)
    if max_fv_components < args.fv_k_cap:
        print(f"FV-side cap reduced from {args.fv_k_cap} to {max_fv_components} "
              f"(only {len(train_tasks)} train tasks -> rank {len(train_tasks) - 1}).")
    fv_k_cap = max_fv_components
    fv_fit = np.stack([fvs[t] for t in train_tasks], axis=0)
    fv_pca = PCA(n_components=fv_k_cap).fit(fv_fit)
    fv_mean, fv_comp = fv_pca.mean_, fv_pca.components_

    if args.fix_act_k is not None and args.fix_fv_k:
        raise ValueError("--fix_act_k and --fix_fv_k are mutually exclusive.")
    fixing_act = args.fix_act_k is not None

    requested_k_values = doubling_k_values(args.k_min, args.k_max)
    if fixing_act:
        print(f"Sweeping k_FV (doubling, capped at {fv_k_cap}); k_activations fixed = {args.fix_act_k}. "
              f"Requested k_FV values: {[k for k in requested_k_values if k <= fv_k_cap]}")
    else:
        fv_k_desc = f"{fv_k_cap} (fixed)" if args.fix_fv_k else f"min(k, {fv_k_cap})"
        print(f"Requested activation k values (doubling): {requested_k_values}; k_FV = {fv_k_desc}")

    rows = []
    dropped_notes = []
    for icl in args.icl_example_indices:
        activations_root = Path(args.activations_root_template.format(icl=icl))
        if not activations_root.exists():
            raise FileNotFoundError(activations_root)
        for role in args.token_roles:
            acts = {t: load_role_activations(activations_root, t, args.split, args.layer, role, icl)
                    for t in all_tasks}
            # Activation PCA: fit once at this (icl, role, fixed layer) on train tasks, with as many
            # components as the largest requested k allows.
            x_train_fit = np.concatenate([acts[t] for t in train_tasks], axis=0)
            n_train_samples = x_train_fit.shape[0]
            max_needed_act = args.fix_act_k if fixing_act else max(requested_k_values)
            max_act_components = min(max_needed_act, n_train_samples, n_features)
            act_pca = PCA(n_components=max_act_components).fit(x_train_fit)
            act_mean, act_comp = act_pca.mean_, act_pca.components_

            if fixing_act:
                # Swept doubling variable is k_FV (capped at fv_k_cap); k_activations is constant.
                valid_k_values = [k for k in requested_k_values if k <= fv_k_cap]
                dropped = [k for k in requested_k_values if k > fv_k_cap]
                if dropped:
                    note = f"icl{icl}/{role}: dropped k_FV={dropped} (> fv_k_cap {fv_k_cap})"
                    print(note); dropped_notes.append(note)
                if args.fix_act_k > max_act_components:
                    note = (f"icl{icl}/{role}: fix_act_k {args.fix_act_k} clipped to {max_act_components} "
                            f"(activation PCA rank from {n_train_samples} train samples)")
                    print(note); dropped_notes.append(note)
            else:
                valid_k_values = [k for k in requested_k_values if k <= max_act_components]
                dropped = [k for k in requested_k_values if k > max_act_components]
                if dropped:
                    note = (f"icl{icl}/{role}: dropped k={dropped} (activation PCA only supports "
                            f"{max_act_components} comps from {n_train_samples} train samples)")
                    print(note); dropped_notes.append(note)

            for k in valid_k_values:
                if fixing_act:
                    act_k = min(args.fix_act_k, max_act_components)
                    fv_k = min(k, fv_k_cap)
                elif args.fix_fv_k:
                    act_k = k
                    fv_k = fv_k_cap
                else:
                    act_k = k
                    fv_k = min(k, fv_k_cap)
                x_tr, y_tr, yraw_tr = build_matrices(train_tasks, acts, fvs, act_mean, act_comp, fv_mean, fv_comp, act_k, fv_k)
                x_te, y_te, yraw_te = build_matrices(test_tasks, acts, fvs, act_mean, act_comp, fv_mean, fv_comp, act_k, fv_k)
                model = LinearRegression().fit(x_tr, y_tr)
                pred_tr, pred_te = model.predict(x_tr), model.predict(x_te)
                fvrec_tr = reconstruct_fv(pred_tr, fv_mean, fv_comp, fv_k)
                fvrec_te = reconstruct_fv(pred_te, fv_mean, fv_comp, fv_k)
                rows.append({
                    "icl_example_index": int(icl),
                    "token_role": role,
                    "k": int(k),
                    "activation_k": int(act_k),
                    "fv_k": int(fv_k),
                    "feature_dim": int(act_k),
                    "target_dim": int(fv_k),
                    "layer": int(args.layer),
                    "train_sample_count": int(x_tr.shape[0]),
                    "test_sample_count": int(x_te.shape[0]),
                    "pca_train_mse": mse(y_tr, pred_tr),
                    "pca_test_mse": mse(y_te, pred_te),
                    "fv_train_mse": mse(yraw_tr, fvrec_tr),
                    "fv_test_mse": mse(yraw_te, fvrec_te),
                    "fv_recon_floor_test_mse": fv_pc_floor(yraw_te, fv_mean, fv_comp, fv_k),
                })
            print(f"icl{icl} {role}: swept k={valid_k_values}")

    csv_path = args.output_dir / "k_sweep_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["icl_example_index", "token_role", "k", "activation_k", "fv_k",
                                               "feature_dim", "target_dim", "layer",
                                               "train_sample_count", "test_sample_count",
                                               "pca_train_mse", "pca_test_mse",
                                               "fv_train_mse", "fv_test_mse", "fv_recon_floor_test_mse"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output_dir / "k_sweep_metrics.json", rows)

    all_k = sorted({r["k"] for r in rows})
    png, pdf = plot_sweep(rows, args.icl_example_indices, args.token_roles, args.layer, all_k, fv_k_cap,
                          args.output_dir, fix_fv_k=args.fix_fv_k, fix_act_k=args.fix_act_k)

    best = {}
    for r in rows:
        key = f"icl{r['icl_example_index']}/{r['token_role']}"
        if key not in best or r["fv_test_mse"] < best[key]["fv_test_mse"]:
            best[key] = {"k": r["k"], "fv_k": r["fv_k"], "fv_test_mse": r["fv_test_mse"]}
    write_json(args.output_dir / "run_config.json", {
        "task_manifest": str(args.task_manifest),
        "fv_root": str(args.fv_root),
        "activations_root_template": args.activations_root_template,
        "output_dir": str(args.output_dir),
        "model": "OLS (LinearRegression, fit_intercept=True)",
        "regression": "direct k_activations -> k_FV (project activations to k_act PCs, FVs to k_FV PCs, regress between them)",
        "layer": args.layer,
        "k_spacing": "doubling (powers of 2)",
        "requested_k_values": requested_k_values,
        "k_values_present": all_k,
        "fv_k_cap": int(fv_k_cap),
        "fix_fv_k": bool(args.fix_fv_k),
        "fix_act_k": args.fix_act_k,
        "sweep_variable": "k_FV" if fixing_act else "k_activations",
        "fv_k_rule": (f"k_activations fixed = {args.fix_act_k}, sweep k_FV = min(k, fv_k_cap)" if fixing_act
                      else ("k_FV = fv_k_cap (fixed)" if args.fix_fv_k else "k_FV = min(k, fv_k_cap)")),
        "fv_pca_max_components": int(max_fv_components),
        "test_mse_metric": "fv_test_mse = MSE vs true 4096-d FV, reconstructed from the model's predicted FV-PCs (fv_mean + pred @ FV-PCs[:k_FV])",
        "best_k_metric": "fv_test_mse",
        "split": args.split,
        "icl_example_indices": args.icl_example_indices,
        "token_roles": args.token_roles,
        "dropped_k_notes": dropped_notes,
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
        print(f"  {key:28s} k={v['k']:>3} (FV {v['fv_k']:>2})  fv_test_mse={v['fv_test_mse']:.4f}")


if __name__ == "__main__":
    main()
