"""
Decode predicted FVs from the magnitude/identity activations using the varicl_top40 PCA-ridge
map (icl10 cells), role-matched, then project the decoded FV onto the constrained top-20
magnitude/identity FVs. Per-layer scatter grids (one figure per token position).

Map (refit deterministically; the fitted map was never saved): per (map_role, layer),
  X = 20 abstractive TRAIN tasks' residual activations at that cell
      (artifacts/residual_activations/gptj_56tasks_170prompts_4tokens, icl10)
  Y = train_varicl_top40 FV for each task
  pipeline = act-PCA(k=16) -> standardize -> ridge(alpha from recorded metrics) -> FV-PCs(k=16)
             -> reconstruct to 4096-d   (exactly mirrors regress_activation_to_fv_pca_ridge.py)
Role match: ambiguous pre/first/last -> same map role; query_predictive -> last_prompt_token map.
"""
import os, sys, json, csv
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "src"))
from regress_activation_to_fv_pca_ridge import (
    torch_pca, project, reconstruct, ridge_eig_prep, ridge_predict,
    load_function_vector, load_task_role_pooled, role_load_icl_index, load_json)
from utils.paths import ARTIFACTS_ROOT, AMBIGUOUS_DIR, FV_FORMATION_DIR

VARICL_DIR = FV_FORMATION_DIR / "activation_to_fv_decoding/pca_ridge/varicl_top40"
FV_ROOT = ARTIFACTS_ROOT / "function_vectors" / "gpt-j" / "train_varicl_top40"
ACT_ROOT = ARTIFACTS_ROOT / "residual_activations" / "gptj_56tasks_170prompts_4tokens"
AMBIG_DIR = ARTIFACTS_ROOT / "magnitude_identity_activations" / "gpt-j-6b"
CONSTR_FV = ARTIFACTS_ROOT / "gptj_fv_ambiguous_constrained_top20"
FIG_DIR = AMBIGUOUS_DIR / "figures"
OUT_DIR = ARTIFACTS_ROOT / "magid_decoded_varicl40"

MAP_ROLES = ["pre_label_token", "first_label_token", "last_label_token", "last_prompt_token"]
SPLITS = ["train", "test"]
K_ACT = K_FV = 16
ICL = 10
COLORS = {"magnitude": "tab:red", "identity": "tab:blue"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32


def read_best_alpha():
    """(role, layer) -> best_alpha and recorded test_mse, from the saved metrics.csv."""
    alpha, rec_mse = {}, {}
    with open(VARICL_DIR / "shard_icl10" / "metrics.csv") as f:
        for row in csv.DictReader(f):
            key = (row["token_role"], int(row["layer"]))
            alpha[key] = float(row["best_alpha"]); rec_mse[key] = float(row["test_mse"])
    return alpha, rec_mse


def unit_fv(task):
    fv = torch.load(CONSTR_FV / task / f"{task}_function_vector.pt",
                    weights_only=False)["function_vector"].squeeze().float().to(DEV)
    return fv / fv.norm()


def main():
    cfg = load_json(VARICL_DIR / "shard_icl10" / "run_config.json")
    train_tasks, test_tasks = cfg["train_tasks"], cfg["test_tasks"]
    best_alpha, rec_mse = read_best_alpha()

    # FV PCA (k_fv) once on the 20 train varicl_top40 FVs
    fv_raw = {t: load_function_vector(FV_ROOT, t).to(DEV, DT) for t in train_tasks + test_tasks}
    fv_mean, fv_comp = torch_pca(torch.stack([fv_raw[t] for t in train_tasks]), K_FV)
    fv_proj = {t: project(fv_raw[t].unsqueeze(0), fv_mean, fv_comp).squeeze(0) for t in fv_raw}

    # preload TRAIN activations per (task, map_role): [n,29,4096]
    train_acts = {}
    for role in MAP_ROLES:
        li = role_load_icl_index(role, ICL)
        for t in train_tasks:
            train_acts[(t, role)] = load_task_role_pooled(ACT_ROOT, t, SPLITS, role, li)
    test_acts = {}
    for role in MAP_ROLES:
        li = role_load_icl_index(role, ICL)
        for t in test_tasks:
            test_acts[(t, role)] = load_task_role_pooled(ACT_ROOT, t, SPLITS, role, li)
    n_layers = train_acts[(train_tasks[0], MAP_ROLES[0])].shape[1]

    # ambiguous activations + projection axes
    fv_mag, fv_id = unit_fv("magnitude"), unit_fv("identity")
    amb = {}
    for task in ["magnitude", "identity"]:
        d = torch.load(AMBIG_DIR / f"{task}.pt", weights_only=False)
        amb[task] = (d["activations"], d["metadata"])

    def amb_rows(task, amb_role, demo):
        A, M = amb[task]
        idx = [i for i, m in enumerate(M) if m["token_role"] == amb_role and m["demo_index"] == demo]
        return A[idx].to(DEV, DT)                                  # [200, 29, 4096]

    # positions: (label, map_role, ambiguous_role, demo)
    positions = []
    for demo in [1, 2, 3, 4]:
        for r in ["pre_label_token", "first_label_token", "last_label_token"]:
            positions.append((f"demo{demo}_{r.split('_')[0]}", r, r, demo))
    positions.append(("query", "last_prompt_token", "query_predictive_token", None))

    # preload ambiguous rows per position
    pos_acts = {lab: {tk: amb_rows(tk, ar, dm) for tk in COLORS} for lab, mr, ar, dm in positions}

    # results[label][task] = dict(x=[200,29], y=[200,29])
    res = {lab: {tk: {"x": np.zeros((pos_acts[lab][tk].shape[0], n_layers)),
                      "y": np.zeros((pos_acts[lab][tk].shape[0], n_layers))} for tk in COLORS}
           for lab, *_ in [(p[0],) for p in positions]}
    sanity = []

    for role in MAP_ROLES:
        for layer in range(n_layers):
            alpha = best_alpha[(role, layer)]
            # fit map on 20 train tasks at this cell
            x_raw = {t: train_acts[(t, role)][:, layer, :].to(DEV, DT) for t in train_tasks}
            x_pool = torch.cat([x_raw[t] for t in train_tasks], 0)
            act_mean, act_comp = torch_pca(x_pool, K_ACT)
            xproj = {t: project(x_raw[t], act_mean, act_comp) for t in train_tasks}
            ppool = torch.cat([xproj[t] for t in train_tasks], 0)
            smean = ppool.mean(0); sstd = ppool.std(0, unbiased=False).clamp_min(1e-6)
            x_fit = torch.cat([(xproj[t] - smean) / sstd for t in train_tasks], 0)
            y_fit = torch.cat([fv_proj[t].unsqueeze(0).expand(xproj[t].shape[0], -1) for t in train_tasks], 0)
            xbar, ybar, evals, evecs, c = ridge_eig_prep(x_fit, y_fit)

            def decode(v):                                          # v: [n,4096] -> [n,4096]
                vp = (project(v, act_mean, act_comp) - smean) / sstd
                pred = ridge_predict(vp, xbar, ybar, evals, evecs, c, alpha)
                return reconstruct(pred, fv_mean, fv_comp)

            # sanity: held-out test FV decode MSE at this cell
            te_sq = te_n = 0.0
            for t in test_tasks:
                v = test_acts[(t, role)][:, layer, :].to(DEV, DT)
                dec = decode(v)
                te_sq += float(((dec - fv_raw[t].unsqueeze(0)) ** 2).sum()); te_n += v.shape[0] * dec.shape[1]
            sanity.append({"role": role, "layer": layer, "test_mse": te_sq / te_n,
                           "recorded": rec_mse[(role, layer)]})

            # apply to matching ambiguous positions
            for lab, mr, ar, dm in positions:
                if mr != role:
                    continue
                for tk in COLORS:
                    dec = decode(pos_acts[lab][tk][:, layer, :])
                    res[lab][tk]["x"][:, layer] = (dec @ fv_mag).cpu().numpy()
                    res[lab][tk]["y"][:, layer] = (dec @ fv_id).cpu().numpy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"res": res, "sanity": sanity, "positions": [p[0] for p in positions]},
               OUT_DIR / "projections.pt")

    # ---- sanity report ----
    L13 = [s for s in sanity if s["layer"] == 13]
    print("refit fidelity @ L13 (decoded vs recorded test_mse):")
    for s in L13:
        print(f"  {s['role']:18s} refit={s['test_mse']:.4f} recorded={s['recorded']:.4f} "
              f"diff={abs(s['test_mse']-s['recorded']):.4f}")

    # ---- per-layer grid figure per position ----
    FIG_DIR.mkdir(exist_ok=True)
    for lab, *_ in [(p[0],) for p in positions]:
        # shared limits across layers
        allx = np.concatenate([res[lab][tk]["x"].ravel() for tk in COLORS])
        ally = np.concatenate([res[lab][tk]["y"].ravel() for tk in COLORS])
        xlim = (allx.min(), allx.max()); ylim = (ally.min(), ally.max())
        ncols = 6; nrows = int(np.ceil(n_layers / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows), squeeze=False)
        for L in range(n_layers):
            ax = axes[L // ncols][L % ncols]
            for tk in COLORS:
                ax.scatter(res[lab][tk]["x"][:, L], res[lab][tk]["y"][:, L], s=7, alpha=0.5,
                           color=COLORS[tk], edgecolors="none", label=f"{tk} task" if L == 0 else None)
            lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
            ax.plot([lo, hi], [lo, hi], ls=":", c="gray", lw=0.7)
            ax.set_title("embed" if L == 0 else f"L{L}", fontsize=8)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.tick_params(labelsize=5)
            ax.set_xlabel("dec·mag FV", fontsize=5); ax.set_ylabel("dec·id FV", fontsize=5)
        for j in range(n_layers, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, fontsize=11, markerscale=2, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"Decoded FV (varicl_top40 icl10 map) projected onto mag/id FVs — {lab} — by layer",
                     fontsize=12, y=1.0)
        fig.tight_layout(rect=[0, 0.02, 1, 0.99])
        out = FIG_DIR / f"decoded_varicl40_perlayer_{lab}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
