"""
Final-query-token only, projecting onto the magnitude/identity FVs defined from the
TRAIN-TASK-SELECTED heads (multitask top-40: ARTIFACTS_ROOT/gptj_fv_multitask_top40_ambiguous).
Two per-layer scatter grids, colored by task:
  (A) RAW query activation projected onto those FVs   -> query_multitask40_raw_perlayer.png
  (B) DECODED FV (varicl_top40 icl10 last_prompt_token map, refit per layer) projected onto
      those FVs                                        -> query_multitask40_decoded_perlayer.png
"""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "src"))
from regress_activation_to_fv_pca_ridge import (
    torch_pca, project, reconstruct, ridge_eig_prep, ridge_predict,
    load_function_vector, load_task_role_pooled, role_load_icl_index, load_json)
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR, AMBIGUOUS_DIR
import csv

VARICL_DIR = FV_FORMATION_DIR / "activation_to_fv_decoding/pca_ridge/varicl_top40"
FV_TARGET = ARTIFACTS_ROOT / "function_vectors" / "gpt-j" / "train_varicl_top40"     # decoder target
PROJ_FV = ARTIFACTS_ROOT / "gptj_fv_multitask_top40_ambiguous"                       # NEW axes
ACT_ROOT = ARTIFACTS_ROOT / "residual_activations" / "gptj_56tasks_170prompts_4tokens"
AMBIG = ARTIFACTS_ROOT / "magnitude_identity_activations" / "gpt-j-6b"
FIG = AMBIGUOUS_DIR / "figures"
ROLE = "last_prompt_token"
K_ACT = K_FV = 16
COLORS = {"magnitude": "tab:red", "identity": "tab:blue"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.float32


def unit_proj_fv(task):
    fv = torch.load(PROJ_FV / task / f"{task}_function_vector.pt",
                    weights_only=False)["function_vector"].squeeze().float().to(DEV)
    return fv / fv.norm()


def query_acts(task):
    d = torch.load(AMBIG / f"{task}.pt", weights_only=False)
    idx = [i for i, m in enumerate(d["metadata"]) if m["token_role"] == "query_predictive_token"]
    return d["activations"][idx].to(DEV, DT)            # [200, 29, 4096]


def grid(projx, projy, title, out):
    n_layers = next(iter(projx.values())).shape[1]
    allx = np.concatenate([projx[t].ravel() for t in COLORS]); ally = np.concatenate([projy[t].ravel() for t in COLORS])
    xlim = (allx.min(), allx.max()); ylim = (ally.min(), ally.max())
    ncols = 6; nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows), squeeze=False)
    for L in range(n_layers):
        ax = axes[L // ncols][L % ncols]
        for t in COLORS:
            ax.scatter(projx[t][:, L], projy[t][:, L], s=8, alpha=0.5, color=COLORS[t],
                       edgecolors="none", label=f"{t} task" if L == 0 else None)
        lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
        ax.plot([lo, hi], [lo, hi], ls=":", c="gray", lw=0.7)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.tick_params(labelsize=6)
        ax.set_title("embed" if L == 0 else f"L{L}", fontsize=8)
        ax.set_xlabel("·mag FV", fontsize=6); ax.set_ylabel("·id FV", fontsize=6)
    for j in range(n_layers, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=11, markerscale=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    FIG.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    fv_mag, fv_id = unit_proj_fv("magnitude"), unit_proj_fv("identity")
    qa = {t: query_acts(t) for t in COLORS}
    n_layers = qa["magnitude"].shape[1]

    # ---- (A) RAW projection ----
    rawx = {t: torch.stack([qa[t][:, L, :] @ fv_mag for L in range(n_layers)], 1).cpu().numpy() for t in COLORS}
    rawy = {t: torch.stack([qa[t][:, L, :] @ fv_id for L in range(n_layers)], 1).cpu().numpy() for t in COLORS}
    grid(rawx, rawy,
         "RAW query activation projected onto magnitude/identity FVs (train-task top-40 heads) — by layer",
         FIG / "query_multitask40_raw_perlayer.png")

    # ---- (B) DECODED projection (refit last_prompt_token map per layer) ----
    cfg = load_json(VARICL_DIR / "shard_icl10" / "run_config.json")
    train_tasks = cfg["train_tasks"]
    best_alpha = {}
    with open(VARICL_DIR / "shard_icl10" / "metrics.csv") as f:
        for r in csv.DictReader(f):
            if r["token_role"] == ROLE:
                best_alpha[int(r["layer"])] = float(r["best_alpha"])
    fv_raw = {t: load_function_vector(FV_TARGET, t).to(DEV, DT) for t in train_tasks}
    fv_mean, fv_comp = torch_pca(torch.stack([fv_raw[t] for t in train_tasks]), K_FV)
    fv_proj = {t: project(fv_raw[t].unsqueeze(0), fv_mean, fv_comp).squeeze(0) for t in train_tasks}
    tr_acts = {t: load_task_role_pooled(ACT_ROOT, t, ["train", "test"], ROLE, role_load_icl_index(ROLE, 10))
               for t in train_tasks}

    decx = {t: np.zeros((qa[t].shape[0], n_layers)) for t in COLORS}
    decy = {t: np.zeros((qa[t].shape[0], n_layers)) for t in COLORS}
    for L in range(n_layers):
        xr = {t: tr_acts[t][:, L, :].to(DEV, DT) for t in train_tasks}
        act_mean, act_comp = torch_pca(torch.cat([xr[t] for t in train_tasks], 0), K_ACT)
        xp = {t: project(xr[t], act_mean, act_comp) for t in train_tasks}
        pp = torch.cat([xp[t] for t in train_tasks], 0)
        sm = pp.mean(0); ss = pp.std(0, unbiased=False).clamp_min(1e-6)
        x_fit = torch.cat([(xp[t] - sm) / ss for t in train_tasks], 0)
        y_fit = torch.cat([fv_proj[t].unsqueeze(0).expand(xp[t].shape[0], -1) for t in train_tasks], 0)
        xbar, ybar, ev, evec, c = ridge_eig_prep(x_fit, y_fit)
        for t in COLORS:
            vp = (project(qa[t][:, L, :], act_mean, act_comp) - sm) / ss
            dec = reconstruct(ridge_predict(vp, xbar, ybar, ev, evec, c, best_alpha[L]), fv_mean, fv_comp)
            decx[t][:, L] = (dec @ fv_mag).cpu().numpy(); decy[t][:, L] = (dec @ fv_id).cpu().numpy()
    grid(decx, decy,
         "DECODED FV (varicl_top40 icl10 query map) projected onto magnitude/identity FVs "
         "(train-task top-40 heads) — by layer",
         FIG / "query_multitask40_decoded_perlayer.png")


if __name__ == "__main__":
    main()
