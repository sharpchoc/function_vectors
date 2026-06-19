"""
Replot only the LAST 3 by-token positions (demo4 first-label, demo4 last-label, query) of the
decoded-FV by-token scatters, at layers 13 and 20, but projecting the decoded FV onto the
TRAIN-TASK-SELECTED-head magnitude/identity FVs (multitask top-40), colored by task.
"""
import sys, csv
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "src"))
from regress_activation_to_fv_pca_ridge import (
    torch_pca, project, reconstruct, ridge_eig_prep, ridge_predict,
    load_function_vector, load_task_role_pooled, role_load_icl_index, load_json)

VARICL = REPO / "results" / "pca_ridge_activation_to_fv_varicl_top40"
FV_TARGET = REPO / "results" / "function_vectors" / "gpt-j" / "train_varicl_top40"
PROJ_FV = REPO / "results" / "gptj_fv_multitask_top40_ambiguous"     # NEW axis
ACT_ROOT = REPO / "results" / "residual_activations" / "gptj_56tasks_170prompts_4tokens"
AMBIG = REPO / "results" / "magnitude_identity_activations" / "gpt-j-6b"
FIG = REPO / "figures"
K_ACT = K_FV = 16
LAYERS = [13, 20]
COLORS = {"magnitude": "tab:red", "identity": "tab:blue"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.float32

# the last 3 by-token positions: (label, map_role, ambiguous_role, demo_index)
POSITIONS = [
    ("demo4_first *diff*", "first_label_token", "first_label_token", 4),
    ("demo4_last *diff*",  "last_label_token",  "last_label_token",  4),
    ("query",              "last_prompt_token", "query_predictive_token", None),
]


def unit_fv(task):
    fv = torch.load(PROJ_FV / task / f"{task}_function_vector.pt",
                    weights_only=False)["function_vector"].squeeze().float().to(DEV)
    return fv / fv.norm()


def amb_rows(task, amb_role, demo):
    d = torch.load(AMBIG / f"{task}.pt", weights_only=False)
    idx = [i for i, m in enumerate(d["metadata"]) if m["token_role"] == amb_role and m["demo_index"] == demo]
    return d["activations"][idx].to(DEV, DT)


def main():
    cfg = load_json(VARICL / "shard_icl10" / "run_config.json"); train_tasks = cfg["train_tasks"]
    alpha = {}
    with open(VARICL / "shard_icl10" / "metrics.csv") as f:
        for r in csv.DictReader(f):
            alpha[(r["token_role"], int(r["layer"]))] = float(r["best_alpha"])

    fv_raw = {t: load_function_vector(FV_TARGET, t).to(DEV, DT) for t in train_tasks}
    fv_mean, fv_comp = torch_pca(torch.stack([fv_raw[t] for t in train_tasks]), K_FV)
    fv_proj = {t: project(fv_raw[t].unsqueeze(0), fv_mean, fv_comp).squeeze(0) for t in train_tasks}
    fv_mag, fv_id = unit_fv("magnitude"), unit_fv("identity")

    # reference points: where the ACTUAL train-task-head magnitude/identity FVs land on these axes
    mag_vec = torch.load(PROJ_FV / "magnitude" / "magnitude_function_vector.pt",
                         weights_only=False)["function_vector"].squeeze().float().to(DEV)
    id_vec = torch.load(PROJ_FV / "identity" / "identity_function_vector.pt",
                        weights_only=False)["function_vector"].squeeze().float().to(DEV)
    ref = {"magnitude": (float(mag_vec @ fv_mag), float(mag_vec @ fv_id)),
           "identity":  (float(id_vec @ fv_mag), float(id_vec @ fv_id))}
    print("reference FV projections (x=·mag, y=·id):", ref)

    # cache train activations + ambiguous rows for the 3 roles
    roles = sorted({mr for _, mr, _, _ in POSITIONS})
    tr = {role: {t: load_task_role_pooled(ACT_ROOT, t, ["train", "test"], role, role_load_icl_index(role, 10))
                 for t in train_tasks} for role in roles}
    amb = {lab: {tk: amb_rows(tk, ar, dm) for tk in COLORS} for lab, mr, ar, dm in POSITIONS}

    def fit(role, L):
        xr = {t: tr[role][t][:, L, :].to(DEV, DT) for t in train_tasks}
        am, ac = torch_pca(torch.cat([xr[t] for t in train_tasks], 0), K_ACT)
        xp = {t: project(xr[t], am, ac) for t in train_tasks}
        pp = torch.cat([xp[t] for t in train_tasks], 0)
        sm = pp.mean(0); ss = pp.std(0, unbiased=False).clamp_min(1e-6)
        xf = torch.cat([(xp[t] - sm) / ss for t in train_tasks], 0)
        yf = torch.cat([fv_proj[t].unsqueeze(0).expand(xp[t].shape[0], -1) for t in train_tasks], 0)
        xbar, ybar, ev, evec, c = ridge_eig_prep(xf, yf)
        return (am, ac, sm, ss, xbar, ybar, ev, evec, c, alpha[(role, L)])

    for L in LAYERS:
        maps = {role: fit(role, L) for role in roles}
        data = {}
        for lab, mr, ar, dm in POSITIONS:
            am, ac, sm, ss, xbar, ybar, ev, evec, c, a = maps[mr]
            data[lab] = {}
            for tk in COLORS:
                vp = (project(amb[lab][tk][:, L, :], am, ac) - sm) / ss
                dec = reconstruct(ridge_predict(vp, xbar, ybar, ev, evec, c, a), fv_mean, fv_comp)
                data[lab][tk] = ((dec @ fv_mag).cpu().numpy(), (dec @ fv_id).cpu().numpy())
        allx = np.concatenate([data[l][t][0] for l, *_ in POSITIONS for t in COLORS] +
                              [np.array([ref["magnitude"][0], ref["identity"][0]])])
        ally = np.concatenate([data[l][t][1] for l, *_ in POSITIONS for t in COLORS] +
                              [np.array([ref["magnitude"][1], ref["identity"][1]])])
        xlim, ylim = (allx.min(), allx.max()), (ally.min(), ally.max())
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        for k, (lab, *_ ) in enumerate(POSITIONS):
            ax = axes[k]
            for tk in COLORS:
                x, y = data[lab][tk]
                ax.scatter(x, y, s=12, alpha=0.55, color=COLORS[tk], edgecolors="none",
                           label=f"{tk} task" if k == 0 else None)
            # actual-FV reference markers (the train-task-head magnitude/identity FVs themselves)
            for tk in COLORS:
                ax.scatter(*ref[tk], marker="X", s=180, color=COLORS[tk], edgecolors="black",
                           linewidths=1.5, zorder=5, label=f"{tk} FV (actual)" if k == 0 else None)
            lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
            ax.plot([lo, hi], [lo, hi], ls=":", c="gray", lw=0.8)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_title(lab, fontsize=10); ax.set_xlabel("dec·mag FV"); ax.set_ylabel("dec·id FV")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4,
                   fontsize=9, markerscale=1.4, bbox_to_anchor=(0.5, -0.08))
        fig.suptitle(f"Layer {L}: decoded FV projected onto magnitude/identity FVs "
                     f"(train-task top-40 heads) — last 3 positions", fontsize=12)
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        out = FIG / f"decoded_varicl40_bytoken_L{L}_last3_multitask40.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
