"""
2D per-layer scatter: each query-predictive-token residual activation (from the
magnitude/identity 3+1+1 capture) projected onto BOTH constrained ambiguous-aware FVs
(top-20 heads). x = projection onto magnitude FV, y = projection onto identity FV.
Points colored by which task the prompt was (magnitude vs identity). One subplot per layer.

Projection = dot product with the UNIT-normalized FV (signed scalar projection onto the
direction), so the two axes are on comparable scales.
"""
import os, json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ACT_DIR = os.path.join(HERE, "..", "..", "results", "magnitude_identity_activations", "gpt-j-6b")
FV_DIR = os.path.join(HERE, "..", "..", "results", "gptj_fv_ambiguous_constrained_top20")
FIG_DIR = os.path.join(HERE, "..", "..", "figures")
ROLE = "query_predictive_token"

COLORS = {"magnitude": "tab:red", "identity": "tab:blue"}


def load_fv(task):
    fv = torch.load(os.path.join(FV_DIR, task, f"{task}_function_vector.pt"),
                    weights_only=False)["function_vector"].squeeze().float()
    return fv / fv.norm()                                   # unit direction


def load_query_acts(task):
    d = torch.load(os.path.join(ACT_DIR, f"{task}.pt"), weights_only=False)
    meta = d["metadata"]
    idx = [i for i, m in enumerate(meta) if m["token_role"] == ROLE]
    acts = d["activations"][idx].float()                    # (n, n_layers, 4096)
    return acts


def main():
    fv_mag, fv_id = load_fv("magnitude"), load_fv("identity")
    data = {t: load_query_acts(t) for t in ["magnitude", "identity"]}
    n_layers = next(iter(data.values())).shape[1]           # 29 (embed + 28)

    # projections: proj[task] -> (x onto mag FV, y onto id FV), shape (n, n_layers)
    proj = {}
    for t, acts in data.items():
        proj[t] = (acts @ fv_mag, acts @ fv_id)             # each (n, n_layers)

    ncols = 6
    nrows = int(np.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows), squeeze=False)
    for L in range(n_layers):
        ax = axes[L // ncols][L % ncols]
        for t in ["magnitude", "identity"]:
            x = proj[t][0][:, L].numpy()
            y = proj[t][1][:, L].numpy()
            ax.scatter(x, y, s=8, alpha=0.5, color=COLORS[t],
                       label=f"{t} task" if L == 0 else None, edgecolors="none")
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], ls=":", c="gray", lw=0.8)   # x=y reference
        layer_name = "embed" if L == 0 else f"L{L}"
        ax.set_title(layer_name, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("proj · magnitude FV", fontsize=6)
        ax.set_ylabel("proj · identity FV", fontsize=6)
    for j in range(n_layers, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=11,
               markerscale=2, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Query-predictive activation projected onto magnitude vs identity constrained FVs "
                 "(top-20), per layer — colored by task", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "magid_query_fv_projection_by_layer.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    print(f"points/task: magnitude={data['magnitude'].shape[0]} identity={data['identity'].shape[0]} | layers={n_layers}")


if __name__ == "__main__":
    main()
