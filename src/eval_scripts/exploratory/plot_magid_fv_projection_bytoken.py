"""
For a single layer (default 20), scatter every captured token position on the SAME two FV
axes (x = projection onto magnitude FV, y = projection onto identity FV; top-20 constrained
FVs, unit-normalized). One subplot per token position, in prompt order:
  demo1 {pre, first-label, last-label}, demo2 {...}, ... demo4 {...}, then query (final pre-label).
Points colored by task (magnitude vs identity). Shared axis limits across subplots.

Note: these 3+1+1 prompts have 4 ICL examples (demos 1-3 = overlap, demo 4 = differentiator).
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ACT_DIR = os.path.join(HERE, "..", "..", "..", "artifacts", "magnitude_identity_activations", "gpt-j-6b")
FV_DIR = os.path.join(HERE, "..", "..", "..", "artifacts", "gptj_fv_ambiguous_constrained_top20")
FIG_DIR = os.path.join(HERE, "..", "..", "..", "results", "exploratory", "direction1_ambiguous", "figures")
LAYER = 20
COLORS = {"magnitude": "tab:red", "identity": "tab:blue"}

ROLE_SHORT = {"pre_label_token": "pre", "first_label_token": "first-lbl", "last_label_token": "last-lbl"}


def load_fv(task):
    fv = torch.load(os.path.join(FV_DIR, task, f"{task}_function_vector.pt"),
                    weights_only=False)["function_vector"].squeeze().float()
    return fv / fv.norm()


def load_task(task):
    d = torch.load(os.path.join(ACT_DIR, f"{task}.pt"), weights_only=False)
    return d["activations"].float(), d["metadata"]


def main():
    fv_mag, fv_id = load_fv("magnitude"), load_fv("identity")
    acts = {t: load_task(t) for t in ["magnitude", "identity"]}

    # position order: demos 1..4 x {pre, first, last}, then query
    positions = []
    for demo in [1, 2, 3, 4]:
        for role in ["pre_label_token", "first_label_token", "last_label_token"]:
            positions.append((demo, role, f"demo{demo} {ROLE_SHORT[role]}"
                              + (" *diff*" if demo == 4 else "")))
    positions.append((None, "query_predictive_token", "query (final pre-label)"))

    def proj(task, demo, role):
        A, M = acts[task]
        idx = [i for i, m in enumerate(M)
               if m["token_role"] == role and m["demo_index"] == demo]
        v = A[idx][:, LAYER, :]                              # (n, 4096)
        return (v @ fv_mag).numpy(), (v @ fv_id).numpy()

    # precompute for shared limits
    P = {(d, r): {t: proj(t, d, r) for t in COLORS} for d, r, _ in positions}
    allx = np.concatenate([P[(d, r)][t][0] for d, r, _ in positions for t in COLORS])
    ally = np.concatenate([P[(d, r)][t][1] for d, r, _ in positions for t in COLORS])
    xlim = (allx.min() - 2, allx.max() + 2)
    ylim = (ally.min() - 2, ally.max() + 2)

    ncols = 4
    nrows = int(np.ceil(len(positions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)
    for k, (d, r, title) in enumerate(positions):
        ax = axes[k // ncols][k % ncols]
        for t in COLORS:
            x, y = P[(d, r)][t]
            ax.scatter(x, y, s=9, alpha=0.5, color=COLORS[t],
                       label=f"{t} task" if k == 0 else None, edgecolors="none")
        ax.plot(xlim, xlim, ls=":", c="gray", lw=0.8)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("proj · magnitude FV", fontsize=6)
        ax.set_ylabel("proj · identity FV", fontsize=6)
    for j in range(len(positions), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=11,
               markerscale=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Layer {LAYER}: activations by token position, projected onto magnitude vs "
                 f"identity constrained FVs (top-20) — colored by task", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"magid_bytoken_L{LAYER}_fv_projection_decimals.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

    # sanity: overlap-demo positions should be identical across tasks (paired, token-identical
    # prefix), diverging only at demo4 label and the query.
    print("\nmean |Δ| between tasks (paired rows), x-proj at layer", LAYER, ":")
    for d, r, title in positions:
        dx = np.abs(P[(d, r)]["magnitude"][0] - P[(d, r)]["identity"][0]).mean()
        print(f"  {title:24s} {dx:.4f}")


if __name__ == "__main__":
    main()
