#!/usr/bin/env python
"""Where does a head's value-channel payload direction appear in the residual stream?

For the task-specific top heads of one task (default: present-past top-10 by per-task CIE),
define the payload direction of head (L, H) as

    d_payload = unit( W_V^T @ m_hat ),   m_hat = unit(task-mean head activation z_bar)

i.e. the residual-stream direction whose presence at a token, when that token is attended,
moves the head's output along its own task-mean output direction. The value pathway has no
RoPE, so d_payload is exactly position-independent; at fixed attention weights,
m_hat . z = sum_t w_t * (d_payload_raw . a_t) with a_t = ln_1(x_t).

HARD GATE per head: the manually recomposed head output sum_t w_t * v_t at the cue token
must match the out_proj input captured from the model's own forward pass.

Outputs: an .npz with all N unit d_payload directions (+ gates, targets) and, for the top
head, the two stacked layer x token heatmaps (cos to d_payload / projection onto it) over
the RAW residual stream at all 29 layer boundaries — mirroring the d_content figure.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR
from src.utils.prompt_utils import create_prompt, load_dataset
from src.utils.varicl_utils import build_varicl_prompt_data

TRAIN_TASKS = [
    "national_parks", "english-spanish", "next_capital_letter", "commonsense_qa",
    "capitalize_last_letter", "country-capital", "english-french", "ag_news",
    "sentiment", "present-past", "person-occupation", "prev_item",
    "capitalize_second_letter", "lowercase_last_letter", "singular-plural",
    "person-sport", "park-country", "english-german", "person-instrument", "next_item",
]
DIV_RAMP = ["#8f1f1f", "#e98a6d", "#f0efec", "#6da7ec", "#0d366b"]


def parse_args():
    p = argparse.ArgumentParser(description="d_payload layer x token heatmaps (per-task top heads).")
    p.add_argument("--task", type=str, default="present-past")
    p.add_argument("--query_idx", type=int, default=21)
    p.add_argument("--n_heads", type=int, default=10,
                   help="How many of the task's own top-CIE heads to build d_payload for.")
    p.add_argument("--plot_head_rank", type=int, default=1,
                   help="Which task-rank head gets the layer x token figure (1 = top head).")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--aie_root", type=Path, default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_shots", type=int, default=1)
    p.add_argument("--max_shots", type=int, default=10)
    p.add_argument("--query_split", type=str, default="valid")
    p.add_argument("--demo_split", type=str, default="train")
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--output_dir", type=Path, default=FV_FORMATION_DIR / "attention_head_analysis/dpayload_layer_token")
    return p.parse_args()


def display_token(tok_str):
    if tok_str == "<|endoftext|>":
        return "<bos>"
    return tok_str.replace("Ġ", " ").replace("Ċ", "\\n")


def main():
    args = parse_args()
    args.prefixes = {"input": "Q:", "output": "A:", "instructions": ""}
    args.separators = {"input": "\n", "output": "\n\n", "instructions": ""}

    cie = torch.load(args.aie_root / args.task / f"{args.task}_cie_result.pt", weights_only=False)
    top_heads = [(int(l), int(h)) for l, h, _ in cie["top_heads"][: args.n_heads]]
    cie_scores = [float(s) for _, _, s in cie["top_heads"][: args.n_heads]]
    mean_acts = torch.load(
        args.aie_root / args.task / f"{args.task}_mean_head_activations_varicl.pt",
        weights_only=False,
    )  # (28, 16, 256) — covers ALL heads, so non-top-40 heads are available too
    print(f"{args.task} top-{args.n_heads} heads (per-task CIE): "
          + " ".join(f"L{l}H{h}" for l, h in top_heads))

    task_index = TRAIN_TASKS.index(args.task)
    dataset = load_dataset(args.task, root_data_dir=args.root_data_dir,
                           test_size=args.test_split, seed=args.seed)
    prompt_data = build_varicl_prompt_data(
        dataset, args, {"prepend_bos": False}, task_index=task_index,
        query_idx=args.query_idx, shuffle_labels=False, seed_base=args.seed,
    )
    prompt = create_prompt(prompt_data)
    target = prompt_data["query_target"]["output"]
    target = target[0] if isinstance(target, list) else target

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Loading model on CPU (float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    tokens_raw = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tokens_disp = [display_token(t) for t in tokens_raw]
    T = len(tokens_raw)

    # Capture each needed layer's out_proj INPUT (the merged pre-projection head outputs).
    layers_needed = sorted({l for l, _ in top_heads})
    captured = {}
    hooks = []
    for L in layers_needed:
        def make_hook(L):
            def hook(module, hook_args):
                captured[L] = hook_args[0].detach()
            return hook
        hooks.append(model.transformer.h[L].attn.out_proj.register_forward_pre_hook(make_hook(L)))
    with torch.no_grad():
        out = model(**inputs, output_attentions=True, output_hidden_states=True)
    for h_ in hooks:
        h_.remove()
    hidden = torch.stack([h[0] for h in out.hidden_states])  # (29, T, 4096)

    HD = 256
    d_rows, m_rows, gate_devs, align = [], [], [], []
    for (L, H) in top_heads:
        block = model.transformer.h[L]
        attn_mod = block.attn
        with torch.no_grad():
            a = block.ln_1(hidden[L])                                  # (T, 4096)
            v = attn_mod.v_proj(a).view(T, attn_mod.num_attention_heads, HD)
        w = out.attentions[L][0, H, -1, :]                             # (T,)
        z_manual = (w[:, None] * v[:, H]).sum(0)                       # (256,)
        z_actual = captured[L][0, -1, H * HD:(H + 1) * HD]
        dev = (z_manual - z_actual).abs().max().item() / max(1e-9, z_actual.abs().max().item())
        gate_devs.append(dev)
        if dev > 1e-3:
            raise RuntimeError(f"HARD-STOP: recomposed head output mismatch for L{L}H{H} "
                               f"(rel dev {dev:.2e}) — value pipeline wrong.")

        m_hat = mean_acts[L, H] / mean_acts[L, H].norm()               # target: unit task-mean z
        w_v = attn_mod.v_proj.weight[H * HD:(H + 1) * HD]              # (256, 4096)
        d = (w_v.T @ m_hat).detach()
        d_rows.append((d / d.norm()).to(torch.float64))
        m_rows.append(m_hat)
        # how aligned is THIS prompt's cue-token head output with the task-mean direction?
        align.append(float((z_actual / z_actual.norm()) @ m_hat))

    D = torch.stack(d_rows)
    print(f"gate: all {len(top_heads)} recomposed head outputs match out_proj inputs, "
          f"max rel dev {max(gate_devs):.2e}")
    print("cos(this prompt's cue head output, task-mean direction) per head:")
    print("  " + "  ".join(f"L{l}H{h}:{c:.2f}" for (l, h), c in zip(top_heads, align)))

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_all = f"{args.task.replace('-', '_')}_q{args.query_idx}"
    npz_all = out_dir / f"{stem_all}_dpayload_top{args.n_heads}.npz"
    np.savez(
        npz_all,
        D=D.numpy(), heads=np.array(top_heads), cie=np.array(cie_scores),
        m_hat=torch.stack(m_rows).numpy(), gate_rel_dev=np.array(gate_devs),
        cue_output_alignment=np.array(align),
        prompt=np.array(prompt), task=np.array(args.task), query_idx=np.array(args.query_idx),
    )
    print(f"Wrote {npz_all}")

    # ---- layer x token figure for the chosen head ----
    L, H = top_heads[args.plot_head_rank - 1]
    d_hat = D[args.plot_head_rank - 1].to(torch.float32)
    proj = (hidden @ d_hat).numpy()
    cosine = proj / (hidden.norm(dim=-1).numpy() + 1e-12)

    cmap = LinearSegmentedColormap.from_list("div_rb", DIV_RAMP)
    n_b = hidden.shape[0]
    fig_w = max(10.0, 0.21 * T + 2.8)
    fig_h = 2 * (0.24 * n_b) + 3.4
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=200, sharex=True)
    panels = [
        (cosine, np.abs(cosine).max(), "cos(residual, d_payload)", "cosine similarity"),
        (proj, np.abs(proj[:, 1:]).max(),
         "projection onto unit d_payload (color scale excludes <bos>; its column saturates)",
         "projection (resid norm units)"),
    ]
    for ax, (grid, vmax, title, cbar_label) in zip(axes, panels):
        mesh = ax.pcolormesh(grid, cmap=cmap, vmin=-vmax, vmax=vmax,
                             edgecolors="white", linewidth=0.3)
        ax.hlines([L, L + 1], 0, T, colors="#29291f", linestyles="dashed", linewidth=1.0)
        ax.text(1.2, L + 1.45, f"L{L}H{H} reads this row", fontsize=7, va="bottom",
                color="#29291f",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2))
        ax.set_yticks(np.arange(0, n_b, 2) + 0.5)
        ax.set_yticklabels([str(i) for i in range(0, n_b, 2)], fontsize=7)
        ax.set_ylabel("layer boundary (0 = embeddings)", fontsize=8, color="#454540")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=10, loc="left", color="#29291f")
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.02)
        cbar.set_label(cbar_label, fontsize=8, color="#454540")
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_visible(False)
    axes[1].set_xticks(np.arange(T) + 0.5)
    axes[1].set_xticklabels(tokens_disp, fontsize=6, rotation=90, family="monospace")
    axes[1].set_xlabel("prompt token", fontsize=9, color="#454540")
    fig.suptitle(
        f"Value-channel payload direction of L{L}H{H} in the raw residual stream (GPT-J)\n"
        f"{args.task}, 10-shot, query_idx {args.query_idx}; "
        f"d_payload = W_V^T @ unit(task-mean head activation), task rank #{args.plot_head_rank}",
        fontsize=11, x=0.01, ha="left", color="#29291f",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    png_path = out_dir / f"{stem_all}_L{L}H{H}_dpayload_layer_token.png"
    fig.savefig(png_path, bbox_inches="tight")
    print(f"Wrote {png_path}")

    # stdout: strongest payload carriers at the head's read layer
    read_cos = cosine[L]
    order = np.argsort(read_cos)[::-1][:8]
    print(f"\ntop tokens by cos(resid at boundary {L}, d_payload):")
    for p in order:
        print(f"  {p:3d} {tokens_disp[p]!r:<14s} cos {read_cos[p]:.3f}  proj {proj[L, p]:.2f}")


if __name__ == "__main__":
    main()
