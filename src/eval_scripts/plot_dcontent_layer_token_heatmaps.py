#!/usr/bin/env python
"""Where does a head's position-free content direction appear in the residual stream?

For one exact varicl prompt (present-past q21, 10-shot) and one head (default L9H14), build
the head's position-free content direction

    d_content = W_K_pass^T @ q_pass

where q_pass is the pass-block (non-rotary, dims 64..255) slice of the cue token's query and
W_K_pass the matching key-matrix rows. Because GPT-J only rotates dims 0..63 of each head
(rotary_dim=64), the pass-block contribution to the pre-softmax score is position-independent:
score_t = [rotary term] + d_content . LN(x_t) / 16.

The script hard-gates on reproducing the model's own attention row for the head from a manual
q/k pipeline (proves head slicing + rotary handling), then plots two stacked heatmaps over the
RAW residual stream hidden_states (29 layer boundaries x 97 tokens): cosine similarity to
d_content and scalar projection onto unit d_content. One summary PNG + a regenerable .npz.
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

from src.utils.paths import FV_FORMATION_DIR
from src.utils.prompt_utils import create_prompt, load_dataset
from src.utils.varicl_utils import build_varicl_prompt_data

TRAIN_TASKS = [
    "national_parks", "english-spanish", "next_capital_letter", "commonsense_qa",
    "capitalize_last_letter", "country-capital", "english-french", "ag_news",
    "sentiment", "present-past", "person-occupation", "prev_item",
    "capitalize_second_letter", "lowercase_last_letter", "singular-plural",
    "person-sport", "park-country", "english-german", "person-instrument", "next_item",
]

# Diverging blue <-> red with neutral gray midpoint (positive = blue, matching the
# sequential blue of the attention figure; negative = red).
DIV_RAMP = ["#8f1f1f", "#e98a6d", "#f0efec", "#6da7ec", "#0d366b"]


def parse_args():
    p = argparse.ArgumentParser(description="d_content layer x token heatmaps for one head.")
    p.add_argument("--task", type=str, default="present-past")
    p.add_argument("--query_idx", type=int, default=21)
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--head", type=int, default=14)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_shots", type=int, default=1)
    p.add_argument("--max_shots", type=int, default=10)
    p.add_argument("--query_split", type=str, default="valid")
    p.add_argument("--demo_split", type=str, default="train")
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--output_dir", type=Path, default=FV_FORMATION_DIR / "attention_head_analysis/dcontent_layer_token")
    return p.parse_args()


def display_token(tok_str):
    if tok_str == "<|endoftext|>":
        return "<bos>"
    return tok_str.replace("Ġ", " ").replace("Ċ", "\\n")


def main():
    args = parse_args()
    args.prefixes = {"input": "Q:", "output": "A:", "instructions": ""}
    args.separators = {"input": "\n", "output": "\n\n", "instructions": ""}
    L, H = args.layer, args.head

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
    print(f"Prompt rebuilt for {args.task} query_idx={args.query_idx}, target={target!r}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gptj.modeling_gptj import apply_rotary_pos_emb

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
    print(f"{T} tokens; final cue token = {tokens_disp[-1]!r}")

    with torch.no_grad():
        out = model(**inputs, output_attentions=True, output_hidden_states=True)
    hidden = torch.stack([h[0] for h in out.hidden_states])  # (29, T, 4096)
    attn_row_model = out.attentions[L][0, H, -1, :]           # (T,)

    # ---- manual q/k pipeline for (L, H) ----
    block = model.transformer.h[L]
    attn_mod = block.attn
    head_dim = attn_mod.head_dim                      # 256
    rot = attn_mod.rotary_dim                         # 64
    x_in = hidden[L]                                  # raw resid input to block L
    with torch.no_grad():
        a = block.ln_1(x_in)                          # (T, 4096)
        q = attn_mod.q_proj(a).view(T, attn_mod.num_attention_heads, head_dim)
        k = attn_mod.k_proj(a).view(T, attn_mod.num_attention_heads, head_dim)

    sincos = attn_mod.embed_positions[:T].unsqueeze(0)          # (1, T, rot)
    sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    q_rot = apply_rotary_pos_emb(q[None, :, :, :rot], sin, cos)[0]
    k_rot = apply_rotary_pos_emb(k[None, :, :, :rot], sin, cos)[0]

    q_vec_rot = q_rot[-1, H]                          # (64,)  rotated rotary block of cue query
    q_vec_pass = q[-1, H, rot:]                       # (192,) pass block (never rotated)
    k_head_rot = k_rot[:, H]                          # (T, 64)
    k_head_pass = k[:, H, rot:]                       # (T, 192)

    scale = attn_mod.scale_attn                       # sqrt(256) = 16
    rot_term = (k_head_rot @ q_vec_rot) / scale       # position-dependent term
    content_term = (k_head_pass @ q_vec_pass) / scale # position-independent term
    scores = rot_term + content_term
    attn_row_manual = torch.softmax(scores, dim=-1)

    # HARD GATE: manual pipeline must reproduce the model's attention row.
    max_dev = (attn_row_manual - attn_row_model).abs().max().item()
    print(f"Gate: manual vs model attention row for L{L}H{H}: max|dev| = {max_dev:.2e}")
    if max_dev > 1e-4:
        raise RuntimeError(
            f"HARD-STOP: manual q/k pipeline does not reproduce the model attention row "
            f"(max dev {max_dev:.2e} > 1e-4). Head slicing / rotary handling is wrong; "
            f"nothing downstream can be trusted."
        )

    # d_content: rows of k_proj.weight for this head's pass block, pulled back by q_pass.
    w_k = attn_mod.k_proj.weight                      # (4096, 4096), row i = output dim i
    w_k_pass = w_k[H * head_dim + rot: (H + 1) * head_dim]   # (192, 4096)
    d_content = (w_k_pass.T @ q_vec_pass).detach()    # (4096,)
    d_hat = d_content / d_content.norm()

    # Cross-check: content term recomputed via d_content on the LN'd stream.
    content_via_d = (a @ d_content) / scale
    assert torch.allclose(content_via_d, content_term, atol=1e-3), \
        f"d_content cross-check failed: max dev {(content_via_d - content_term).abs().max():.2e}"

    # Score decomposition at the read layer for the top-attended tokens.
    print(f"\nScore decomposition at L{L} (content = position-free, rotary = positional):")
    order = torch.argsort(attn_row_model, descending=True)[:8]
    print(f"  {'pos':>4s} {'token':<14s} {'attn':>6s} {'score':>8s} {'content':>8s} {'rotary':>8s}")
    for p in order.tolist():
        print(f"  {p:4d} {tokens_disp[p]!r:<14s} {attn_row_model[p]:6.3f} "
              f"{scores[p]:8.3f} {content_term[p]:8.3f} {rot_term[p]:8.3f}")

    # ---- grids over the RAW residual stream at all 29 layer boundaries ----
    proj = (hidden @ d_hat).numpy()                                   # (29, T)
    cosine = proj / (hidden.norm(dim=-1).numpy() + 1e-12)             # since proj = |x| cos
    assert cosine.shape == proj.shape == (hidden.shape[0], T)
    assert np.abs(cosine).max() <= 1.0 + 1e-6

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task.replace('-', '_')}_q{args.query_idx}_L{L}H{H}"
    npz_path = out_dir / f"{stem}_dcontent_layer_token.npz"
    np.savez(
        npz_path,
        cosine=cosine, projection=proj,
        d_content=d_content.numpy(), q_head=torch.cat([q_vec_rot, q_vec_pass]).numpy(),
        attention_row=attn_row_model.numpy(),
        content_term=content_term.numpy(), rotary_term=rot_term.numpy(),
        tokens_raw=np.array(tokens_raw), tokens_display=np.array(tokens_disp),
        layer=np.array(L), head=np.array(H),
        prompt=np.array(prompt), task=np.array(args.task),
        query_idx=np.array(args.query_idx), target=np.array(target),
    )
    print(f"\nWrote {npz_path}")

    # ---- figure: two stacked heatmaps ----
    cmap = LinearSegmentedColormap.from_list("div_rb", DIV_RAMP)
    n_layers_b = hidden.shape[0]
    fig_w = max(10.0, 0.21 * T + 2.8)
    fig_h = 2 * (0.24 * n_layers_b) + 3.4
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=200, sharex=True)
    # <bos> is a residual-norm outlier (~10-100x other tokens); scale the projection
    # panel from the non-BOS columns and let the BOS column saturate.
    panels = [
        (cosine, np.abs(cosine).max(), "cos(residual, d_content)", "cosine similarity"),
        (proj, np.abs(proj[:, 1:]).max(),
         "projection onto unit d_content (color scale excludes <bos>; its column saturates)",
         "projection (resid norm units)"),
    ]
    for ax, (grid, vmax, title, cbar_label) in zip(axes, panels):
        mesh = ax.pcolormesh(grid, cmap=cmap, vmin=-vmax, vmax=vmax,
                             edgecolors="white", linewidth=0.3)
        # Row L is hidden_states[L] = the raw resid the head reads (through ln_1).
        ax.hlines([L, L + 1], 0, T, colors="#29291f", linestyles="dashed", linewidth=1.0)
        ax.text(1.2, L + 1.45, f"L{L}H{H} reads this row", fontsize=7, va="bottom",
                color="#29291f",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2))
        ax.set_yticks(np.arange(0, n_layers_b, 2) + 0.5)
        ax.set_yticklabels([str(i) for i in range(0, n_layers_b, 2)], fontsize=7)
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
        f"Position-free content direction of L{L}H{H} in the raw residual stream (GPT-J)\n"
        f"{args.task}, 10-shot, query_idx {args.query_idx}; d_content = W_K_pass^T q_pass "
        f"from the final cue token's query",
        fontsize=11, x=0.01, ha="left", color="#29291f",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    png_path = out_dir / f"{stem}_dcontent_layer_token.png"
    fig.savefig(png_path, bbox_inches="tight")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
