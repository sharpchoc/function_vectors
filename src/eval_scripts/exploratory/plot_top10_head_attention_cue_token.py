#!/usr/bin/env python
"""Heatmap of where the top-10 varicl heads attend at the final cue token.

Rebuilds one exact variable-ICL prompt from the head-selection pipeline (same seeding as
compute_multitask_varicl_heads.py), runs a single CPU forward pass with output_attentions,
and plots each top-CIE head's softmax attention row at the final cue token (the position
where CIE head selection and the FV mean activations were measured): x = prompt tokens,
y = heads ordered by descending pooled CIE, color = attention weight. Saves one summary
PNG plus an .npz with the rows/tokens/head list so other views can be regenerated.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR
from src.utils.prompt_utils import create_prompt, load_dataset
from src.utils.varicl_utils import build_varicl_prompt_data

# Train-task order of the canonical varicl head-selection run (defines global task_index
# for prompt RNG seeding); matches artifacts/multitask_aie_heads_varicl metadata.
TRAIN_TASKS = [
    "national_parks", "english-spanish", "next_capital_letter", "commonsense_qa",
    "capitalize_last_letter", "country-capital", "english-french", "ag_news",
    "sentiment", "present-past", "person-occupation", "prev_item",
    "capitalize_second_letter", "lowercase_last_letter", "singular-plural",
    "person-sport", "park-country", "english-german", "person-instrument", "next_item",
]

# Sequential single-hue blue ramp, near-zero receding toward the surface.
SEQ_RAMP = ["#f7fafe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def parse_args():
    p = argparse.ArgumentParser(description="Top-10 varicl head attention at the final cue token.")
    p.add_argument("--task", type=str, default="present-past")
    p.add_argument("--query_idx", type=int, default=21)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--heads_artifact", type=Path,
                   default=ARTIFACTS_ROOT / "multitask_aie_heads_varicl" / "multitask_top_aie_heads.pt")
    p.add_argument("--n_heads_plot", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_shots", type=int, default=1)
    p.add_argument("--max_shots", type=int, default=10)
    p.add_argument("--query_split", type=str, default="valid")
    p.add_argument("--demo_split", type=str, default="train")
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--output_dir", type=Path,
                   default=FV_FORMATION_DIR / "attention_head_analysis/top10_head_attention_cue_token")
    return p.parse_args()


def display_token(tok_str):
    """Readable label for a GPT-2 BPE token: Ġ = leading space, Ċ = newline."""
    if tok_str == "<|endoftext|>":
        return "<bos>"
    out = tok_str.replace("Ġ", " ").replace("Ċ", "\\n")
    return out


def main():
    args = parse_args()
    args.prefixes = {"input": "Q:", "output": "A:", "instructions": ""}
    args.separators = {"input": "\n", "output": "\n\n", "instructions": ""}

    heads_result = torch.load(args.heads_artifact, weights_only=False)
    top_heads = heads_result["top_heads"][: args.n_heads_plot]  # already CIE-descending
    head_labels = [f"L{l}H{h} (CIE {s:.4f})" for l, h, s in top_heads]

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
    print("=" * 60 + f"\n{prompt}\n" + "=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Loading model on CPU (float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs["input_ids"][0]
    tokens_raw = tokenizer.convert_ids_to_tokens(token_ids)
    tokens_disp = [display_token(t) for t in tokens_raw]
    T = len(tokens_raw)
    print(f"{T} tokens; final cue token = {tokens_disp[-1]!r}")

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    # attentions: tuple of n_layers x (1, n_heads, T, T); take the final-row slice.
    attn = np.stack([
        out.attentions[l][0, h, -1, :].numpy() for l, h, _ in top_heads
    ])  # (n_heads_plot, T)
    row_sums = attn.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), f"attention rows do not sum to 1: {row_sums}"
    assert attn.shape == (len(top_heads), T)

    # Per-head top-5 attended tokens.
    print("\nTop-5 attended tokens per head (position: token = weight):")
    for label, row in zip(head_labels, attn):
        top5 = np.argsort(row)[::-1][:5]
        cells = ", ".join(f"{p}:{tokens_disp[p]!r}={row[p]:.3f}" for p in top5)
        print(f"  {label:22s} {cells}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task.replace('-', '_')}_q{args.query_idx}"

    npz_path = out_dir / f"{stem}_cue_attention.npz"
    np.savez(
        npz_path,
        attention=attn,
        tokens_raw=np.array(tokens_raw),
        tokens_display=np.array(tokens_disp),
        head_layer=np.array([l for l, _, _ in top_heads]),
        head_index=np.array([h for _, h, _ in top_heads]),
        head_cie=np.array([s for _, _, s in top_heads]),
        prompt=np.array(prompt),
        task=np.array(args.task),
        query_idx=np.array(args.query_idx),
        target=np.array(target),
    )
    print(f"\nWrote {npz_path}")

    # ---- heatmap ----
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    n_rows = len(top_heads)
    fig_w = max(10.0, 0.21 * T + 2.6)
    fig_h = 0.42 * n_rows + 2.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    mesh = ax.pcolormesh(
        attn, cmap=cmap, vmin=0.0, vmax=attn.max(),
        edgecolors="white", linewidth=0.4,
    )
    ax.invert_yaxis()  # rank 1 on top
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(head_labels, fontsize=8)
    ax.set_xticks(np.arange(T) + 0.5)
    ax.set_xticklabels(tokens_disp, fontsize=6, rotation=90, family="monospace")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("prompt token", fontsize=9, color="#454540")
    ax.set_title(
        f"Attention from the final cue token — top-{n_rows} varicl heads (GPT-J)\n"
        f"{args.task}, 10-shot, query_idx {args.query_idx}, target {target.strip()!r}; "
        f"rows ordered by pooled CIE",
        fontsize=10, loc="left", color="#29291f",
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label("attention weight", fontsize=8, color="#454540")
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    png_path = out_dir / f"{stem}_cue_attention.png"
    fig.savefig(png_path, bbox_inches="tight")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
