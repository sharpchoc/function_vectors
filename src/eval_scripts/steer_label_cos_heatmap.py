"""
Label-token -> query-final cosine-shift heatmaps (GPT-J-6B; loads model + baukit).

Extends the Stream E label-token steering (results/direction2_label_geometry/oneshot_steering/)
to a 2-D layer x layer map. For a fixed task pair and source->target direction:

  - Build OVERLAPPING paired 1-shot prompts (same construction as
    capture_and_grade_oneshot_paired.py): identical demo label w (shared single-token output) and
    identical query q (shared input); only the demo INPUT differs by function. So the demo-label
    token and the query-final token are byte-identical across the src/tgt prompts -- the steering
    signal is pure function context.

  - steer_vec(i) = mean_pairs[ act_tgt_label(i) - act_src_label(i) ]   (per residual layer i; src->tgt)

  - For each steering strength alpha and intervention layer i: inject alpha*steer_vec(i) at the
    SOURCE prompt's demo-label token at layer i, then at each read layer k read the source's
    QUERY-FINAL activation and measure how much closer it moved to the (unsteered) target's
    query-final activation:
        baseline_cos(k)  = cos( src_final(k),        tgt_final(k) )
        steered_cos(i,k) = cos( steered_src_final(k), tgt_final(k) )
        cell(i,k) = mean_pairs[ steered_cos(i,k) - baseline_cos(k) ]

  - One heatmap per (task, alpha). x = intervention layer i, y = read layer k.

Layers = 29 residual entries INCLUDING the embedding (entry 0 = transformer.drop output,
entries 1..28 = transformer.h.{0..27} outputs). The embedding row/col and earliest layers are
~0 by construction (label & query-final tokens identical across functions); reads are downstream
of the injection so the lower triangle (k <= i) is ~0.

baukit is imported inside main (precedent: steer_label_to_query.py); the baukit-free position
helpers are inlined so the module top stays import-safe.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt, get_token_meta_labels
from utils.paths import LABEL_GEOMETRY_DIR

# task_pair -> (f1, f2)
TASK_PAIRS = {
    "antonym_synonym": ("antonym", "synonym"),
    "next_number_digits_prev_number_digits": ("next_number_digits", "prev_number_digits"),
}
# task_pair -> (source task, target task)  [push source toward target]
SRC_TGT = {
    "antonym_synonym": ("antonym", "synonym"),                       # push antonym -> synonym
    "next_number_digits_prev_number_digits": ("prev_number_digits", "next_number_digits"),  # prev -> next
}

PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}

# --- inlined verbatim (baukit-free) from extract_residual_stream_activations.py ---
LABEL_TOKEN_RE = re.compile(r"^demonstration_(\d+)_label_token$")


def make_token_record(token_role, icl_example_index, token):
    token_position, token_text, token_label = token
    return {"token_role": token_role, "icl_example_index": icl_example_index,
            "token_position": int(token_position), "token_text": token_text, "token_label": token_label}


def selected_token_records(token_labels):
    tokens_by_position = {int(p): (p, t, l) for p, t, l in token_labels}
    label_groups = {}
    for p, t, l in token_labels:
        m = LABEL_TOKEN_RE.match(l)
        if m:
            label_groups.setdefault(int(m.group(1)), []).append((p, t, l))
    records = []
    for icl in sorted(label_groups):
        lab = sorted(label_groups[icl], key=lambda x: x[0])
        pre_pos = int(lab[0][0]) - 1
        if pre_pos < 0 or pre_pos not in tokens_by_position:
            raise ValueError(f"no pre-label token for ICL {icl}")
        records.extend([
            make_token_record("pre_label_token", icl, tokens_by_position[pre_pos]),
            make_token_record("first_label_token", icl, lab[0]),
            make_token_record("last_label_token", icl, lab[-1]),
            make_token_record("label_token", icl, lab[-1]),
        ])
    final_cands = [x for x in token_labels if x[2] == "query_predictive_token"]
    final_token = max(final_cands, key=lambda x: x[0]) if final_cands else token_labels[-1]
    records.extend([make_token_record("last_prompt_token", None, final_token),
                    make_token_record("final_token", None, final_token)])
    return records


def extract_positions(token_labels):
    """(demo_label_idx = last_label_token@icl 1, query_final_idx = last_prompt_token@None)."""
    recs = selected_token_records(token_labels)
    label_idx = final_idx = None
    for r in recs:
        if r["token_role"] == "last_label_token" and r["icl_example_index"] == 1:
            label_idx = r["token_position"]
        elif r["token_role"] == "last_prompt_token" and r["icl_example_index"] is None:
            final_idx = r["token_position"]
    if label_idx is None or final_idx is None:
        raise ValueError("could not derive label/final positions")
    return label_idx, final_idx
# ----------------------------------------------------------------------------------


def stable_seed(*parts):
    import hashlib
    d = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(d[:8], "little") % (2 ** 32)


def stable_rng(*parts):
    return np.random.default_rng(stable_seed(*parts))


def load_task(root, t):
    recs = json.load(open(Path(root) / "abstractive" / f"{t}.json"))
    o2i, i2o = defaultdict(set), {}
    for r in recs:
        inp, out = str(r["input"]).strip(), str(r["output"]).strip()
        o2i[out].add(inp)
        i2o[inp] = out
    return {k: sorted(v) for k, v in o2i.items()}, i2o


def build_prompt(demo_input, demo_output, query_input):
    pd = word_pairs_to_prompt_data(
        {"input": [demo_input], "output": [demo_output]},
        query_target_pair={"input": query_input, "output": query_input},
        prepend_bos_token=False, prefixes=PREFIXES, separators=SEPARATORS, prepend_space=True,
    )
    return pd


def parse_args():
    p = argparse.ArgumentParser(description="Label-token -> query-final cosine-shift 29x29 heatmaps.")
    p.add_argument("--task_pair", choices=sorted(TASK_PAIRS), default="antonym_synonym")
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0])
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Subset of INTERVENTION layers (0..n) to sweep; default = all 29. Reads are always all 29.")
    p.add_argument("--max_pairs", type=int, default=None, help="Cap number of prompt pairs (smoke tests).")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--output_root", type=str,
                   default=str(LABEL_GEOMETRY_DIR / "oneshot_label_intervention_cos_heatmap"))
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    from baukit import TraceDict  # noqa: local import (model-side dep)

    f1, f2 = TASK_PAIRS[args.task_pair]
    src_task, tgt_task = SRC_TGT[args.task_pair]
    print(f"task_pair={args.task_pair}  f1={f1} f2={f2}  steer {src_task} -> {tgt_task}")

    o2i_f1, i2o_f1 = load_task(args.root_data_dir, f1)
    o2i_f2, i2o_f2 = load_task(args.root_data_dir, f2)

    print("Loading model...")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision)
    model.eval()
    device = args.device
    dtype = next(model.parameters()).dtype
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 29 residual entries: 0 = embedding (transformer.drop), 1..28 = transformer.h.{0..27}
    block_names = model_config["layer_hook_names"]            # ['transformer.h.0', ...]
    n_blocks = model_config["n_layers"]
    assert n_blocks == len(block_names) == 28, f"expected 28 GPT-J blocks, got {n_blocks}"
    emb_name = "transformer.drop"
    layer_names = [emb_name] + list(block_names)              # 29 ordered residual entries
    n_layers = len(layer_names)                              # 29
    name2idx = {nm: i for i, nm in enumerate(layer_names)}
    resid_dim = model_config["resid_dim"]

    def single(w):
        return len(tokenizer(" " + w).input_ids) == 1

    shared_out = sorted(set(o2i_f1) & set(o2i_f2))
    label_words = [w for w in shared_out if single(w)]
    shared_in = sorted(set(i2o_f1) & set(i2o_f2))
    query_pool = list(shared_in)
    print(f"label words (shared single-tok output): {len(label_words)}; query pool: {len(query_pool)}")

    is_src_f1 = (src_task == f1)

    # ---- build prompt pairs (deterministic, identical construction to the capture) ----
    src_ids_list, tgt_ids_list = [], []     # token id lists (unpadded)
    src_lab_pos, tgt_lab_pos = [], []        # demo-label index (unpadded)
    pair_meta = []
    for w in label_words:
        rng = stable_rng(args.seed, args.task_pair, w)
        d1 = str(rng.choice(o2i_f1[w]))
        d2 = str(rng.choice(o2i_f2[w]))
        forbidden = {w, d1, d2}
        cand = [q for q in query_pool if q not in forbidden]
        if not cand:
            continue
        q = str(rng.choice(cand))
        src_demo_in = d1 if is_src_f1 else d2
        tgt_demo_in = d1 if (tgt_task == f1) else d2
        out = {}
        for tag, demo_in in (("src", src_demo_in), ("tgt", tgt_demo_in)):
            pd = build_prompt(demo_in, w, q)
            token_labels, prompt_string = get_token_meta_labels(
                pd, tokenizer, query=q, prepend_bos=model_config["prepend_bos"])
            lab_idx, fin_idx = extract_positions(token_labels)
            ids = tokenizer(prompt_string).input_ids
            assert fin_idx == len(ids) - 1, "query-final not last token"
            out[tag] = (ids, lab_idx)
        # invariant: label token + query-final token byte-identical across src/tgt
        s_ids, s_lab = out["src"]; t_ids, t_lab = out["tgt"]
        assert s_ids[s_lab] == t_ids[t_lab], f"label token differs across functions (w={w!r})"
        assert s_ids[-1] == t_ids[-1], f"query-final token differs across functions (w={w!r})"
        src_ids_list.append(s_ids); src_lab_pos.append(s_lab)
        tgt_ids_list.append(t_ids); tgt_lab_pos.append(t_lab)
        pair_meta.append({"output_word": w, "query": q, "src_demo": src_demo_in, "tgt_demo": tgt_demo_in})
        if args.max_pairs is not None and len(pair_meta) >= args.max_pairs:
            break
    n_pairs = len(pair_meta)
    print(f"built {n_pairs} prompt pairs")

    inter_layers = sorted(args.layers) if args.layers is not None else list(range(n_layers))
    pad_id = tokenizer.pad_token_id

    def make_chunks(ids_list, lab_pos):
        """Left-padded chunk dicts with per-row padded label index; final idx = -1."""
        chunks = []
        for s in range(0, len(ids_list), args.batch_size):
            sub_ids = ids_list[s:s + args.batch_size]
            sub_lab = lab_pos[s:s + args.batch_size]
            mx = max(len(x) for x in sub_ids)
            inp = torch.full((len(sub_ids), mx), pad_id, dtype=torch.long)
            att = torch.zeros((len(sub_ids), mx), dtype=torch.long)
            lab = torch.zeros(len(sub_ids), dtype=torch.long)
            for r, ids in enumerate(sub_ids):
                pad = mx - len(ids)
                inp[r, pad:] = torch.tensor(ids, dtype=torch.long)
                att[r, pad:] = 1
                lab[r] = pad + sub_lab[r]
            chunks.append({"input_ids": inp.to(device), "attention_mask": att.to(device),
                           "label_idx": lab.to(device), "n": len(sub_ids)})
        return chunks

    src_chunks = make_chunks(src_ids_list, src_lab_pos)
    tgt_chunks = make_chunks(tgt_ids_list, tgt_lab_pos)

    def read_entries(td, want_label):
        """Return (final[B,29,D], label[B,29,D] or None) for one traced forward (on GPU, fp16)."""
        B = td[layer_names[0]].output
        B = (B[0] if isinstance(B, tuple) else B).shape[0]
        fin = torch.empty((B, n_layers, resid_dim), device=device, dtype=dtype)
        lab = torch.empty((B, n_layers, resid_dim), device=device, dtype=dtype) if want_label else None
        rows = torch.arange(B, device=device)
        return fin, lab, rows

    def make_edit_hook(edit_name, add_vec, rows0, lab_idx):
        # EXACT 2-arg (output, layer_name) closure -- baukit's invoke_with_optional_args
        # mis-binds any extra params positionally (see DECISIONS 2026-06-11). Capture the rest
        # of the state via the enclosing scope.
        def hook(output, layer_name):
            if layer_name != edit_name:
                return output
            if isinstance(output, tuple):
                output[0][rows0, lab_idx] += add_vec
                return output
            output[rows0, lab_idx] += add_vec
            return output
        return hook

    def capture(chunks, want_label, edit_name=None, add_vec=None):
        """Run forwards; collect final (and optionally label) acts at all 29 entries."""
        fins, labs = [], []
        for ch in chunks:
            if edit_name is not None:
                rows0 = torch.arange(ch["n"], device=device)
                hook = make_edit_hook(edit_name, add_vec, rows0, ch["label_idx"])
                cm = TraceDict(model, layers=layer_names, edit_output=hook, retain_output=True)
            else:
                cm = TraceDict(model, layers=layer_names, retain_output=True)
            with cm as td:
                model(input_ids=ch["input_ids"], attention_mask=ch["attention_mask"])
                fin, lab, rows = read_entries(td, want_label)
                for li, nm in enumerate(layer_names):
                    out = td[nm].output
                    out = out[0] if isinstance(out, tuple) else out
                    fin[:, li, :] = out[:, -1, :]
                    if want_label:
                        lab[:, li, :] = out[rows, ch["label_idx"], :]
            fins.append(fin)
            if want_label:
                labs.append(lab)
        fin_all = torch.cat(fins, 0)
        lab_all = torch.cat(labs, 0) if want_label else None
        return fin_all, lab_all

    # ---- unsteered passes ----
    print("unsteered src pass...")
    src_final, src_label = capture(src_chunks, want_label=True)
    print("unsteered tgt pass...")
    tgt_final, tgt_label = capture(tgt_chunks, want_label=True)

    # steer_vec(i) = mean_pairs[ tgt_label(i) - src_label(i) ]   (src -> tgt direction)
    steer_vec = (tgt_label.float() - src_label.float()).mean(0)            # [29, D]
    steer_norms = torch.linalg.norm(steer_vec, dim=-1).cpu().numpy()        # [29]
    # baseline cos per pair per read layer
    baseline_cos = F.cosine_similarity(src_final.float(), tgt_final.float(), dim=-1)  # [n_pairs, 29]
    mean_baseline = baseline_cos.mean(0).cpu().numpy()                                 # [29]
    tgt_final_f = tgt_final.float()

    assert torch.isfinite(src_final).all() and torch.isfinite(tgt_final).all(), "non-finite acts"
    print(f"steer_vec norms: emb={steer_norms[0]:.3f} L6={steer_norms[6]:.3f} "
          f"L12={steer_norms[12]:.3f} L28={steer_norms[28]:.3f}")
    print(f"mean baseline cos: emb={mean_baseline[0]:.4f} L6={mean_baseline[6]:.4f} L28={mean_baseline[28]:.4f}")

    out_dir = Path(args.output_root)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    summary = {"task_pair": args.task_pair, "src_task": src_task, "tgt_task": tgt_task,
               "n_pairs": n_pairs, "n_layers": n_layers, "alphas": args.alphas,
               "intervention_layers": inter_layers,
               "steer_vec_norm_by_layer": steer_norms.tolist(),
               "mean_baseline_cos_by_read_layer": mean_baseline.tolist(),
               "embedding_entry": "transformer.drop (residual layer 0); layers 1..28 = block outputs",
               "grids": {}}

    for alpha in args.alphas:
        # grid[read_layer k, intervention_layer i] = mean_pairs(steered_cos - baseline_cos)
        grid = np.full((n_layers, n_layers), np.nan, dtype=np.float64)
        for i in inter_layers:
            add_vec = (alpha * steer_vec[i]).to(device=device, dtype=dtype)
            sfin, _ = capture(src_chunks, want_label=False, edit_name=layer_names[i], add_vec=add_vec)
            steered_cos = F.cosine_similarity(sfin.float(), tgt_final_f, dim=-1)   # [n_pairs, 29]
            shift = (steered_cos - baseline_cos).mean(0).cpu().numpy()              # [29]
            grid[:, i] = shift
            print(f"  alpha={alpha} intervene L{i:>2}: max read-shift {np.nanmax(shift):+.4f} "
                  f"@k={int(np.nanargmax(shift))}")
        assert np.isfinite(grid[:, inter_layers]).all(), "non-finite grid cell"

        tag = f"{args.task_pair}_alpha{alpha:g}"
        np.save(out_dir / f"{tag}_grid.npy", grid)
        np.savetxt(out_dir / f"{tag}_grid.csv", grid, delimiter=",",
                   header="rows=read_layer(0..28), cols=intervention_layer(0..28)", comments="")
        # peak cell
        fk, fi = np.unravel_index(np.nanargmax(grid), grid.shape)
        summary["grids"][tag] = {"alpha": alpha, "npy": f"{tag}_grid.npy",
                                 "peak_shift": float(grid[fk, fi]),
                                 "peak_intervention_layer": int(fi), "peak_read_layer": int(fk)}

        # ---- render ----
        vmax = float(np.nanmax(np.abs(grid))) or 1e-6
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        im = ax.imshow(grid, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xlabel("intervention layer (label token)")
        ax.set_ylabel("read layer (query-final token)")
        ax.set_title(f"steer {src_task}→{tgt_task}  (α={alpha:g}, n={n_pairs})\n"
                     f"mean Δcos(steered src_final, tgt_final)", fontsize=10)
        ax.plot([0, n_layers - 1], [0, n_layers - 1], color="k", lw=0.6, ls=":", alpha=0.5)
        fig.colorbar(im, ax=ax, label="steered − baseline cosine")
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / f"{tag}_cos_shift_heatmap.png", dpi=140)
        plt.close(fig)
        print(f"  saved heatmap {tag}: peak {grid[fk, fi]:+.4f} @ intervene L{fi} read L{fk}")

    with open(out_dir / f"{args.task_pair}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()
