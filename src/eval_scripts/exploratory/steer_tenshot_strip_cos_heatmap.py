"""
TEN-shot intervene-token STRIP cosine-shift heatmaps (GPT-J-6B; loads model + baukit).

Extends steer_twoshot_tokenpair_cos_heatmap.py to 10-shot ICL with a DIFFERENT pairing: we do NOT
enforce matched/overlapping label tokens. For each task pair (f1,f2) and pair index p we build two
10-shot prompts that share ONLY the query q (⇒ the final predictive "A:" token is byte-identical);
the 10 demos are independently random per function (unmatched inputs & labels).

Per residual layer ℓ and each of the 30 demo tokens t (input / pre-label / last-label × 10 demos):
    steer_vec(t, ℓ) = mean_pairs[ act_tgt(t, ℓ) − act_src(t, ℓ) ]           (src→tgt direction)
i.e. the difference of the two tasks' MEAN activations at that structural slot (demos unmatched ⇒ this
carries lexical+function content). We inject α·steer_vec(t_i, i) at t_i's position in the SOURCE prompt
at layer i (single point-edit; the forward recomputes everything downstream) and READ only at the query
final token qfinal, at every read layer k, computing the DIRECTION-ALIGNMENT cosine ("dircos", metric of
record since 2026-07-14 — the earlier Δcos-to-target metric is deprecated, see DECISIONS):
    cell(i,k) = mean_pairs[ cos( tgt_qfinal(k) − src_qfinal(k),  steered_src_qfinal(k) − src_qfinal(k) ) ]
→ one 29×29 grid (x=intervene layer i, y=read layer k) per (direction, intervene token, α). Because the
read token is fixed, the token×token matrix collapses to a vertical STRIP over the 30 intervene tokens.

Layers = 29 residual entries (0 = transformer.drop / embedding, 1..28 = transformer.h.{0..27}).
Structural invariant (asserted): lower triangle k ≤ i ≡ 0 (a demo-token edit reaches qfinal only at read
layers deeper than the edit). NO embedding-column-zero invariant (tokens unmatched ⇒ nonzero).

baukit imported inside main; baukit-free position helpers inlined so the module top stays import-safe.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt, get_token_meta_labels
from utils.paths import LABEL_GEOMETRY_DIR

TASK_PAIRS = {
    "antonym_synonym": ("antonym", "synonym"),
    "next_number_digits_prev_number_digits": ("next_number_digits", "prev_number_digits"),
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


def intervene_keys(n_shots):
    keys = []
    for i in range(1, n_shots + 1):
        keys += [f"d{i}_in", f"d{i}_pre", f"d{i}_lab"]
    return keys


INPUT_TOKEN_RE = re.compile(r"^demonstration_(\d+)_token$")


def token_positions(token_labels, n_shots):
    """{tkey: position} for the 3·n_shots demo tokens + 'qfinal' (query predictive token).

    BUGFIX 2026-07-13: d{i}_in was previously pre-label − 1, which is the constant "A" template
    token, NOT the input word ("Q: hot\\nA: cold" tokenizes as Q,:, hot,\\n,A,:, cold — the label
    already carries the leading space, so pre−1 lands on "A"). It is now the LAST token of demo
    i's input word (the `demonstration_{i}_token` group), mirroring last_label_token for labels.
    All *_in_* grids computed before this date used the "A" token and were deleted/recomputed.
    """
    recs = selected_token_records(token_labels)

    def get(role, icl):
        for r in recs:
            if r["token_role"] == role and r["icl_example_index"] == icl:
                return r["token_position"]
        raise ValueError(f"missing {role}@icl={icl}")

    input_last = {}
    for p, t, l in token_labels:
        m = INPUT_TOKEN_RE.match(l)
        if m:
            g = int(m.group(1))
            input_last[g] = max(input_last.get(g, -1), int(p))

    pos = {}
    for i in range(1, n_shots + 1):
        if i not in input_last:
            raise ValueError(f"no input-word token for ICL {i}")
        pos[f"d{i}_in"] = input_last[i]  # last token of demo i's INPUT WORD
        pos[f"d{i}_pre"] = get("pre_label_token", i)   # the ":" before demo i's label
        pos[f"d{i}_lab"] = get("last_label_token", i)
    pos["qfinal"] = get("last_prompt_token", None)
    seq = [pos[k] for k in intervene_keys(n_shots)] + [pos["qfinal"]]
    assert all(seq[j] < seq[j + 1] for j in range(len(seq) - 1)), f"positions not strictly ordered: {seq}"
    assert seq[0] >= 0, f"input token underflow: {seq}"
    return pos
# ----------------------------------------------------------------------------------


def stable_seed(*parts):
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


def build_prompt(demo_inputs, demo_outputs, query_input):
    return word_pairs_to_prompt_data(
        {"input": list(demo_inputs), "output": list(demo_outputs)},
        query_target_pair={"input": query_input, "output": query_input},
        prepend_bos_token=False, prefixes=PREFIXES, separators=SEPARATORS, prepend_space=True,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Ten-shot intervene-token strip cosine-shift heatmaps.")
    p.add_argument("--task_pair", choices=sorted(TASK_PAIRS), default="antonym_synonym")
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    p.add_argument("--n_shots", type=int, default=10)
    p.add_argument("--n_pairs", type=int, default=300)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Subset of INTERVENTION layers to sweep; default = all 29. Reads are always all 29.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--output_root", type=str,
                   default=str(LABEL_GEOMETRY_DIR / "tenshot_strip_intervention_cos_heatmap"))
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    from baukit import TraceDict  # noqa: local import (model-side dep)

    f1, f2 = TASK_PAIRS[args.task_pair]
    directions = [(f1, f2, "f1", "f2"), (f2, f1, "f2", "f1")]
    print(f"task_pair={args.task_pair}  f1={f1} f2={f2}  n_shots={args.n_shots}  n_pairs={args.n_pairs}")

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

    block_names = model_config["layer_hook_names"]
    n_blocks = model_config["n_layers"]
    assert n_blocks == len(block_names) == 28, f"expected 28 GPT-J blocks, got {n_blocks}"
    emb_name = "transformer.drop"
    layer_names = [emb_name] + list(block_names)             # 29 residual entries
    n_layers = len(layer_names)
    resid_dim = model_config["resid_dim"]

    IKEYS = intervene_keys(args.n_shots)                     # 30 demo tokens
    ALL_KEYS = IKEYS + ["qfinal"]                            # 31 captured positions
    n_intervene = len(IKEYS)
    READ_COL = len(ALL_KEYS) - 1                             # qfinal column index

    inputs_f1 = sorted(i2o_f1.keys())
    inputs_f2 = sorted(i2o_f2.keys())
    shared_in = sorted(set(i2o_f1) & set(i2o_f2))
    print(f"inputs: {f1}={len(inputs_f1)} {f2}={len(inputs_f2)}; shared query pool={len(shared_in)}")

    # ---- build 10-shot prompt pairs (shared query only; independent random demos) ----
    f1_ids_list, f2_ids_list = [], []
    f1_pos_list, f2_pos_list = [], []
    pair_meta = []
    funcs = {"f1": (inputs_f1, i2o_f1), "f2": (inputs_f2, i2o_f2)}
    for p in range(args.n_pairs):
        rng = stable_rng(args.seed, args.task_pair, p)
        q = str(rng.choice(shared_in))
        ok = {}
        for tag in ("f1", "f2"):
            inputs, i2o = funcs[tag]
            cand = [x for x in inputs if x != q]
            idx = rng.choice(len(cand), size=args.n_shots, replace=False)
            demo_in = [cand[j] for j in idx]
            demo_out = [i2o[x] for x in demo_in]
            pd = build_prompt(demo_in, demo_out, q)
            token_labels, prompt_string = get_token_meta_labels(
                pd, tokenizer, query=q, prepend_bos=model_config["prepend_bos"])
            pos = token_positions(token_labels, args.n_shots)
            ids = tokenizer(prompt_string).input_ids
            assert pos["qfinal"] == len(ids) - 1, "query-final not last token"
            ok[tag] = (ids, pos)
        (f1_ids, f1_pos), (f2_ids, f2_pos) = ok["f1"], ok["f2"]
        assert f1_ids[f1_pos["qfinal"]] == f2_ids[f2_pos["qfinal"]], f"query-final differs (p={p})"
        f1_ids_list.append(f1_ids); f1_pos_list.append(f1_pos)
        f2_ids_list.append(f2_ids); f2_pos_list.append(f2_pos)
        pair_meta.append({"query": q})
    n_pairs = len(pair_meta)
    print(f"built {n_pairs} prompt pairs")

    inter_layers = sorted(args.layers) if args.layers is not None else list(range(n_layers))
    pad_id = tokenizer.pad_token_id

    def make_chunks(ids_list, pos_list):
        chunks = []
        for s in range(0, len(ids_list), args.batch_size):
            sub_ids = ids_list[s:s + args.batch_size]
            sub_pos = pos_list[s:s + args.batch_size]
            mx = max(len(x) for x in sub_ids)
            inp = torch.full((len(sub_ids), mx), pad_id, dtype=torch.long)
            att = torch.zeros((len(sub_ids), mx), dtype=torch.long)
            pos = torch.zeros((len(sub_ids), len(ALL_KEYS)), dtype=torch.long)
            for r, ids in enumerate(sub_ids):
                pad = mx - len(ids)
                inp[r, pad:] = torch.tensor(ids, dtype=torch.long)
                att[r, pad:] = 1
                for ti, k in enumerate(ALL_KEYS):
                    pos[r, ti] = pad + sub_pos[r][k]
            chunks.append({"input_ids": inp.to(device), "attention_mask": att.to(device),
                           "pos": pos.to(device), "n": len(sub_ids)})
        return chunks

    f1_chunks = make_chunks(f1_ids_list, f1_pos_list)
    f2_chunks = make_chunks(f2_ids_list, f2_pos_list)

    def make_edit_hook(edit_name, add_vec, rows0, edit_pos):
        def hook(output, layer_name):
            if layer_name != edit_name:
                return output
            if isinstance(output, tuple):
                output[0][rows0, edit_pos] += add_vec
                return output
            output[rows0, edit_pos] += add_vec
            return output
        return hook

    def capture(chunks, read_cols, edit_name=None, add_vec=None, edit_col=None):
        """Forward; gather acts at the given position columns × 29 layers → [N, len(read_cols), 29, D]."""
        outs = []
        for ch in chunks:
            if edit_name is not None:
                rows0 = torch.arange(ch["n"], device=device)
                hook = make_edit_hook(edit_name, add_vec, rows0, ch["pos"][:, edit_col])
                cm = TraceDict(model, layers=layer_names, edit_output=hook, retain_output=True)
            else:
                cm = TraceDict(model, layers=layer_names, retain_output=True)
            with cm as td:
                # model.transformer (not model) runs the blocks our hooks sit on but SKIPS lm_head —
                # we never use logits, and lm_head over [B, seq, vocab] is a large needless allocation.
                model.transformer(input_ids=ch["input_ids"], attention_mask=ch["attention_mask"])
                B = ch["n"]
                rows = torch.arange(B, device=device)
                acts = torch.empty((B, len(read_cols), n_layers, resid_dim), device=device, dtype=dtype)
                for li, nm in enumerate(layer_names):
                    out = td[nm].output
                    out = out[0] if isinstance(out, tuple) else out
                    for j, col in enumerate(read_cols):
                        acts[:, j, li, :] = out[rows, ch["pos"][:, col], :]
            outs.append(acts)
        return torch.cat(outs, 0)

    # ---- unsteered passes (all 31 positions; shared across directions) ----
    print("unsteered f1 pass...")
    f1_acts = capture(f1_chunks, list(range(len(ALL_KEYS))))     # [N, 31, 29, D]
    print("unsteered f2 pass...")
    f2_acts = capture(f2_chunks, list(range(len(ALL_KEYS))))
    assert torch.isfinite(f1_acts).all() and torch.isfinite(f2_acts).all(), "non-finite acts"
    acts_by_tag = {"f1": f1_acts, "f2": f2_acts}
    chunks_by_tag = {"f1": f1_chunks, "f2": f2_chunks}

    out_dir = Path(args.output_root) / args.task_pair
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    summary = {"task_pair": args.task_pair, "f1": f1, "f2": f2, "n_pairs": n_pairs,
               "metric": "dircos: mean_pairs[cos(tgt-src, steered-src)] at qfinal "
                         "(2026-07-14; replaces the deprecated delta-cos-to-target)",
               "n_shots": args.n_shots, "n_layers": n_layers, "alphas": args.alphas,
               "intervene_tokens": IKEYS, "read_token": "qfinal",
               "intervention_layers": inter_layers,
               "embedding_entry": "transformer.drop (residual layer 0); layers 1..28 = block outputs",
               "pairing": "shared query only; 10 independent random demos per function (unmatched labels)",
               "note": ("steer_vec at each demo token = difference of the two tasks' MEAN activations at "
                        "that slot (unmatched demos ⇒ mixes lexical+function). Only qfinal is byte-matched."),
               "directions": {}}

    CB = 64
    for src_task, tgt_task, src_tag, tgt_tag in directions:
        dir_name = f"{src_task}_to_{tgt_task}"
        src_h = acts_by_tag[src_tag]
        tgt_h = acts_by_tag[tgt_tag]
        src_chunks = chunks_by_tag[src_tag]
        # per-demo-token per-layer steer vector + per-pair baseline cos at qfinal (chunked over pairs)
        steer_sum = torch.zeros((n_intervene, n_layers, resid_dim), device=device, dtype=torch.float32)
        baseline_cos = torch.empty((n_pairs, n_layers), device=device, dtype=torch.float32)
        for s in range(0, n_pairs, CB):
            sd = src_h[s:s + CB, :n_intervene].float(); td_ = tgt_h[s:s + CB, :n_intervene].float()
            steer_sum += (td_ - sd).sum(0)
            sq = src_h[s:s + CB, READ_COL].float(); tq = tgt_h[s:s + CB, READ_COL].float()
            baseline_cos[s:s + CB] = F.cosine_similarity(sq, tq, dim=-1)
            del sd, td_, sq, tq
        steer_vec = steer_sum / n_pairs                              # [30, 29, D]
        steer_norms = torch.linalg.norm(steer_vec, dim=-1).cpu().numpy()   # [30, 29]
        mean_baseline = baseline_cos.mean(0).cpu().numpy()                 # [29]
        tgt_q = tgt_h[:, READ_COL]                                   # fp16 [N,29,D]
        src_q = src_h[:, READ_COL]                                   # fp16 [N,29,D] (clean source qfinal)
        dir_tgt_q = (tgt_q.float() - src_q.float())                  # [N,29,D] counterfactual direction
        print(f"\n=== direction {dir_name} (inject into {src_tag}) ===")

        dsum = {"src_task": src_task, "tgt_task": tgt_task, "inject_into": src_tag,
                "mean_baseline_cos_by_read_layer": mean_baseline.tolist(),
                "steer_vec_norm_by_token_layer": {IKEYS[t]: steer_norms[t].tolist() for t in range(n_intervene)},
                "grids": {}}

        for alpha in args.alphas:
            for t_idx, tkey in enumerate(IKEYS):
                tag = f"{dir_name}__{tkey}_alpha{alpha:g}"
                npy_path = out_dir / f"{tag}_grid.npy"
                if npy_path.exists():                       # resumable: skip completed grids
                    grid = np.load(npy_path)
                else:
                    grid = np.full((n_layers, n_layers), np.nan, dtype=np.float64)
                    for i in inter_layers:
                        add_vec = (alpha * steer_vec[t_idx, i]).to(device=device, dtype=dtype)
                        sq = capture(src_chunks, read_cols=[READ_COL],
                                     edit_name=layer_names[i], add_vec=add_vec, edit_col=t_idx)[:, 0]  # [N,29,D]
                        # METRIC (2026-07-14, user-specified): dircos — cos(counterfactual direction,
                        # steering displacement) at qfinal. Zero displacement (read layer <= edit
                        # layer) -> cos 0 via eps, so the lower-tri==0 assert still holds.
                        disp = sq.float() - src_q.float()                                               # [N,29,D]
                        grid[:, i] = F.cosine_similarity(disp, dir_tgt_q, dim=-1).mean(0).cpu().numpy()
                        del sq, disp
                    # structural assert: read layer k <= intervene layer i must be exactly 0
                    for i in inter_layers:
                        assert np.all(grid[: i + 1, i] == 0.0), f"lower-tri nonzero {tag} i={i}"
                    assert np.isfinite(grid[:, inter_layers]).all(), f"non-finite grid {tag}"
                    np.save(npy_path, grid)
                    np.savetxt(out_dir / f"{tag}_grid.csv", grid, delimiter=",",
                               header="rows=read_layer(0..28), cols=intervention_layer(0..28)", comments="")
                fk, fi = np.unravel_index(np.nanargmax(grid), grid.shape)
                dsum["grids"][f"{tkey}_alpha{alpha:g}"] = {
                    "alpha": alpha, "intervene_token": tkey, "npy": f"{tag}_grid.npy",
                    "peak_shift": float(grid[fk, fi]),
                    "peak_intervention_layer": int(fi), "peak_read_layer": int(fk)}
                if device == "cuda":
                    torch.cuda.empty_cache()
            best = max((v for v in dsum["grids"].values() if v["alpha"] == alpha),
                       key=lambda v: v["peak_shift"])
            print(f"  α={alpha:g} strongest token: {best['intervene_token']} peak {best['peak_shift']:+.4f} "
                  f"@ i{best['peak_intervention_layer']}/k{best['peak_read_layer']}")

        summary["directions"][dir_name] = dsum

    with open(out_dir / f"{args.task_pair}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nDONE -> {out_dir}")


if __name__ == "__main__":
    main()
