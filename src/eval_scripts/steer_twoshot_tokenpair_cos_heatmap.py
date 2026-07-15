"""
TWO-shot token-pair × layer×layer cosine-shift heatmaps (GPT-J-6B; loads model + baukit).

Generalises the 1-shot label→query-final heatmap (steer_label_cos_heatmap.py) from a single
{intervene label token → read query-final token} pair to **every ordered pair of tokens** in a
2-shot paired ICL prompt, in BOTH source→target directions.

Construction (matched-label paired 2-shot, same as capture_and_grade_twoshot_paired.py): each pair
of prompts shares the two demo labels (L1, L2, distinct within a prompt) and the query q; only the
two demo INPUTS differ by function. So 5 of the 6 search-space tokens are byte-identical across the
f1/f2 prompts (pure function context); only the demo-2 input token differs.

Search-space tokens (sequence order):
    t1 label1   = last_label_token @ icl 1      (identical across functions)
    t2 input2   = last token of demo-2 INPUT WORD, via demonstration_2_token meta labels
                  (DIFFERS across functions; BUGFIX 2026-07-13 — was pre_label−1 = the constant "A")
    t3 prelabel2= pre_label_token  @ icl 2      (the ":" before L2; identical)
    t4 label2   = last_label_token @ icl 2      (identical)
    t5 qinput   = last token of the QUERY WORD, via query_demonstration_token meta labels
                  (identical; BUGFIX 2026-07-13 — was qfinal−1 = the constant "A")
    t6 qfinal   = last_prompt_token @ None      (trailing ":" = query-final/predictive; identical)

For a direction src→tgt and per residual layer ℓ:
    steer_vec(t, ℓ)   = mean_pairs[ act_tgt(t, ℓ) − act_src(t, ℓ) ]
    baseline_cos(t,ℓ) = mean_pairs[ cos( act_src(t,ℓ), act_tgt(t,ℓ) ) ]
For each intervention token t_i (t1..t5), strength α and intervention layer i: inject α·steer_vec(t_i,i)
at t_i's position in the SOURCE prompt at layer i, then for each LATER read token t_j (j>i) and read
layer k compute the DIRECTION-ALIGNMENT cosine ("dircos", metric of record since 2026-07-14 — the
earlier Δcos-to-target metric is deprecated, see DECISIONS):
    cell(i,k) = mean_pairs[ cos( tgt(t_j,k) − src(t_j,k),  steered_src(t_j,k) − src(t_j,k) ) ]
i.e. does the displacement caused by the intervention point along the counterfactual direction.
→ one 29×29 grid (x=intervention layer, y=read layer) per (direction, t_i→t_j, α). 15 token-pairs.

--steer_mode perpair replaces the pair-MEAN steer vector with each pair's OWN difference
    steer_vec_p(t, ℓ) = act_tgt_p(t, ℓ) − act_src_p(t, ℓ)
injected per prompt (α=1 ≡ exact single-site activation patching; asserted). Evaluation is
identical. Outputs go to twoshot_tokenpair_perpair_cos_heatmap/ (same file layout).

--layer_mode cumulative (perpair only): instead of a single-site edit at layer i, the intervention
token's activation is hard-CLAMPED to the matched target's at EVERY layer ℓ∈[i..28] (trajectory
patching from the start layer i on; asserted at ℓ=i and ℓ=28). No strength sweep (clamping makes α
irrelevant) — grids are tagged with the nominal alpha1 so the plot script runs with --alphas 1.
The grid x-axis is the clamp START layer. Outputs: twoshot_tokenpair_perpair_cumclamp_cos_heatmap/.

Layers = 29 residual entries (0 = embedding / transformer.drop, 1..28 = transformer.h.{0..27}).
Structural invariants (asserted): lower triangle k≤i ≡ 0 (read is downstream of the edit); the
embedding column i=0 ≡ 0 for CLEAN source tokens (steer_vec=0 at the embedding) — the demo-2 input
token (t2) is the documented exception (its embedding diff is nonzero ⇒ column 0 may be nonzero).

baukit imported inside main (precedent: steer_label_cos_heatmap.py); the baukit-free position helpers
are inlined so the module top stays import-safe.
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

PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}

# 6 search-space tokens, in sequence order. CLEAN = byte-identical across f1/f2.
TOKENS = ["label1", "input2", "prelabel2", "label2", "qinput", "qfinal"]
CLEAN = {"label1": True, "input2": False, "prelabel2": True, "label2": True, "qinput": True, "qfinal": True}
N_TOKENS = len(TOKENS)
# intervention sources = every token that has at least one later token (t1..t5; qfinal is read-only)
SRC_TOKEN_IDX = list(range(N_TOKENS - 1))

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


INPUT_TOKEN_RE = re.compile(r"^demonstration_(\d+)_token$")


def token_positions(token_labels):
    """Return {tN: position} for the 6 search-space tokens of a 2-shot paired prompt.

    BUGFIX 2026-07-13 (same as the tenshot strip scripts, DECISIONS "Verify token positions
    against the tokenizer"): input2/qinput were previously pre-label − 1 / qfinal − 1, which is
    the constant "A" template token, NOT the input word ("Q: hot\\nA: cold" tokenizes as
    Q,:, hot,\\n,A,:, cold — the label carries the leading space, so −1 lands on "A"). They are
    now the LAST token of the demo-2 input word (`demonstration_2_token` group) and of the query
    word (`query_demonstration_token` group). All input2/qinput grids computed before this date
    measured the "A" token.
    """
    recs = selected_token_records(token_labels)

    def get(role, icl):
        for r in recs:
            if r["token_role"] == role and r["icl_example_index"] == icl:
                return r["token_position"]
        raise ValueError(f"missing {role}@icl={icl}")

    input2 = max((int(p) for p, t, l in token_labels if INPUT_TOKEN_RE.match(l)
                  and int(INPUT_TOKEN_RE.match(l).group(1)) == 2), default=None)
    qinput = max((int(p) for p, t, l in token_labels if l == "query_demonstration_token"),
                 default=None)
    if input2 is None or qinput is None:
        raise ValueError("missing demo-2 input / query input word tokens")

    label1 = get("last_label_token", 1)
    prelabel2 = get("pre_label_token", 2)
    label2 = get("last_label_token", 2)
    qfinal = get("last_prompt_token", None)
    pos = {"label1": label1, "input2": input2, "prelabel2": prelabel2,
           "label2": label2, "qinput": qinput, "qfinal": qfinal}
    seq = [pos[t] for t in TOKENS]
    assert all(seq[i] < seq[i + 1] for i in range(len(seq) - 1)), f"tokens not strictly ordered: {pos}"
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
    p = argparse.ArgumentParser(description="Two-shot token-pair → layer×layer cosine-shift heatmaps.")
    p.add_argument("--task_pair", choices=sorted(TASK_PAIRS), default="antonym_synonym")
    p.add_argument("--steer_mode", choices=["mean", "perpair"], default="mean",
                   help="mean: inject the pair-mean steer vector (original study). perpair: inject each "
                        "pair's OWN tgt−src difference at the edited (token, layer) — α=1 is exact "
                        "single-site activation patching.")
    p.add_argument("--layer_mode", choices=["single", "cumulative"], default="single",
                   help="single: edit only at the intervention layer i (original). cumulative: hard-CLAMP "
                        "the intervention token's activation to the matched target's at EVERY layer "
                        "ℓ∈[i..28] (trajectory patching from layer i on; perpair only, no α sweep — "
                        "--alphas is ignored and files are tagged with the nominal alpha1).")
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0])
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Subset of INTERVENTION layers (0..28) to sweep; default = all 29. Reads are always all 29.")
    p.add_argument("--max_pairs", type=int, default=None, help="Cap number of prompt pairs (smoke tests).")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--output_root", type=str, default=None,
                   help="Default depends on --steer_mode: twoshot_tokenpair_intervention_cos_heatmap "
                        "(mean) or twoshot_tokenpair_perpair_cos_heatmap (perpair).")
    args = p.parse_args()
    if args.layer_mode == "cumulative":
        if args.steer_mode != "perpair":
            p.error("--layer_mode cumulative requires --steer_mode perpair (clamp is per-pair by definition)")
        if args.alphas != [1.0]:
            print(f"NOTE: --layer_mode cumulative clamps (no strength); ignoring --alphas {args.alphas} -> [1.0]")
            args.alphas = [1.0]
    if args.output_root is None:
        if args.layer_mode == "cumulative":
            sub = "twoshot_tokenpair_perpair_cumclamp_cos_heatmap"
        elif args.steer_mode == "mean":
            sub = "twoshot_tokenpair_intervention_cos_heatmap"
        else:
            sub = "twoshot_tokenpair_perpair_cos_heatmap"
        args.output_root = str(LABEL_GEOMETRY_DIR / sub)
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    from baukit import TraceDict  # noqa: local import (model-side dep)

    f1, f2 = TASK_PAIRS[args.task_pair]
    # both directions for this task pair: (src_task, tgt_task, src_func_tag, tgt_func_tag)
    directions = [(f1, f2, "f1", "f2"), (f2, f1, "f2", "f1")]
    print(f"task_pair={args.task_pair}  f1={f1} f2={f2}  directions={[f'{s}->{t}' for s, t, *_ in directions]}")

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
    block_names = model_config["layer_hook_names"]
    n_blocks = model_config["n_layers"]
    assert n_blocks == len(block_names) == 28, f"expected 28 GPT-J blocks, got {n_blocks}"
    emb_name = "transformer.drop"
    layer_names = [emb_name] + list(block_names)
    n_layers = len(layer_names)                              # 29
    resid_dim = model_config["resid_dim"]

    def single(w):
        return len(tokenizer(" " + w).input_ids) == 1

    shared_out = sorted(set(o2i_f1) & set(o2i_f2))
    label_words = [w for w in shared_out if single(w)]
    shared_in = sorted(set(i2o_f1) & set(i2o_f2))
    query_pool = list(shared_in)
    label_set = list(label_words)
    print(f"label words (shared single-tok output): {len(label_words)}; query pool: {len(query_pool)}")

    # ---- build 2-shot prompt pairs (deterministic, identical to capture_and_grade_twoshot_paired) ----
    f1_ids_list, f2_ids_list = [], []      # token id lists (unpadded)
    f1_pos_list, f2_pos_list = [], []       # per-prompt dict of 6 token positions (unpadded)
    pair_meta = []
    for w in label_words:
        rng = stable_rng(args.seed, args.task_pair, w)
        L1 = w
        cand_L2 = [x for x in label_set if x != L1]
        if not cand_L2:
            continue
        L2 = str(rng.choice(cand_L2))
        labels = [L1, L2]
        demo_inputs = {
            "f1": [str(rng.choice(o2i_f1[L1])), str(rng.choice(o2i_f1[L2]))],
            "f2": [str(rng.choice(o2i_f2[L1])), str(rng.choice(o2i_f2[L2]))],
        }
        forbidden = {L1, L2, *demo_inputs["f1"], *demo_inputs["f2"]}
        cand_q = [q for q in query_pool if q not in forbidden]
        if not cand_q:
            continue
        q = str(rng.choice(cand_q))

        expected_label_ids = [tokenizer(" " + lab).input_ids[-1] for lab in labels]
        ok = {}
        for tag in ("f1", "f2"):
            pd = build_prompt(demo_inputs[tag], labels, q)
            token_labels, prompt_string = get_token_meta_labels(
                pd, tokenizer, query=q, prepend_bos=model_config["prepend_bos"])
            pos = token_positions(token_labels)
            ids = tokenizer(prompt_string).input_ids
            assert pos["qfinal"] == len(ids) - 1, "query-final not last token"
            # paired invariant: shared labels identical across f1/f2
            assert ids[pos["label1"]] == expected_label_ids[0], f"L1 mismatch w={w!r} {tag}"
            assert ids[pos["label2"]] == expected_label_ids[1], f"L2 mismatch w={w!r} {tag}"
            ok[tag] = (ids, pos)
        (f1_ids, f1_pos), (f2_ids, f2_pos) = ok["f1"], ok["f2"]
        # the 5 CLEAN tokens must be byte-identical across the pair; input2 (t2) may differ
        for t in TOKENS:
            if CLEAN[t]:
                assert f1_ids[f1_pos[t]] == f2_ids[f2_pos[t]], f"clean token {t} differs (w={w!r})"
        f1_ids_list.append(f1_ids); f1_pos_list.append(f1_pos)
        f2_ids_list.append(f2_ids); f2_pos_list.append(f2_pos)
        pair_meta.append({"L1": L1, "L2": L2, "query": q, "demo_inputs": demo_inputs})
        if args.max_pairs is not None and len(pair_meta) >= args.max_pairs:
            break
    n_pairs = len(pair_meta)
    print(f"built {n_pairs} prompt pairs")
    assert n_pairs > 0, "no prompt pairs built"
    # DECISIONS 2026-07-13 (verify token positions against the tokenizer): print the decoded
    # token at each of the 6 positions for the first pair, both functions.
    for tag, ids, pos in (("f1", f1_ids_list[0], f1_pos_list[0]),
                          ("f2", f2_ids_list[0], f2_pos_list[0])):
        decoded = {t: tokenizer.decode([ids[pos[t]]]) for t in TOKENS}
        print(f"  sample pair 0 [{tag}]: " + "  ".join(f"{t}={decoded[t]!r}" for t in TOKENS))
        assert decoded["input2"].strip() not in ("A", ":", "Q", ""), \
            f"input2 landed on a template token: {decoded['input2']!r}"
        assert decoded["qinput"].strip() not in ("A", ":", "Q", ""), \
            f"qinput landed on a template token: {decoded['qinput']!r}"

    inter_layers = sorted(args.layers) if args.layers is not None else list(range(n_layers))
    pad_id = tokenizer.pad_token_id

    def make_chunks(ids_list, pos_list):
        """Left-padded chunks; per-row positions for all 6 tokens (shifted by left-pad)."""
        chunks = []
        for s in range(0, len(ids_list), args.batch_size):
            sub_ids = ids_list[s:s + args.batch_size]
            sub_pos = pos_list[s:s + args.batch_size]
            mx = max(len(x) for x in sub_ids)
            inp = torch.full((len(sub_ids), mx), pad_id, dtype=torch.long)
            att = torch.zeros((len(sub_ids), mx), dtype=torch.long)
            pos = torch.zeros((len(sub_ids), N_TOKENS), dtype=torch.long)
            for r, ids in enumerate(sub_ids):
                pad = mx - len(ids)
                inp[r, pad:] = torch.tensor(ids, dtype=torch.long)
                att[r, pad:] = 1
                for ti, t in enumerate(TOKENS):
                    pos[r, ti] = pad + sub_pos[r][t]
            chunks.append({"input_ids": inp.to(device), "attention_mask": att.to(device),
                           "pos": pos.to(device), "n": len(sub_ids)})
        return chunks

    f1_chunks = make_chunks(f1_ids_list, f1_pos_list)
    f2_chunks = make_chunks(f2_ids_list, f2_pos_list)

    def make_edit_hook(edit_name, add_vec, rows0, edit_pos):
        # EXACT 2-arg (output, layer_name) closure -- baukit's invoke_with_optional_args mis-binds
        # extra params positionally (see DECISIONS 2026-06-11). Capture state via closure.
        def hook(output, layer_name):
            if layer_name != edit_name:
                return output
            if isinstance(output, tuple):
                output[0][rows0, edit_pos] += add_vec
                return output
            output[rows0, edit_pos] += add_vec
            return output
        return hook

    def make_clamp_hook(clamp_idx, vals, rows0, edit_pos):
        # Same 2-arg closure constraint as make_edit_hook. clamp_idx: layer_name -> column into
        # vals [B, n_clamp, D]; hard-REPLACES the token's activation at every clamped layer.
        def hook(output, layer_name):
            j = clamp_idx.get(layer_name)
            if j is None:
                return output
            out = output[0] if isinstance(output, tuple) else output
            out[rows0, edit_pos] = vals[:, j]
            return output
        return hook

    def capture(chunks, edit_name=None, add_vec=None, edit_col=None,
                clamp_names=None, clamp_vals=None):
        """Forward over chunks; gather acts at all 6 tokens × 29 layers → [N, 6, 29, D] (GPU, model dtype).

        add_vec: [D] (shared vector, broadcast over rows — mean mode) or [N, D] (one vector per
        prompt pair, sliced per chunk in pair order — perpair mode).
        clamp_names/clamp_vals: cumulative mode — hard-clamp the edit_col token to
        clamp_vals [N, len(clamp_names), D] (model dtype) at each named layer."""
        outs = []
        off = 0
        for ch in chunks:
            if clamp_names is not None:
                rows0 = torch.arange(ch["n"], device=device)
                clamp_idx = {nm: j for j, nm in enumerate(clamp_names)}
                vals = clamp_vals[off:off + ch["n"]].to(dtype)
                hook = make_clamp_hook(clamp_idx, vals, rows0, ch["pos"][:, edit_col])
                cm = TraceDict(model, layers=layer_names, edit_output=hook, retain_output=True)
            elif edit_name is not None:
                rows0 = torch.arange(ch["n"], device=device)
                av = add_vec if add_vec.dim() == 1 else add_vec[off:off + ch["n"]].to(dtype)
                hook = make_edit_hook(edit_name, av, rows0, ch["pos"][:, edit_col])
                cm = TraceDict(model, layers=layer_names, edit_output=hook, retain_output=True)
            else:
                cm = TraceDict(model, layers=layer_names, retain_output=True)
            with cm as td:
                model(input_ids=ch["input_ids"], attention_mask=ch["attention_mask"])
                B = ch["n"]
                rows = torch.arange(B, device=device)
                acts = torch.empty((B, N_TOKENS, n_layers, resid_dim), device=device, dtype=dtype)
                for li, nm in enumerate(layer_names):
                    out = td[nm].output
                    out = out[0] if isinstance(out, tuple) else out
                    for ti in range(N_TOKENS):
                        acts[:, ti, li, :] = out[rows, ch["pos"][:, ti], :]
            outs.append(acts)
            off += ch["n"]
        return torch.cat(outs, 0)       # [N, 6, 29, D]

    # ---- unsteered passes (shared across both directions) ----
    print("unsteered f1 pass...")
    f1_acts = capture(f1_chunks)        # [N, 6, 29, D]
    print("unsteered f2 pass...")
    f2_acts = capture(f2_chunks)
    assert torch.isfinite(f1_acts).all() and torch.isfinite(f2_acts).all(), "non-finite acts"
    acts_by_tag = {"f1": f1_acts, "f2": f2_acts}
    chunks_by_tag = {"f1": f1_chunks, "f2": f2_chunks}

    out_dir = Path(args.output_root) / args.task_pair
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    summary = {"task_pair": args.task_pair, "f1": f1, "f2": f2, "n_pairs": n_pairs,
               "steer_mode": args.steer_mode, "layer_mode": args.layer_mode,
               "metric": "dircos: mean_pairs[cos(tgt-src, steered-src)] at the read site "
                         "(2026-07-14; replaces the deprecated delta-cos-to-target)",
               "n_layers": n_layers, "alphas": args.alphas, "tokens": TOKENS,
               "clean_token": CLEAN, "intervention_source_tokens": [TOKENS[i] for i in SRC_TOKEN_IDX],
               "intervention_layers": inter_layers,
               "embedding_entry": "transformer.drop (residual layer 0); layers 1..28 = block outputs",
               "note_input2": ("demo-2 input token (t2) differs across functions: its steer direction "
                               "mixes lexical+function content and its read baseline cos < 1; the other "
                               "5 tokens are byte-identical (pure function context)."),
               "directions": {}}

    for src_task, tgt_task, src_tag, tgt_tag in directions:
        dir_name = f"{src_task}_to_{tgt_task}"
        src_h = acts_by_tag[src_tag]                 # fp16 [N,6,29,D] (kept on device, low-mem)
        tgt_h = acts_by_tag[tgt_tag]
        src_chunks = chunks_by_tag[src_tag]
        # per-token per-layer steer vector + per-pair baseline cos, chunked over pairs to bound
        # transient float memory (full-N float copies would be ~1.5 GB each; CB rows is ~0.2 GB).
        steer_sum = torch.zeros((N_TOKENS, n_layers, resid_dim), device=device, dtype=torch.float32)
        perpair_norm_sum = torch.zeros((N_TOKENS, n_layers), device=device, dtype=torch.float32)
        baseline_cos = torch.empty((n_pairs, N_TOKENS, n_layers), device=device, dtype=torch.float32)
        CB = 64
        for s in range(0, n_pairs, CB):
            sa = src_h[s:s + CB].float(); ta = tgt_h[s:s + CB].float()
            steer_sum += (ta - sa).sum(0)
            perpair_norm_sum += torch.linalg.norm(ta - sa, dim=-1).sum(0)
            baseline_cos[s:s + CB] = F.cosine_similarity(sa, ta, dim=-1)
            del sa, ta
        steer_vec = steer_sum / n_pairs                                   # [6,29,D] (mean mode only)
        mean_baseline = baseline_cos.mean(0).cpu().numpy()                # [6,29]
        print(f"\n=== direction {dir_name} (inject into {src_tag}, steer_mode={args.steer_mode}, "
              f"layer_mode={args.layer_mode}) ===")

        dsum = {"src_task": src_task, "tgt_task": tgt_task, "inject_into": src_tag,
                "steer_mode": args.steer_mode,
                "mean_baseline_cos_by_token_layer": {TOKENS[t]: mean_baseline[t].tolist() for t in range(N_TOKENS)},
                "grids": {}}
        if args.steer_mode == "mean":
            steer_norms = torch.linalg.norm(steer_vec, dim=-1).cpu().numpy()      # [6,29]
            dsum["steer_vec_norm_by_token_layer"] = {
                TOKENS[t]: steer_norms[t].tolist() for t in range(N_TOKENS)}
        else:
            perpair_norms = (perpair_norm_sum / n_pairs).cpu().numpy()            # [6,29]
            dsum["mean_perpair_steer_norm_by_token_layer"] = {
                TOKENS[t]: perpair_norms[t].tolist() for t in range(N_TOKENS)}

        for alpha in args.alphas:
            # grids[(si,sj)] : 29×29 (read_layer k, intervention_layer i)
            grids = {(si, sj): np.full((n_layers, n_layers), np.nan, dtype=np.float64)
                     for si in SRC_TOKEN_IDX for sj in range(si + 1, N_TOKENS)}
            for i in inter_layers:
                for si in SRC_TOKEN_IDX:
                    if args.layer_mode == "cumulative":
                        # clamp the token to the matched target's activations at EVERY layer ℓ∈[i..28]
                        sfin = capture(src_chunks, edit_col=si,
                                       clamp_names=layer_names[i:], clamp_vals=tgt_h[:, si, i:])
                        for lchk in (i, n_layers - 1):  # clamp identity at the first & last clamped layer
                            pc = F.cosine_similarity(sfin[:, si, lchk].float(),
                                                     tgt_h[:, si, lchk].float(), dim=-1)
                            assert pc.min().item() > 0.999, \
                                f"clamp mismatch at {TOKENS[si]} L{lchk} (start {i}): min cos {pc.min().item():.6f}"
                    else:
                        if args.steer_mode == "mean":
                            add_vec = (alpha * steer_vec[si, i]).to(device=device, dtype=dtype)  # [D]
                        else:  # perpair: each pair's own tgt−src diff at this (token, layer) site
                            add_vec = alpha * (tgt_h[:, si, i].float() - src_h[:, si, i].float())  # [N,D] fp32
                        sfin = capture(src_chunks, edit_name=layer_names[i], add_vec=add_vec, edit_col=si)  # fp16 [N,6,29,D]
                        if args.steer_mode == "perpair" and alpha == 1.0:
                            # α=1 ≡ single-site activation patching: the edited site must equal the target
                            # activation (up to fp16 rounding; skip degenerate zero-diff sites, e.g. layer 0
                            # of clean tokens where src == tgt exactly and cos is trivially 1 anyway).
                            pc = F.cosine_similarity(sfin[:, si, i].float(), tgt_h[:, si, i].float(), dim=-1)
                            assert pc.min().item() > 0.999, \
                                f"α=1 patch mismatch at {TOKENS[si]} L{i}: min cos {pc.min().item():.6f}"
                    for sj in range(si + 1, N_TOKENS):
                        # METRIC (2026-07-14, user-specified): direction-alignment cosine "dircos" —
                        # does the DISPLACEMENT caused by steering point along the COUNTERFACTUAL
                        # direction at the read site. Zero displacement (read layer <= edit layer)
                        # gives cos 0 via the eps clamp, so the lower-tri==0 invariant holds.
                        dir_tgt = tgt_h[:, sj].float() - src_h[:, sj].float()          # [N,29,D]
                        disp = sfin[:, sj].float() - src_h[:, sj].float()              # [N,29,D]
                        cell = F.cosine_similarity(disp, dir_tgt, dim=-1).mean(0).cpu().numpy()  # [29]
                        grids[(si, sj)][:, i] = cell
                        del dir_tgt, disp
                    del sfin
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"  α={alpha:g} intervene L{i:>2}: done ({len(SRC_TOKEN_IDX)} source tokens)")

            # save + summarise each token-pair grid; structural asserts
            for (si, sj), grid in grids.items():
                src_t, read_t = TOKENS[si], TOKENS[sj]
                swept = grid[:, inter_layers]
                assert np.isfinite(swept).all(), f"non-finite grid {src_t}->{read_t} α{alpha}"
                # lower triangle (read layer k <= intervention layer i) must be exactly 0
                for i in inter_layers:
                    assert np.all(grid[: i + 1, i] == 0.0), f"lower-tri nonzero {src_t}->{read_t} i={i}"
                # embedding column (intervene at layer 0) == 0 for CLEAN source tokens only.
                # single mode only: a cumulative clamp starting at i=0 also patches layers 1..28,
                # so its column 0 is legitimately nonzero even for clean tokens.
                if 0 in inter_layers and CLEAN[src_t] and args.layer_mode == "single":
                    assert np.all(grid[:, 0] == 0.0), f"embedding col nonzero for clean src {src_t}"

                tag = f"{dir_name}__{src_t}_to_{read_t}_alpha{alpha:g}"
                np.save(out_dir / f"{tag}_grid.npy", grid)
                np.savetxt(out_dir / f"{tag}_grid.csv", grid, delimiter=",",
                           header="rows=read_layer(0..28), cols=intervention_layer(0..28)", comments="")
                fk, fi = np.unravel_index(np.nanargmax(grid), grid.shape)
                dsum["grids"][f"{src_t}_to_{read_t}_alpha{alpha:g}"] = {
                    "alpha": alpha, "src_token": src_t, "read_token": read_t,
                    "src_token_clean": CLEAN[src_t], "read_token_clean": CLEAN[read_t],
                    "npy": f"{tag}_grid.npy", "peak_shift": float(grid[fk, fi]),
                    "peak_intervention_layer": int(fi), "peak_read_layer": int(fk)}
            # quick console digest: peak per source token (to the strongest read token)
            best = max(dsum["grids"].items(),
                       key=lambda kv: kv[1]["peak_shift"] if kv[1]["alpha"] == alpha else -1)
            print(f"  α={alpha:g} strongest pair: {best[0]} peak {best[1]['peak_shift']:+.4f} "
                  f"@ i{best[1]['peak_intervention_layer']}/k{best[1]['peak_read_layer']}")

        summary["directions"][dir_name] = dsum

    with open(out_dir / f"{args.task_pair}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nDONE -> {out_dir}")


if __name__ == "__main__":
    main()
