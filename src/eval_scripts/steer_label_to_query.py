"""
Phase 2 — causal steering at the demo label token (GPT-J-6B; loads the model + baukit).

Phase 1 (observational) found a dominant "function axis" at the demo label token whose
relational geometry is preserved at the query token. Phase 2 asks the causal question:
if we inject the mean synonym->antonym difference at the demo's label token, does the
query-token representation move along the natural synonym->antonym direction, and does the
model's answer flip from synonym to antonym?

Setup (1-shot):
    - Steer the single demo's label token; read the shift at the query final token.
    - Steering vector = the mean (f1 - f2) difference at the label token, computed from the
      Phase-1 captures (held out from the test queries by the shared-output vs shared-input
      construction). Coefficient sweep (alpha multiplies the natural magnitude).
    - tasks: antonym_synonym -> (antonym, synonym)  [geometry + behavioral flip]
             landmark_park   -> (landmark-country, park-country)  [geometry only]

Steering vectors (from results/oneshot_paired/<pair>/ shards, roles source/target):
    Delta_label(L) = mean_w [ act_source(f1,w,L) - act_source(f2,w,L) ]  (label token)
    Delta_final(L) = mean_w [ act_target(f1,w,L) - act_target(f2,w,L) ]  (query final token)
Default --direction f2_to_f1 adds +Delta (so synonym->antonym for antonym_synonym).

This script loads GPT-J via baukit (TraceDict) and is NOT import-safe without those deps.
baukit is imported inside main only; the small intervention/eval helpers are inlined
(per the precedent in mixed_icl_antonym_synonym_topk.py) to avoid the bitsandbytes import
chain in utils.intervention_utils / utils.eval_utils.
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import get_token_meta_labels, word_pairs_to_prompt_data

# NOTE: extract_residual_stream_activations imports baukit at module top, so we do NOT import
# from it here (that would pull baukit into this module's top level). Instead we inline its
# pure, baukit-free position helpers (selected_token_records / make_token_record / LABEL_TOKEN_RE)
# below verbatim, so baukit is only imported inside main where TraceDict/model are used.


# --- inlined verbatim from extract_residual_stream_activations.py (baukit-free) ---
LABEL_TOKEN_RE = re.compile(r"^demonstration_(\d+)_label_token$")


def make_token_record(token_role, icl_example_index, token):
    token_position, token_text, token_label = token
    return {
        "token_role": token_role,
        "icl_example_index": icl_example_index,
        "token_position": int(token_position),
        "token_text": token_text,
        "token_label": token_label,
    }


def selected_token_records(token_labels):
    tokens_by_position = {int(token_position): (token_position, token_text, token_label)
                          for token_position, token_text, token_label in token_labels}
    label_groups = {}
    for token_position, token_text, token_label in token_labels:
        match = LABEL_TOKEN_RE.match(token_label)
        if match:
            icl_example_index = int(match.group(1))
            label_groups.setdefault(icl_example_index, []).append((token_position, token_text, token_label))

    records = []
    for icl_example_index in sorted(label_groups):
        label_tokens = sorted(label_groups[icl_example_index], key=lambda x: x[0])
        first_label_token = label_tokens[0]
        last_label_token = label_tokens[-1]
        pre_label_position = int(first_label_token[0]) - 1
        if pre_label_position < 0 or pre_label_position not in tokens_by_position:
            raise ValueError(f"Could not find pre-label token for ICL example {icl_example_index}")
        pre_label_token = tokens_by_position[pre_label_position]
        records.extend([
            make_token_record("pre_label_token", icl_example_index, pre_label_token),
            make_token_record("first_label_token", icl_example_index, first_label_token),
            make_token_record("last_label_token", icl_example_index, last_label_token),
            make_token_record("label_token", icl_example_index, last_label_token),
        ])

    final_candidates = [x for x in token_labels if x[2] == "query_predictive_token"]
    if final_candidates:
        final_token = max(final_candidates, key=lambda x: x[0])
    else:
        final_token = token_labels[-1]
    records.extend([
        make_token_record("last_prompt_token", None, final_token),
        make_token_record("final_token", None, final_token),
    ])
    return records
# ----------------------------------------------------------------------------------


# task_pair -> (function-1 task name, function-2 task name)
TASK_PAIRS = {
    "antonym_synonym": ("antonym", "synonym"),
    "landmark_park": ("landmark-country", "park-country"),
}

PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}


# --- inlined from utils.eval_utils (avoids the baukit/bitsandbytes import chain) ---
def get_answer_id(query, answer, tokenizer):
    source = tokenizer(query, truncation=False, padding=False).input_ids
    target = tokenizer(query + answer, truncation=False, padding=False).input_ids
    assert len(source) < len(target) < tokenizer.model_max_length
    return target[len(source):]


def compute_individual_token_rank(prob_dist, target_id):
    if isinstance(target_id, list):
        target_id = target_id[0]
    return torch.where(torch.argsort(prob_dist.squeeze(), descending=True) == target_id)[0].item()
# -----------------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Causal steering at the demo label token; read geometry+behavior at the query final token."
    )
    p.add_argument("--task_pair", choices=sorted(TASK_PAIRS), default="antonym_synonym")
    p.add_argument("--direction", choices=["f2_to_f1", "f1_to_f2"], default="f2_to_f1",
                   help="f2_to_f1 (default, syn->ant) adds +Delta; f1_to_f2 adds -Delta.")
    p.add_argument("--steer_layers", type=int, nargs="+", default=[6, 9, 11],
                   help="Layers at which to inject Delta_label at the demo label token.")
    p.add_argument("--read_layers", type=int, nargs="+", default=None,
                   help="Read layers for geometry (default: all L >= min(steer_layers)).")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
                   help="Multiples of the natural Delta_label(L_steer) magnitude.")
    p.add_argument("--n_queries", type=int, default=None, help="Cap on test queries (None=all).")
    p.add_argument("--control_random", action="store_true",
                   help="Also run a random-vector control matched in norm at the same position.")
    p.add_argument("--capture_root", type=str, default="results/oneshot_paired")
    p.add_argument("--output_root", type=str, default="results/oneshot_steering")
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--revision", type=str, default=None)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Dataset + capture loading (baukit-free)
# --------------------------------------------------------------------------- #
def load_task_json(root_data_dir, task):
    """Load an abstractive dataset JSON (list of {'input','output'} dicts)."""
    path = Path(root_data_dir) / "abstractive" / f"{task}.json"
    with open(path, "r") as f:
        return json.load(f)


def build_output_to_inputs(records):
    """Map output_word -> sorted list of distinct input words that produce it."""
    out_to_in = {}
    for rec in records:
        out = str(rec["output"]).strip()
        inp = str(rec["input"]).strip()
        out_to_in.setdefault(out, set()).add(inp)
    return {out: sorted(inputs) for out, inputs in out_to_in.items()}


def build_input_to_outputs(records):
    """Map input_word -> sorted list of distinct output words it maps to."""
    in_to_out = {}
    for rec in records:
        out = str(rec["output"]).strip()
        inp = str(rec["input"]).strip()
        in_to_out.setdefault(inp, set()).add(out)
    return {inp: sorted(outs) for inp, outs in in_to_out.items()}


def is_single_space_token(tokenizer, word):
    """True iff ' '+word is a single token (the space-prefixed convention used by the prompt builder)."""
    return len(tokenizer(" " + word).input_ids) == 1


def torch_load_trusted(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_capture_diffs(capture_dir):
    """Load Phase-1 shards and compute per-layer Delta_label and Delta_final.

    Returns (delta_label[n_layers, hidden], delta_final[n_layers, hidden], n_layers, n_words),
    both as torch.float32 tensors. Per layer the difference is the mean over the words present
    under BOTH functions of (f1 - f2); words whose full per-word diff is exactly zero
    (degenerate captures) are dropped before averaging.
    """
    index = json.load(open(capture_dir / "index.json"))
    # role -> function -> word -> [n_layers, hidden]
    acts = {}
    n_layers = None
    for shard in index["shards"]:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = capture_dir / shard_path.name
        data = torch_load_trusted(shard_path, map_location="cpu")
        activations = data["activations"].to(torch.float32)
        metadata = data["metadata"]
        if len(metadata) != activations.shape[0]:
            raise ValueError(f"Metadata/activation mismatch in {shard_path}")
        if n_layers is None:
            n_layers = activations.shape[1]
        for i, meta in enumerate(metadata):
            role = meta["role"]
            function = meta["function"]
            w = meta["output_word"]
            acts.setdefault(role, {}).setdefault(function, {})[w] = activations[i]

    def mean_diff(role):
        f1 = acts.get(role, {}).get("f1", {})
        f2 = acts.get(role, {}).get("f2", {})
        words = sorted(set(f1).intersection(f2))
        # per-word diff [n_layers, hidden]; drop words that are degenerate (all-zero diff).
        per_word = []
        for w in words:
            d = f1[w] - f2[w]
            if torch.linalg.norm(d) == 0:
                continue
            per_word.append(d)
        if not per_word:
            raise ValueError(f"No non-degenerate words for role={role} in {capture_dir}")
        stacked = torch.stack(per_word, dim=0)  # [W, n_layers, hidden]
        mean = stacked.mean(dim=0)  # [n_layers, hidden]
        return mean, len(per_word)

    delta_label, n_words = mean_diff("source")
    delta_final, _ = mean_diff("target")
    return delta_label, delta_final, n_layers, n_words


# --------------------------------------------------------------------------- #
# Test prompts
# --------------------------------------------------------------------------- #
def build_prompt_data(demo_input, demo_output, query_input, query_output):
    """One-demo ICL prompt (GPT-J convention: prepend_bos False, prepend_space True)."""
    word_pairs = {"input": [demo_input], "output": [demo_output]}
    query_target_pair = {"input": query_input, "output": query_output}
    return word_pairs_to_prompt_data(
        word_pairs,
        query_target_pair=query_target_pair,
        prepend_bos_token=False,
        prefixes=PREFIXES,
        separators=SEPARATORS,
        prepend_space=True,
    )


def extract_positions(token_labels):
    """Re-derive the demo-label position (last_label_token, icl_example_index==1) and the
    query-final position (last_prompt_token, icl_example_index is None) from selected_token_records.
    Returns (demo_label_idx, query_final_idx).
    """
    records = selected_token_records(token_labels)
    demo_label_idx = None
    query_final_idx = None
    for rec in records:
        if rec["token_role"] == "last_label_token" and rec["icl_example_index"] == 1:
            demo_label_idx = rec["token_position"]
        elif rec["token_role"] == "last_prompt_token" and rec["icl_example_index"] is None:
            query_final_idx = rec["token_position"]
    if demo_label_idx is None or query_final_idx is None:
        raise ValueError("Could not derive both demo-label (last_label_token@1) and query-final positions.")
    return demo_label_idx, query_final_idx


def build_antonym_synonym_queries(tokenizer, ant_records, syn_records):
    """Test queries = words appearing as INPUT in BOTH antonym and synonym with single-token
    (space-prefixed) gold syn(q) AND ant(q). Returns list of dicts {q, ant, syn}.
    """
    ant_in2out = build_input_to_outputs(ant_records)
    syn_in2out = build_input_to_outputs(syn_records)
    shared_inputs = sorted(set(ant_in2out).intersection(syn_in2out))
    queries = []
    for q in shared_inputs:
        # pick a single-token (space-prefixed) gold for each function (first that qualifies).
        ant_gold = next((w for w in ant_in2out[q] if is_single_space_token(tokenizer, w)), None)
        syn_gold = next((w for w in syn_in2out[q] if is_single_space_token(tokenizer, w)), None)
        if ant_gold is None or syn_gold is None:
            continue
        queries.append({"q": q, "ant": ant_gold, "syn": syn_gold})
    return queries


def build_landmark_park_queries(tokenizer, lm_records, park_records):
    """Test queries = shared OUTPUT countries (single-token, space-prefixed). Geometry only,
    so we only need the query word. Returns list of dicts {q}.
    """
    lm_out2in = build_output_to_inputs(lm_records)
    park_out2in = build_output_to_inputs(park_records)
    shared_outputs = sorted(set(lm_out2in).intersection(park_out2in))
    queries = [{"q": c} for c in shared_outputs if is_single_space_token(tokenizer, c)]
    return queries


def pick_demo_pair(records, query_word, rng):
    """Pick a random demo (input, output) pair from records whose input != query_word."""
    candidates = [(str(r["input"]).strip(), str(r["output"]).strip())
                  for r in records if str(r["input"]).strip() != query_word]
    idx = int(rng.integers(len(candidates)))
    return candidates[idx]


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def mean_ci(values):
    """Return (mean, half-width of 95% CI) for a list/array of floats; (nan,nan) if empty."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, float("nan")
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return mean, 1.96 * sem


def cos(a, b):
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(torch.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    from baukit import TraceDict  # imported here to keep baukit out of module top level

    task_f1, task_f2 = TASK_PAIRS[args.task_pair]
    behavioral = (args.task_pair == "antonym_synonym")
    print(f"task_pair={args.task_pair} -> f1={task_f1}, f2={task_f2}; behavioral={behavioral}")

    capture_root = Path(args.capture_root)
    if not capture_root.is_absolute():
        capture_root = Path.cwd() / capture_root
    capture_dir = capture_root / args.task_pair

    delta_label, delta_final, n_layers, n_words = load_capture_diffs(capture_dir)
    print(f"loaded captures: n_layers={n_layers}, n_words={n_words}")

    # direction sign: f2_to_f1 adds +Delta, f1_to_f2 adds -Delta.
    sign = 1.0 if args.direction == "f2_to_f1" else -1.0
    delta_label = sign * delta_label
    delta_final = sign * delta_final

    # Load model.
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision
    )
    model.eval()
    device = model.device
    dtype = next(model.parameters()).dtype
    layer_hook_names = model_config["layer_hook_names"]
    assert n_layers == model_config["n_layers"], (
        f"capture n_layers={n_layers} != model n_layers={model_config['n_layers']}"
    )

    # Precompute per-steer-layer Delta norms (natural magnitude reference for alpha).
    delta_label_norm = {L: float(torch.linalg.norm(delta_label[L])) for L in args.steer_layers}

    # Read layers: default all L >= min(steer_layers).
    min_steer = min(args.steer_layers)
    if args.read_layers is None:
        read_layers = list(range(min_steer, n_layers))
    else:
        read_layers = sorted(args.read_layers)
    print(f"steer_layers={args.steer_layers} read_layers={read_layers} alphas={args.alphas}")

    # Build test queries.
    records_f1 = load_task_json(args.root_data_dir, task_f1)
    records_f2 = load_task_json(args.root_data_dir, task_f2)
    if behavioral:
        queries = build_antonym_synonym_queries(tokenizer, records_f1, records_f2)
    else:
        queries = build_landmark_park_queries(tokenizer, records_f1, records_f2)
    print(f"test queries: {len(queries)}")
    if args.n_queries is not None:
        queries = queries[: args.n_queries]
        print(f"capped to {len(queries)} queries (--n_queries)")

    # Demo context = the f2 task (synonym-context for antonym_synonym; park-context for
    # landmark_park, i.e. records_f2). Unsteered the model should favor f2; steering pushes f1.
    demo_records = records_f2

    # Move Delta vectors to device for steering / comparison.
    delta_label_dev = delta_label.to(device)
    delta_final_dev = delta_final.to(device)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_dir = output_root / args.task_pair
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV columns.
    fieldnames = [
        "task_pair", "direction", "query", "L_steer", "L_read", "alpha",
        "cos_shift_dfinal", "shift_norm",
        "rank_ant", "rank_syn", "logit_ant", "logit_syn",
        "cos_shift_dfinal_random", "shift_norm_random",
        "rank_ant_random", "rank_syn_random", "logit_ant_random", "logit_syn_random",
        "delta_label_norm", "delta_final_norm",
    ]
    csv_path = output_dir / "per_query.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Aggregation accumulators keyed by (L_steer, L_read, alpha) for geometry, and
    # (L_steer, alpha) for behavior.
    geo = {}        # key -> list of cos_shift_dfinal
    geo_norm = {}   # key -> list of shift_norm
    geo_rand = {}   # key -> list of cos_shift_dfinal_random
    beh_flip = {}      # (L_steer, alpha) -> list of 1/0 rank(ant)<rank(syn)
    beh_logitdiff = {}  # (L_steer, alpha) -> list of (logit_ant - logit_syn)
    beh_flip_rand = {}
    beh_logitdiff_rand = {}

    rng_master = np.random.default_rng(args.seed)

    def read_act(td, L, pos):
        out = td[layer_hook_names[L]].output
        if isinstance(out, tuple):
            out = out[0]
        # [batch, seq, hidden]
        return out[0, pos, :].detach().to(torch.float32).cpu()

    def make_add_hook(add_vec, edit_layer, idx):
        # Factory closure -> 2-arg (output, layer_name) signature exactly like
        # intervention_utils.add_function_vector (baukit fills these by name; extra
        # params would be mis-bound positionally).
        def add_at(output, layer_name):
            if int(layer_name.split(".")[2]) == edit_layer and isinstance(output, tuple):
                output[0][:, idx] += add_vec
                return output
            return output
        return add_at

    for qi, qrec in enumerate(queries):
        q = qrec["q"]
        # Build the synonym-/park-context 1-shot prompt: random demo pair from f2, demo input != q.
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        demo_in, demo_out = pick_demo_pair(demo_records, q, rng)
        # Query output is unused for scoring (we score ant/syn golds explicitly); pass demo_out
        # as a placeholder to satisfy the prompt builder.
        prompt_data = build_prompt_data(demo_in, demo_out, q, demo_out)
        token_labels, prompt_string = get_token_meta_labels(
            prompt_data, tokenizer, query=q, prepend_bos=model_config["prepend_bos"]
        )
        demo_label_idx, query_final_idx = extract_positions(token_labels)

        inputs = tokenizer([prompt_string], return_tensors="pt").to(device)

        # Behavioral target ids (first token of " "+gold), antonym_synonym only.
        if behavioral:
            ant_ids = get_answer_id(prompt_string, " " + qrec["ant"], tokenizer)
            syn_ids = get_answer_id(prompt_string, " " + qrec["syn"], tokenizer)
            ant_id = ant_ids[0]
            syn_id = syn_ids[0]

        # --- clean pass (read all read layers + final logits) ---
        with TraceDict(model, layers=layer_hook_names, retain_output=True) as td:
            clean_logits = model(**inputs).logits
        clean_acts = {L: read_act(td, L, query_final_idx) for L in read_layers}
        clean_final_logits = clean_logits[0, query_final_idx, :].detach()
        if behavioral:
            clean_rank_ant = compute_individual_token_rank(clean_final_logits, ant_id)
            clean_rank_syn = compute_individual_token_rank(clean_final_logits, syn_id)
            clean_logit_ant = float(clean_final_logits[ant_id])
            clean_logit_syn = float(clean_final_logits[syn_id])

        # Precompute a per-query random unit direction (resid_dim), fixed across alphas/layers
        # but seeded; scaled per (L_steer, alpha) to norm alpha*||Delta_label(L_steer)||.
        if args.control_random:
            rvec = torch.from_numpy(rng.standard_normal(model_config["resid_dim"]).astype(np.float32))
            rnorm = torch.linalg.norm(rvec)
            rvec_unit = (rvec / rnorm) if rnorm > 0 else rvec
            rvec_unit = rvec_unit.to(device)

        for L_steer in args.steer_layers:
            steer_vec_base = delta_label_dev[L_steer]  # natural Delta_label at L_steer
            dnorm = delta_label_norm[L_steer]
            for alpha in args.alphas:
                add_vec = (alpha * steer_vec_base).to(device=device, dtype=dtype)

                # Inlined add_function_vector hook (intervention_utils:98-122),
                # at L_steer / demo_label_idx.
                add_at = make_add_hook(add_vec, L_steer, demo_label_idx)

                with TraceDict(model, layers=layer_hook_names, edit_output=add_at,
                               retain_output=True) as td:
                    steered_logits = model(**inputs).logits
                steered_final_logits = steered_logits[0, query_final_idx, :].detach()
                steered_acts = {L: read_act(td, L, query_final_idx) for L in read_layers}

                if behavioral:
                    s_rank_ant = compute_individual_token_rank(steered_final_logits, ant_id)
                    s_rank_syn = compute_individual_token_rank(steered_final_logits, syn_id)
                    s_logit_ant = float(steered_final_logits[ant_id])
                    s_logit_syn = float(steered_final_logits[syn_id])
                    bkey = (L_steer, alpha)
                    beh_flip.setdefault(bkey, []).append(1.0 if s_rank_ant < s_rank_syn else 0.0)
                    beh_logitdiff.setdefault(bkey, []).append(s_logit_ant - s_logit_syn)
                else:
                    s_rank_ant = s_rank_syn = ""
                    s_logit_ant = s_logit_syn = ""

                # alpha==0 sanity: steered must equal clean.
                if alpha == 0:
                    assert torch.allclose(steered_final_logits, clean_final_logits, atol=1e-4), (
                        f"alpha=0 steered != clean (L_steer={L_steer}, q={q!r})"
                    )

                # --- random control: matched-norm random vector at the same position ---
                rand_cols = {k: "" for k in (
                    "cos_shift_dfinal_random", "shift_norm_random",
                    "rank_ant_random", "rank_syn_random", "logit_ant_random", "logit_syn_random")}
                rand_acts = None
                if args.control_random:
                    rand_add = (alpha * dnorm * rvec_unit).to(device=device, dtype=dtype)
                    add_at_rand = make_add_hook(rand_add, L_steer, demo_label_idx)

                    with TraceDict(model, layers=layer_hook_names, edit_output=add_at_rand,
                                   retain_output=True) as tdr:
                        rand_logits = model(**inputs).logits
                    rand_acts = {L: read_act(tdr, L, query_final_idx) for L in read_layers}
                    rand_final_logits = rand_logits[0, query_final_idx, :].detach()
                    if behavioral:
                        r_rank_ant = compute_individual_token_rank(rand_final_logits, ant_id)
                        r_rank_syn = compute_individual_token_rank(rand_final_logits, syn_id)
                        rand_cols["rank_ant_random"] = r_rank_ant
                        rand_cols["rank_syn_random"] = r_rank_syn
                        rand_cols["logit_ant_random"] = float(rand_final_logits[ant_id])
                        rand_cols["logit_syn_random"] = float(rand_final_logits[syn_id])
                        rbkey = (L_steer, alpha)
                        beh_flip_rand.setdefault(rbkey, []).append(1.0 if r_rank_ant < r_rank_syn else 0.0)
                        beh_logitdiff_rand.setdefault(rbkey, []).append(
                            float(rand_final_logits[ant_id]) - float(rand_final_logits[syn_id]))

                for L in read_layers:
                    shift = steered_acts[L] - clean_acts[L]
                    c = cos(shift, delta_final[L])
                    snorm = float(torch.linalg.norm(shift))
                    gkey = (L_steer, L, alpha)
                    geo.setdefault(gkey, []).append(c)
                    geo_norm.setdefault(gkey, []).append(snorm)

                    row = {
                        "task_pair": args.task_pair,
                        "direction": args.direction,
                        "query": q,
                        "L_steer": L_steer,
                        "L_read": L,
                        "alpha": alpha,
                        "cos_shift_dfinal": c,
                        "shift_norm": snorm,
                        "rank_ant": s_rank_ant if behavioral else "",
                        "rank_syn": s_rank_syn if behavioral else "",
                        "logit_ant": s_logit_ant if behavioral else "",
                        "logit_syn": s_logit_syn if behavioral else "",
                        "delta_label_norm": dnorm,
                        "delta_final_norm": float(torch.linalg.norm(delta_final[L])),
                    }
                    row.update(rand_cols)
                    if args.control_random and rand_acts is not None:
                        rshift = rand_acts[L] - clean_acts[L]
                        rc = cos(rshift, delta_final[L])
                        row["cos_shift_dfinal_random"] = rc
                        row["shift_norm_random"] = float(torch.linalg.norm(rshift))
                        geo_rand.setdefault(gkey, []).append(rc)
                    writer.writerow(row)

        if (qi + 1) % 50 == 0:
            csv_file.flush()
            print(f"  processed {qi + 1}/{len(queries)} queries")

    csv_file.close()
    print(f"wrote {csv_path}")

    # --------------------------------------------------------------------- #
    # Summary aggregates.
    # --------------------------------------------------------------------- #
    geometry_summary = []
    for (L_steer, L_read, alpha) in sorted(geo):
        cos_mean, cos_ci = mean_ci(geo[(L_steer, L_read, alpha)])
        norm_mean, norm_ci = mean_ci(geo_norm[(L_steer, L_read, alpha)])
        entry = {
            "L_steer": L_steer, "L_read": L_read, "alpha": alpha,
            "n": len(geo[(L_steer, L_read, alpha)]),
            "cos_shift_dfinal_mean": cos_mean, "cos_shift_dfinal_ci95": cos_ci,
            "shift_norm_mean": norm_mean, "shift_norm_ci95": norm_ci,
        }
        if args.control_random and (L_steer, L_read, alpha) in geo_rand:
            rc_mean, rc_ci = mean_ci(geo_rand[(L_steer, L_read, alpha)])
            entry["cos_shift_dfinal_random_mean"] = rc_mean
            entry["cos_shift_dfinal_random_ci95"] = rc_ci
        geometry_summary.append(entry)

    behavioral_summary = []
    if behavioral:
        for (L_steer, alpha) in sorted(beh_flip):
            flips = beh_flip[(L_steer, alpha)]
            ld_mean, ld_ci = mean_ci(beh_logitdiff[(L_steer, alpha)])
            entry = {
                "L_steer": L_steer, "alpha": alpha,
                "n": len(flips),
                "flip_rate": float(np.mean(flips)) if flips else float("nan"),
                "logit_diff_ant_syn_mean": ld_mean,
                "logit_diff_ant_syn_ci95": ld_ci,
            }
            if args.control_random and (L_steer, alpha) in beh_flip_rand:
                rflips = beh_flip_rand[(L_steer, alpha)]
                rld_mean, rld_ci = mean_ci(beh_logitdiff_rand[(L_steer, alpha)])
                entry["flip_rate_random"] = float(np.mean(rflips)) if rflips else float("nan")
                entry["logit_diff_ant_syn_random_mean"] = rld_mean
                entry["logit_diff_ant_syn_random_ci95"] = rld_ci
            behavioral_summary.append(entry)

    delta_norms = {
        "delta_label_norm": {int(L): float(torch.linalg.norm(delta_label[L])) for L in range(n_layers)},
        "delta_final_norm": {int(L): float(torch.linalg.norm(delta_final[L])) for L in range(n_layers)},
    }

    summary = {
        "task_pair": args.task_pair,
        "direction": args.direction,
        "function_tasks": {"f1": task_f1, "f2": task_f2},
        "model_name": args.model_name,
        "seed": args.seed,
        "n_queries": len(queries),
        "n_capture_words": n_words,
        "steer_layers": args.steer_layers,
        "read_layers": read_layers,
        "alphas": args.alphas,
        "control_random": args.control_random,
        "behavioral": behavioral,
        "delta_norms": delta_norms,
        "geometry": geometry_summary,
        "behavioral_summary": behavioral_summary,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")

    # --------------------------------------------------------------------- #
    # Plots (matplotlib Agg). Straightforward; skip silently on failure.
    # --------------------------------------------------------------------- #
    try:
        # cos(shift, Delta_final) vs alpha at the natural read layer (= each L_steer), per L_steer.
        for L_steer in args.steer_layers:
            L_read = L_steer if L_steer in read_layers else min(read_layers)
            xs = sorted({a for (ls, lr, a) in geo if ls == L_steer and lr == L_read})
            ys = [mean_ci(geo[(L_steer, L_read, a)])[0] for a in xs]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xs, ys, marker="o", label="steering")
            if args.control_random:
                yr = [mean_ci(geo_rand.get((L_steer, L_read, a), []))[0] for a in xs]
                ax.plot(xs, yr, marker="x", linestyle="--", label="random control")
            ax.set_xlabel("alpha")
            ax.set_ylabel(f"mean cos(shift, Delta_final) @ L_read={L_read}")
            ax.set_title(f"cos(shift, Delta_final) vs alpha (L_steer={L_steer})")
            ax.axhline(0.0, color="0.7", linewidth=0.8)
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / f"fig_cos_shift_vs_alpha_L{L_steer}.png", dpi=150)
            plt.close(fig)

        if behavioral and behavioral_summary:
            # flip rate vs alpha, per L_steer.
            fig, ax = plt.subplots(figsize=(6, 4))
            for L_steer in args.steer_layers:
                xs = sorted({a for (ls, a) in beh_flip if ls == L_steer})
                ys = [float(np.mean(beh_flip[(L_steer, a)])) for a in xs]
                ax.plot(xs, ys, marker="o", label=f"L_steer={L_steer}")
                if args.control_random:
                    yr = [float(np.mean(beh_flip_rand[(L_steer, a)]))
                          for a in xs if (L_steer, a) in beh_flip_rand]
                    if len(yr) == len(xs):
                        ax.plot(xs, yr, marker="x", linestyle="--", label=f"random L_steer={L_steer}")
            ax.set_xlabel("alpha")
            ax.set_ylabel("flip rate (rank(ant) < rank(syn))")
            ax.set_title("Synonym->antonym flip rate vs alpha")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "fig_flip_rate_vs_alpha.png", dpi=150)
            plt.close(fig)

            # logit diff (ant - syn) vs alpha, per L_steer.
            fig, ax = plt.subplots(figsize=(6, 4))
            for L_steer in args.steer_layers:
                xs = sorted({a for (ls, a) in beh_logitdiff if ls == L_steer})
                ys = [mean_ci(beh_logitdiff[(L_steer, a)])[0] for a in xs]
                ax.plot(xs, ys, marker="o", label=f"L_steer={L_steer}")
                if args.control_random:
                    yr = [mean_ci(beh_logitdiff_rand[(L_steer, a)])[0]
                          for a in xs if (L_steer, a) in beh_logitdiff_rand]
                    if len(yr) == len(xs):
                        ax.plot(xs, yr, marker="x", linestyle="--", label=f"random L_steer={L_steer}")
            ax.set_xlabel("alpha")
            ax.set_ylabel("mean (logit_ant - logit_syn)")
            ax.set_title("Logit difference (ant - syn) vs alpha")
            ax.axhline(0.0, color="0.7", linewidth=0.8)
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / "fig_logitdiff_vs_alpha.png", dpi=150)
            plt.close(fig)
    except Exception as e:  # noqa: BLE001 - plotting is best-effort
        print(f"plotting skipped: {e}")

    print("done")


if __name__ == "__main__":
    main()
