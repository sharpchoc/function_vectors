"""
Two-shot FV-projection-ablation test: are function vectors the machinery of task imitation? (GPT-J-6B)

Builds matched-label 2-shot prompt pairs (same construction as steer_twoshot_tokenpair_cos_heatmap.py:
each src/tgt pair shares the two demo labels L1,L2 and the query q; only the demo INPUTS differ by
function). For each direction src_task→tgt_task and steering strength α we measure, at the query-final
predictive token (qfinal), the answer-logit gap
        Δlogit = logit(a_tgt) − logit(a_src)          a_src = src_task(q),  a_tgt = tgt_task(q)

Interventions on the SOURCE prompt, two independent switches:
  STEER at a SINGLE (token position, layer ℓ): add α·steer_vec(t,ℓ) at ONE site t at ONE layer ℓ, then
  let the forward recompute downstream. We produce a SEPARATE layer-sweep (ℓ=0..28) for EACH steer site
  t ∈ {label1, label2, qfinal} -- one localized curve per token position, not all sites at once.
      steer_vec(t,ℓ) = mean_pairs[ act_tgt(t,ℓ) − act_src(t,ℓ) ]
  ABLATE the target-FV-specific direction at ALL 29 layers at qfinal only (fixed, not swept):
      F = FV(src_task), F' = FV(tgt_task);  F_perp = F' − (F'·F̂)F̂ ;  u = F_perp/‖F_perp‖
      at qfinal, each layer output h ← h − (h·u) u        (steer applied first, then ablate)

Per (direction, α, steer site t) we therefore get:
  clean          : scalar Δlogit (no steer, no ablate)                       -- flat baseline
  ablate         : scalar Δlogit (no steer, ablate)                          -- flat baseline
  steer(t,ℓ)     : Δlogit curve over injection layer ℓ (steer at t, no ablate)
  steer+ablate(t,ℓ): Δlogit curve over injection layer ℓ (steer at t, ablate)
The question: at the (site,layer) where steer(t,ℓ) lifts Δlogit toward the target task (steer_gain =
steer(t,ℓ) − clean > 0), does ablating F_perp shrink that gain (steer_ablate_gain = steer+ablate(t,ℓ) −
ablate)? retention(t,ℓ) = steer_ablate_gain / steer_gain. clean/ablate are site/α-independent (once).

Readout skips the lm_head over the full [B,seq,vocab] tensor: run model.transformer(...) under the
edit hook, then apply lm_head to the qfinal hidden slice only.

Layers = 29 residual entries (0 = embedding / transformer.drop, 1..28 = transformer.h.{0..27}).
baukit imported inside main (model-side dep); position helpers inlined (import-safe module top).
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

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import word_pairs_to_prompt_data, get_token_meta_labels
from utils.eval_utils import get_answer_id
from utils.paths import LABEL_GEOMETRY_DIR, ARTIFACTS_ROOT

# task_pair -> (f1, f2)
TASK_PAIRS = {
    "antonym_synonym": ("antonym", "synonym"),
    "next_number_digits_prev_number_digits": ("next_number_digits", "prev_number_digits"),
}

PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}

# Full 2-shot token set (sequence order) used by the position helper; we steer only at the 3 below.
TOKENS = ["label1", "input2", "prelabel2", "label2", "qinput", "qfinal"]
# steer sites (user decision): the two demo answer tokens + the query-final predictive "A:"
STEER_TOKENS = ["label1", "label2", "qfinal"]
N_STEER = len(STEER_TOKENS)

# --- inlined verbatim (baukit-free) from steer_twoshot_tokenpair_cos_heatmap.py ---
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


def token_positions(token_labels):
    """Return {tN: position} for the 6 search-space tokens of a 2-shot paired prompt."""
    recs = selected_token_records(token_labels)

    def get(role, icl):
        for r in recs:
            if r["token_role"] == role and r["icl_example_index"] == icl:
                return r["token_position"]
        raise ValueError(f"missing {role}@icl={icl}")

    label1 = get("last_label_token", 1)
    prelabel2 = get("pre_label_token", 2)
    label2 = get("last_label_token", 2)
    qfinal = get("last_prompt_token", None)
    pos = {"label1": label1, "input2": prelabel2 - 1, "prelabel2": prelabel2,
           "label2": label2, "qinput": qfinal - 1, "qfinal": qfinal}
    seq = [pos[t] for t in TOKENS]
    assert all(seq[i] < seq[i + 1] for i in range(len(seq) - 1)), f"tokens not strictly ordered: {pos}"
    assert pos["input2"] >= 0 and pos["qinput"] >= 0, f"input token underflow: {pos}"
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


def load_fv(fv_root, task, resid_dim, device):
    """Load a task function vector as a float32 [resid_dim] tensor.

    compute_function_vectors.py saves a dict {'function_vector','top_heads','n_top_heads'}; top_heads
    holds numpy scalars so weights_only=False is required on torch>=2.6.
    """
    p = Path(fv_root) / task / f"{task}_function_vector.pt"
    if not p.exists():
        raise FileNotFoundError(f"missing FV for {task}: {p} (compute it with compute_function_vectors.py)")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    fv = obj["function_vector"] if isinstance(obj, dict) else obj
    fv = fv.reshape(-1).float()
    assert fv.numel() == resid_dim, f"FV {task} has {fv.numel()} dims, expected {resid_dim}"
    return fv.to(device)


def parse_args():
    p = argparse.ArgumentParser(description="Two-shot FV-projection-ablation task-imitation logit test "
                                            "(per-layer steer sweep).")
    p.add_argument("--task_pair", choices=sorted(TASK_PAIRS), default="antonym_synonym")
    p.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    p.add_argument("--n_pairs", type=int, default=300, help="Cap number of prompt pairs.")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Subset of injection layers (0..28) to sweep; default = all 29.")
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--fv_root", type=str, default=str(ARTIFACTS_ROOT / "gptj_fv"),
                   help="Root holding <task>/<task>_function_vector.pt (task-specific top-10 FVs).")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output_root", type=str,
                   default=str(LABEL_GEOMETRY_DIR / "twoshot_fv_ablation_imitation"))
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    from baukit import TraceDict  # noqa: local import (model-side dep)

    f1, f2 = TASK_PAIRS[args.task_pair]
    directions = [(f1, f2, "f1", "f2"), (f2, f1, "f2", "f1")]
    print(f"task_pair={args.task_pair}  f1={f1} f2={f2}  directions={[f'{s}->{t}' for s, t, *_ in directions]}")

    o2i_f1, i2o_f1 = load_task(args.root_data_dir, f1)
    o2i_f2, i2o_f2 = load_task(args.root_data_dir, f2)
    i2o_by_task = {f1: i2o_f1, f2: i2o_f2}

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
    layer_names = [emb_name] + list(block_names)     # 29 residual read/edit points
    n_layers = len(layer_names)
    resid_dim = model_config["resid_dim"]
    name2li = {nm: li for li, nm in enumerate(layer_names)}
    sweep_layers = sorted(args.layers) if args.layers is not None else list(range(n_layers))

    def single(w):
        return len(tokenizer(" " + w).input_ids) == 1

    shared_out = sorted(set(o2i_f1) & set(o2i_f2))
    label_words = [w for w in shared_out if single(w)]
    shared_in = sorted(set(i2o_f1) & set(i2o_f2))
    query_pool = list(shared_in)
    label_set = list(label_words)
    print(f"label words (shared single-tok output): {len(label_words)}; query pool: {len(query_pool)}")

    # ---- build matched-label 2-shot prompt pairs (identical construction to the heatmap script) ----
    data = {"f1": {"ids": [], "pos": [], "str": []}, "f2": {"ids": [], "pos": [], "str": []}}
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
            assert ids[pos["label1"]] == expected_label_ids[0], f"L1 mismatch w={w!r} {tag}"
            assert ids[pos["label2"]] == expected_label_ids[1], f"L2 mismatch w={w!r} {tag}"
            ok[tag] = (ids, pos, prompt_string)
        for t in STEER_TOKENS:
            assert ok["f1"][0][ok["f1"][1][t]] == ok["f2"][0][ok["f2"][1][t]], f"steer token {t} differs (w={w!r})"
        for tag in ("f1", "f2"):
            ids, pos, ps = ok[tag]
            data[tag]["ids"].append(ids); data[tag]["pos"].append(pos); data[tag]["str"].append(ps)
        pair_meta.append({"L1": L1, "L2": L2, "query": q, "demo_inputs": demo_inputs})
        if len(pair_meta) >= args.n_pairs:
            break
    n_pairs = len(pair_meta)
    print(f"built {n_pairs} prompt pairs")
    pad_id = tokenizer.pad_token_id

    def make_chunks(tag):
        """Left-padded chunks; per-row positions for the 3 steer-site tokens (shifted by left-pad)."""
        ids_list, pos_list = data[tag]["ids"], data[tag]["pos"]
        chunks = []
        for s in range(0, len(ids_list), args.batch_size):
            sub_ids = ids_list[s:s + args.batch_size]
            sub_pos = pos_list[s:s + args.batch_size]
            mx = max(len(x) for x in sub_ids)
            inp = torch.full((len(sub_ids), mx), pad_id, dtype=torch.long)
            att = torch.zeros((len(sub_ids), mx), dtype=torch.long)
            pos = torch.zeros((len(sub_ids), N_STEER), dtype=torch.long)
            for r, ids in enumerate(sub_ids):
                pad = mx - len(ids)
                inp[r, pad:] = torch.tensor(ids, dtype=torch.long)
                att[r, pad:] = 1
                for ti, t in enumerate(STEER_TOKENS):
                    pos[r, ti] = pad + sub_pos[r][t]
            chunks.append({"input_ids": inp.to(device), "attention_mask": att.to(device),
                           "pos": pos.to(device), "n": len(sub_ids)})
        return chunks

    chunks_by_tag = {"f1": make_chunks("f1"), "f2": make_chunks("f2")}

    # ---- unsteered activation capture at the 3 steer-site tokens × 29 layers (for the steer vector) ----
    def capture_steer_acts(chunks):
        outs = []
        for ch in chunks:
            with TraceDict(model, layers=layer_names, retain_output=True) as td:
                model.transformer(input_ids=ch["input_ids"], attention_mask=ch["attention_mask"])
                B = ch["n"]
                rows = torch.arange(B, device=device)
                acts = torch.empty((B, N_STEER, n_layers, resid_dim), device=device, dtype=dtype)
                for li, nm in enumerate(layer_names):
                    out = td[nm].output
                    out = out[0] if isinstance(out, tuple) else out
                    for ti in range(N_STEER):
                        acts[:, ti, li, :] = out[rows, ch["pos"][:, ti], :]
            outs.append(acts)
        return torch.cat(outs, 0)      # [N, 3, 29, D]

    print("unsteered f1 pass...")
    f1_acts = capture_steer_acts(chunks_by_tag["f1"])
    print("unsteered f2 pass...")
    f2_acts = capture_steer_acts(chunks_by_tag["f2"])
    assert torch.isfinite(f1_acts).all() and torch.isfinite(f2_acts).all(), "non-finite acts"
    acts_by_tag = {"f1": f1_acts, "f2": f2_acts}

    # ---- answer-token first ids per pair, per task (context-dependent tokenization) ----
    def first_answer_ids(tag, task):
        i2o = i2o_by_task[task]
        ids0 = []
        for k in range(n_pairs):
            q = pair_meta[k]["query"]
            aid = get_answer_id(data[tag]["str"][k], " " + i2o[q], tokenizer)
            ids0.append(int(aid[0]))
        return ids0

    out_dir = Path(args.output_root) / args.task_pair
    out_dir.mkdir(parents=True, exist_ok=True)

    QF_COL = STEER_TOKENS.index("qfinal")   # ablation/read column in the position tensor

    def edit_hook(steer_si, steer_layer, ablate, alpha, sv_dev, u_dev, p_steer, p_qf, rows0):
        # steer_si: index into STEER_TOKENS of the SINGLE site to inject at (-1 = no steer);
        # steer_layer: the SINGLE residual layer to inject at. p_steer = that site's positions.
        # ablate: project out u at qfinal at EVERY layer (all points at the query pre-label token).
        # 2-arg (output, layer_name) closure (baukit mis-binds extra args -- see DECISIONS 2026-06-11).
        def hook(output, layer_name):
            li = name2li.get(layer_name)
            if li is None:
                return output
            h = output[0] if isinstance(output, tuple) else output
            if steer_si >= 0 and li == steer_layer:
                h[rows0, p_steer] += (alpha * sv_dev[steer_si, li]).to(h.dtype)
            if ablate:
                hq = h[rows0, p_qf].float()                       # [B, D]
                coeff = hq @ u_dev                                 # [B]
                h[rows0, p_qf] = (hq - coeff[:, None] * u_dev[None, :]).to(h.dtype)
            return output
        return hook

    def run_condition(src_chunks, steer_si, steer_layer, ablate, alpha, sv_dev, u_dev):
        """qfinal logits [N, vocab] under (steer at site steer_si @ layer steer_layer, ablate)."""
        logits_out = []
        for ch in src_chunks:
            rows0 = torch.arange(ch["n"], device=device)
            p_steer = ch["pos"][:, steer_si if steer_si >= 0 else 0]
            hook = edit_hook(steer_si, steer_layer, ablate, alpha, sv_dev, u_dev,
                             p_steer, ch["pos"][:, QF_COL], rows0)
            with TraceDict(model, layers=layer_names, edit_output=hook):
                out = model.transformer(input_ids=ch["input_ids"], attention_mask=ch["attention_mask"])
            last = out[0] if isinstance(out, tuple) else out.last_hidden_state
            logits_last = model.lm_head(last[:, -1, :]).float()    # [B, vocab]; qfinal is final column
            logits_out.append(logits_last)
        return torch.cat(logits_out, 0)

    def dlogit(logits, a_tgt_ids, a_src_ids):
        idx = torch.arange(logits.shape[0], device=logits.device)
        return (logits[idx, a_tgt_ids] - logits[idx, a_src_ids]).cpu().numpy()

    summary = {"task_pair": args.task_pair, "f1": f1, "f2": f2, "n_pairs": n_pairs,
               "n_layers": n_layers, "alphas": args.alphas, "steer_tokens": STEER_TOKENS,
               "steer_mode": "single (token position, layer) injection; separate layer-sweep per token",
               "ablate_site": "qfinal, all layers", "fv_root": str(args.fv_root),
               "metric": "logit(a_tgt) - logit(a_src) at qfinal; a_task = task(query)",
               "curves": {"steer": "Δlogit vs injection layer (steer one site, no ablate)",
                          "steer_ablate": "Δlogit vs injection layer (steer one site, F_perp ablated at qfinal)",
                          "clean": "scalar baseline (no steer, no ablate)",
                          "ablate": "scalar baseline (no steer, ablate)"},
               "directions": {}}

    for src_task, tgt_task, src_tag, tgt_tag in directions:
        dir_name = f"{src_task}_to_{tgt_task}"
        src_h = acts_by_tag[src_tag]
        tgt_h = acts_by_tag[tgt_tag]
        src_chunks = chunks_by_tag[src_tag]

        steer_vec = (tgt_h.float() - src_h.float()).mean(0)          # [3, 29, D]
        sv_dev = steer_vec.to(device)
        steer_norms = torch.linalg.norm(steer_vec, dim=-1).cpu().numpy()  # [3, 29]

        F = load_fv(args.fv_root, src_task, resid_dim, device)
        Fp = load_fv(args.fv_root, tgt_task, resid_dim, device)
        assert not torch.allclose(F, Fp), f"FV(src)==FV(tgt) for {dir_name}"
        Fhat = F / F.norm()
        F_perp = Fp - (Fp @ Fhat) * Fhat
        fp_norm = float(F_perp.norm())
        assert fp_norm > 1e-6, f"F_perp ~ 0 for {dir_name} (target FV nearly parallel to source FV)"
        u_dev = (F_perp / F_perp.norm()).float()
        cos_FFp = float((Fhat @ (Fp / Fp.norm())))
        print(f"\n=== direction {dir_name} (inject into {src_tag}) ===")
        print(f"  cos(F,F')={cos_FFp:+.3f}  ||F_perp||={fp_norm:.3f}")

        a_src_ids = torch.tensor(first_answer_ids(src_tag, src_task), device=device)
        a_tgt_ids = torch.tensor(first_answer_ids(src_tag, tgt_task), device=device)
        keep = (a_src_ids != a_tgt_ids).cpu().numpy()
        n_keep = int(keep.sum())

        # steering-free baselines (steer_si=-1), site/alpha-independent -> compute once
        d_clean = dlogit(run_condition(src_chunks, -1, -1, False, 0.0, sv_dev, u_dev), a_tgt_ids, a_src_ids)
        d_ablate = dlogit(run_condition(src_chunks, -1, -1, True, 0.0, sv_dev, u_dev), a_tgt_ids, a_src_ids)
        clean_mean = float(d_clean[keep].mean())
        ablate_mean = float(d_ablate[keep].mean())

        dsum = {"src_task": src_task, "tgt_task": tgt_task, "inject_into": src_tag,
                "n_keep": n_keep, "cos_F_Fprime": cos_FFp, "F_perp_norm": fp_norm,
                "clean_mean": clean_mean, "ablate_mean": ablate_mean,
                "steer_vec_norm_by_token_layer": {STEER_TOKENS[t]: steer_norms[t].tolist() for t in range(N_STEER)},
                "tokens": {}}

        def mean_sem(row):
            rk = row[keep]
            return float(np.nanmean(rk)), float(np.nanstd(rk, ddof=1) / max(1, np.sqrt(len(rk))))

        for alpha in args.alphas:
            for si, tok in enumerate(STEER_TOKENS):
                csv_path = out_dir / f"{dir_name}_{tok}_alpha{alpha:g}_layersweep.csv"
                npz_path = out_dir / f"{dir_name}_{tok}_alpha{alpha:g}_perpair.npz"
                if csv_path.exists() and npz_path.exists() and not args.overwrite:
                    print(f"  {tok} α={alpha:g}: exists, skip")
                    z = np.load(npz_path)
                    steer_curve, steerabl_curve = z["steer"], z["steer_ablate"]
                else:
                    steer_curve = np.full((n_layers, n_pairs), np.nan)
                    steerabl_curve = np.full((n_layers, n_pairs), np.nan)
                    for li in sweep_layers:
                        steer_curve[li] = dlogit(run_condition(src_chunks, si, li, False, alpha, sv_dev, u_dev),
                                                 a_tgt_ids, a_src_ids)
                        steerabl_curve[li] = dlogit(run_condition(src_chunks, si, li, True, alpha, sv_dev, u_dev),
                                                    a_tgt_ids, a_src_ids)
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    np.savez(npz_path, clean=d_clean, ablate=d_ablate,
                             steer=steer_curve, steer_ablate=steerabl_curve)

                steer_mean = np.array([mean_sem(steer_curve[li])[0] for li in range(n_layers)])
                steer_sem = np.array([mean_sem(steer_curve[li])[1] for li in range(n_layers)])
                sa_mean = np.array([mean_sem(steerabl_curve[li])[0] for li in range(n_layers)])
                sa_sem = np.array([mean_sem(steerabl_curve[li])[1] for li in range(n_layers)])
                steer_gain = steer_mean - clean_mean
                sa_gain = sa_mean - ablate_mean
                with np.errstate(divide="ignore", invalid="ignore"):
                    retention = np.where(np.abs(steer_gain) > 1e-9, sa_gain / steer_gain, np.nan)

                with open(csv_path, "w") as fh:
                    fh.write("layer,clean_mean,ablate_mean,steer_mean,steer_sem,steer_gain,"
                             "steer_ablate_mean,steer_ablate_sem,steer_ablate_gain,retention\n")
                    for li in range(n_layers):
                        fh.write(f"{li},{clean_mean:.6f},{ablate_mean:.6f},{steer_mean[li]:.6f},"
                                 f"{steer_sem[li]:.6f},{steer_gain[li]:.6f},{sa_mean[li]:.6f},"
                                 f"{sa_sem[li]:.6f},{sa_gain[li]:.6f},{retention[li]:.6f}\n")

                valid = np.array([li in sweep_layers for li in range(n_layers)])
                peak_li = int(np.nanargmax(np.where(valid, steer_gain, -np.inf)))
                eff = valid & (steer_gain > 0.5)          # "effective" injection layers
                mean_ret_eff = float(np.nanmean(retention[eff])) if eff.any() else float("nan")
                dsum["tokens"][f"{tok}_alpha{alpha:g}"] = {
                    "token": tok, "alpha": alpha,
                    "peak_steer_layer": peak_li,
                    "peak_steer_gain": float(steer_gain[peak_li]),
                    "peak_steer_ablate_gain": float(sa_gain[peak_li]),
                    "peak_retention": float(retention[peak_li]),
                    "n_effective_layers": int(eff.sum()),
                    "mean_retention_effective_layers": mean_ret_eff,
                    "steer_gain_by_layer": steer_gain.tolist(),
                    "steer_ablate_gain_by_layer": sa_gain.tolist()}
                print(f"  {tok:>7} α={alpha:g}: peak steer L{peak_li} gain={steer_gain[peak_li]:+.3f} "
                      f"| steer+abl gain={sa_gain[peak_li]:+.3f} retention={retention[peak_li]:.2f} "
                      f"| mean ret over {int(eff.sum())} eff. L={mean_ret_eff:.2f}")

        summary["directions"][dir_name] = dsum
        del F, Fp, Fhat, F_perp, u_dev, sv_dev

    with open(out_dir / f"{args.task_pair}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nDONE -> {out_dir}")


if __name__ == "__main__":
    main()
