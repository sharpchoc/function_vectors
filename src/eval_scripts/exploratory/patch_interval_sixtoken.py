"""
Six-token INTERVAL activation-patching on TWO-shot paired prompts (GPT-J-6B; baukit).

Studies which token positions carry the function signal to which downstream positions and to the
final output, by *patching* (not Delta-vector steering). For a fixed task pair with a source->target
direction (base = source; patch-in = target):

  6 token positions (prompt order; all single-token => fixed indices, every prompt same length):
      t0 L1   = demo-1 label
      t1 in2  = demo-2 input (last subword)
      t2 pre2 = demo-2 pre-label "A:"
      t3 L2   = demo-2 label
      t4 qin  = query input (last subword)
      t5 qfin = query-final "A:" (output position)

  Interval pin at residual entries 6..28 (29-entry stack: 0 = embedding transformer.drop,
  entry b+1 = output of block transformer.h.b). For each ordered pair (i, j), i < j:
    run the BASE prompt and at every entry 6..28 OVERWRITE (assign):
        token i <- TARGET prompt's activation at i   (i carries the target function for L6+)
        token j <- BASE   prompt's clean activation at j  (j pinned to original; blocks relay)
    Tokens strictly between i and j, and after j, recompute freely.

  Metric 1 (output logit flip), read at qfin of the patched run, per (i,j):
        logit_diff = logit(target_gold_first_tok) - logit(source_gold_first_tok)
     reported as mean over pairs; also the no-patch baseline (single mean scalar) and the
     shift = steered - baseline. -> 6x6 upper-triangle grid per task.

  Metric 2 (downstream cosine propagation), per (i,j), per downstream k>j, per entry L:
        dcos(k,L) = mean_pairs[ cos(steered_k[L], target_k[L]) - cos(base_k[L], target_k[L]) ]
     valid for L>=7 (entry 6 of k is unaffected by construction -> ~0). -> [6,6,6,29] array.

Efficiency: all pairs batched per forward (the expensive axis), fp16 on GPU, the two clean passes
computed once and reused, base_cos precomputed once, each (i,j) a single fused patch+read forward.
~ 2 + 15 = 17 short forwards per task.

baukit imported inside main (precedent: steer_label_cos_heatmap.py).
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt, get_token_meta_labels
from utils.paths import LABEL_GEOMETRY_DIR

TASK_PAIRS = {
    "antonym_synonym": ("antonym", "synonym"),
    "next_number_digits_prev_number_digits": ("next_number_digits", "prev_number_digits"),
}
# task_pair -> (source/base task, target/patch-in task)  [patch base toward target]
SRC_TGT = {
    "antonym_synonym": ("antonym", "synonym"),                                # base antonym, patch synonym
    "next_number_digits_prev_number_digits": ("prev_number_digits", "next_number_digits"),  # prev -> next
}

PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}

TOKEN_NAMES = ["demo1 label", "demo2 input", "demo2 pre label", "demo2 label",
               "query input", "query pre label"]   # t0..t5, prompt order
PATCH_FROM_ENTRY = 6   # patch residual entries 6..28 ("L6 and higher"; 0=embedding)

# --- baukit-free position helpers (inlined, mirror steer_label_cos_heatmap.py) ---
LABEL_TOKEN_RE = re.compile(r"^demonstration_(\d+)_label_token$")


def make_token_record(token_role, icl_example_index, token):
    p, t, l = token
    return {"token_role": token_role, "icl_example_index": icl_example_index,
            "token_position": int(p), "token_text": t, "token_label": l}


def selected_token_records(token_labels):
    by_pos = {int(p): (p, t, l) for p, t, l in token_labels}
    groups = {}
    for p, t, l in token_labels:
        m = LABEL_TOKEN_RE.match(l)
        if m:
            groups.setdefault(int(m.group(1)), []).append((p, t, l))
    recs = []
    for icl in sorted(groups):
        lab = sorted(groups[icl], key=lambda x: x[0])
        pre = int(lab[0][0]) - 1
        if pre < 0 or pre not in by_pos:
            raise ValueError(f"no pre-label token for ICL {icl}")
        recs += [make_token_record("pre_label_token", icl, by_pos[pre]),
                 make_token_record("last_label_token", icl, lab[-1])]
    fc = [x for x in token_labels if x[2] == "query_predictive_token"]
    final = max(fc, key=lambda x: x[0]) if fc else token_labels[-1]
    recs.append(make_token_record("last_prompt_token", None, final))
    return recs


def get_six_positions(token_labels):
    """Absolute indices of [L1, in2, pre2, L2, qin, qfin] (strictly increasing)."""
    recs = selected_token_records(token_labels)
    role = {(r["token_role"], r["icl_example_index"]): r["token_position"] for r in recs}
    demo2_in = max((int(p) for p, t, l in token_labels if l == "demonstration_2_token"), default=None)
    query_in = max((int(p) for p, t, l in token_labels if l == "query_demonstration_token"), default=None)
    pos = [role[("last_label_token", 1)], demo2_in, role[("pre_label_token", 2)],
           role[("last_label_token", 2)], query_in, role[("last_prompt_token", None)]]
    if any(p is None for p in pos):
        raise ValueError(f"missing one of the 6 positions: {pos}")
    if pos != sorted(pos) or len(set(pos)) != 6:
        raise ValueError(f"6 positions not strictly increasing/unique: {pos}")
    return pos
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


def build_two_shot(demo_inputs, labels, query):
    return word_pairs_to_prompt_data(
        {"input": list(demo_inputs), "output": list(labels)},
        query_target_pair={"input": query, "output": query},
        prepend_bos_token=False, prefixes=PREFIXES, separators=SEPARATORS, prepend_space=True,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Six-token interval activation-patching (2-shot paired).")
    p.add_argument("--task_pair", choices=sorted(TASK_PAIRS), default="antonym_synonym")
    p.add_argument("--max_pairs", type=int, default=None, help="Cap prompt pairs (smoke tests).")
    p.add_argument("--patch_from_entry", type=int, default=6,
                   help="First residual entry to patch (6 = 'L6 and above'; 0 = all layers incl. embedding).")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--output_root", type=str,
                   default=str(LABEL_GEOMETRY_DIR / "twoshot" / "interval_patch_sixtoken"))
    return p.parse_args()


def main():
    args = parse_args()
    global PATCH_FROM_ENTRY
    PATCH_FROM_ENTRY = args.patch_from_entry
    regime = "all_layers" if PATCH_FROM_ENTRY == 0 else f"L{PATCH_FROM_ENTRY}_and_above"
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    from baukit import TraceDict  # noqa: local import (model-side dep)

    f1, f2 = TASK_PAIRS[args.task_pair]
    src_task, tgt_task = SRC_TGT[args.task_pair]
    print(f"task_pair={args.task_pair}  base(src)={src_task}  patch-in(tgt)={tgt_task}")

    o2i_src, i2o_src = load_task(args.root_data_dir, src_task)
    o2i_tgt, i2o_tgt = load_task(args.root_data_dir, tgt_task)

    print("Loading model...")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision)
    model.eval()
    device = args.device
    dtype = next(model.parameters()).dtype
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    block_names = model_config["layer_hook_names"]            # ['transformer.h.0', ...]
    assert model_config["n_layers"] == len(block_names) == 28
    emb_name = "transformer.drop"
    layer_names = [emb_name] + list(block_names)              # 29 ordered residual entries
    n_layers = len(layer_names)                              # 29
    name2idx = {nm: i for i, nm in enumerate(layer_names)}
    resid_dim = model_config["resid_dim"]

    def single(w):
        return len(tokenizer(" " + w).input_ids) == 1

    shared_out = sorted(set(o2i_src) & set(o2i_tgt))
    label_words = [w for w in shared_out if single(w)]
    shared_in = sorted(set(i2o_src) & set(i2o_tgt))
    query_pool = list(shared_in)
    print(f"label words (shared single-tok output): {len(label_words)}; query pool: {len(query_pool)}")

    def single_inputs(o2i, L):
        return [x for x in o2i.get(L, []) if single(x)]

    # ---- build matched-label 2-shot prompt pairs (mirror capture_and_grade_twoshot_paired) ----
    # Positions are computed PER ROW (in-context tokenization can shift the query block by a token),
    # then converted to padded indices at batch time. Role-alignment (the 6 named positions) is what
    # matters for patching/reading, not absolute indices.
    base_ids, tgt_ids = [], []
    base_pos6, tgt_pos6 = [], []
    src_gold_id, tgt_gold_id = [], []
    pair_meta = []
    for w in label_words:
        rng = stable_rng(args.seed, args.task_pair, w)
        L1 = w
        cand_L2 = [x for x in label_words if x != L1]
        if not cand_L2:
            continue
        L2 = str(rng.choice(cand_L2))
        labels = [L1, L2]
        # single-token demo inputs per function (verified: always available)
        try:
            src_in = [str(rng.choice(single_inputs(o2i_src, L1))), str(rng.choice(single_inputs(o2i_src, L2)))]
            tgt_in = [str(rng.choice(single_inputs(o2i_tgt, L1))), str(rng.choice(single_inputs(o2i_tgt, L2)))]
        except ValueError:
            continue  # no single-token input for some label (should not happen)
        forbidden = {L1, L2, *src_in, *tgt_in}
        cand_q = [q for q in query_pool if q not in forbidden and single(q)]
        if not cand_q:
            continue
        q = str(rng.choice(cand_q))

        out = {}
        ok = True
        for tag, demo_in in (("src", src_in), ("tgt", tgt_in)):
            pd = build_two_shot(demo_in, labels, q)
            token_labels, prompt_string = get_token_meta_labels(
                pd, tokenizer, query=q, prepend_bos=model_config["prepend_bos"])
            pos6 = get_six_positions(token_labels)
            ids = tokenizer(prompt_string).input_ids
            assert pos6[-1] == len(ids) - 1, "qfin not last token"
            out[tag] = (ids, pos6)
        s_ids, s_pos = out["src"]
        t_ids, t_pos = out["tgt"]
        # paired invariant: the non-input roles (L1, pre2, L2, qin, qfin) are byte-identical across
        # base/target at their respective role positions (only the 2 demo INPUTS differ by function).
        for ridx in (0, 2, 3, 4, 5):
            assert s_ids[s_pos[ridx]] == t_ids[t_pos[ridx]], \
                f"role {TOKEN_NAMES[ridx]} differs across functions (w={w!r})"

        base_ids.append(s_ids); base_pos6.append(s_pos)
        tgt_ids.append(t_ids); tgt_pos6.append(t_pos)
        src_gold_id.append(tokenizer(" " + i2o_src[q]).input_ids[0])
        tgt_gold_id.append(tokenizer(" " + i2o_tgt[q]).input_ids[0])
        pair_meta.append({"L1": L1, "L2": L2, "query": q, "src_in": src_in, "tgt_in": tgt_in})
        if args.max_pairs is not None and len(pair_meta) >= args.max_pairs:
            break

    N = len(pair_meta)
    print(f"built {N} prompt pairs (per-row positions; query block can shift by a token)")
    src_gold_t = torch.tensor(src_gold_id, device=device)
    tgt_gold_t = torch.tensor(tgt_gold_id, device=device)
    pad_id = tokenizer.pad_token_id

    # ---- chunking: left-pad; per-row padded 6-position table [B,6]; qfin is always the last token ----
    def make_chunks(ids_list, pos6_list, with_gold):
        chunks, s = [], 0
        while s < N:
            sub_ids = ids_list[s:s + args.batch_size]
            sub_pos = pos6_list[s:s + args.batch_size]
            mx = max(len(x) for x in sub_ids)
            B = len(sub_ids)
            inp = torch.full((B, mx), pad_id, dtype=torch.long)
            att = torch.zeros((B, mx), dtype=torch.long)
            pos = torch.zeros((B, 6), dtype=torch.long)
            for r, ids in enumerate(sub_ids):
                pad = mx - len(ids)
                inp[r, pad:] = torch.tensor(ids, dtype=torch.long)
                att[r, pad:] = 1
                pos[r] = torch.tensor(sub_pos[r], dtype=torch.long) + pad
            ch = {"input_ids": inp.to(device), "attention_mask": att.to(device),
                  "pos6": pos.to(device), "n": B, "s": s}
            if with_gold:
                ch["src_gold"] = src_gold_t[s:s + B]
                ch["tgt_gold"] = tgt_gold_t[s:s + B]
            chunks.append(ch)
            s += B
        return chunks

    base_chunks = make_chunks(base_ids, base_pos6, with_gold=True)
    tgt_chunks = make_chunks(tgt_ids, tgt_pos6, with_gold=False)

    def read_six(td, ch):
        """[B, 6, 29, D] acts at this chunk's per-row 6 positions across all entries."""
        B = ch["n"]
        pos = ch["pos6"]                                   # [B,6]
        rows = torch.arange(B, device=device).unsqueeze(1)  # [B,1]
        a = torch.empty((B, 6, n_layers, resid_dim), device=device, dtype=dtype)
        for li, nm in enumerate(layer_names):
            o = td[nm].output
            o = o[0] if isinstance(o, tuple) else o
            a[:, :, li, :] = o[rows, pos, :]               # [B,6,D]
        return a

    def logit_diff_from(out_obj, ch):
        logits = out_obj.logits[:, -1, :]                  # qfin = last token (left-padded)
        ar = torch.arange(ch["n"], device=device)
        return logits[ar, ch["tgt_gold"]] - logits[ar, ch["src_gold"]]

    # ---- clean passes (once) ----
    def run_clean(chunks, with_logits):
        acts, lds = [], []
        for ch in chunks:
            with TraceDict(model, layers=layer_names, retain_output=True) as td:
                out_obj = model(input_ids=ch["input_ids"], attention_mask=ch["attention_mask"])
                acts.append(read_six(td, ch))
                if with_logits:
                    lds.append(logit_diff_from(out_obj, ch))
        return torch.cat(acts, 0), (torch.cat(lds, 0) if with_logits else None)

    print("clean base pass...")
    base_act, base_ld = run_clean(base_chunks, with_logits=True)     # [N,6,29,D], [N]
    print("clean target pass...")
    tgt_act, _ = run_clean(tgt_chunks, with_logits=False)            # [N,6,29,D]
    assert torch.isfinite(base_act).all() and torch.isfinite(tgt_act).all()

    baseline_logit_diff = float(base_ld.float().mean())
    base_cos = F.cosine_similarity(base_act.float(), tgt_act.float(), dim=-1).mean(0).cpu().numpy()  # [6,29]
    print(f"baseline mean logit_diff (tgt-src gold) = {baseline_logit_diff:+.4f}  "
          f"(expect negative: base prompt favors its own answer)")

    # ---- patched passes: one fused forward per (i,j) ----
    def make_patch_hook(base_slice, tgt_slice, pos_i_vec, pos_j_vec, idx_i, idx_j, rows):
        # EXACT 2-arg (output, layer_name) closure (baukit mis-binds extra params; DECISIONS 2026-06-11).
        # pos_*_vec are per-row padded positions [B]; overwrite at entries >= PATCH_FROM_ENTRY.
        def hook(output, layer_name):
            e = name2idx.get(layer_name, -1)
            if e < PATCH_FROM_ENTRY:
                return output
            h = output[0] if isinstance(output, tuple) else output
            h[rows, pos_i_vec, :] = tgt_slice[:, idx_i, e, :]    # token i -> target  (switch on)
            h[rows, pos_j_vec, :] = base_slice[:, idx_j, e, :]   # token j -> base    (pin to original)
            return output
        return hook

    pairs = list(combinations(range(6), 2))   # 15 ordered (i<j)
    logit_grid = np.full((6, 6), np.nan)       # [i, j] mean steered logit_diff
    down = np.full((6, 6, 6, n_layers), np.nan)  # [i, j, k, L] mean dcos, k>j

    for (i, j) in pairs:
        cos_sum = torch.zeros((6, n_layers), device=device, dtype=torch.float32)
        lds = []
        for ch in base_chunks:
            n, s = ch["n"], ch["s"]
            rows = torch.arange(n, device=device)
            hook = make_patch_hook(base_act[s:s + n], tgt_act[s:s + n],
                                   ch["pos6"][:, i], ch["pos6"][:, j], i, j, rows)
            with TraceDict(model, layers=layer_names, edit_output=hook, retain_output=True) as td:
                out_obj = model(input_ids=ch["input_ids"], attention_mask=ch["attention_mask"])
                lds.append(logit_diff_from(out_obj, ch))
                sact = read_six(td, ch)                                      # [n,6,29,D]
                cos_sum += F.cosine_similarity(sact.float(), tgt_act[s:s + n].float(), dim=-1).sum(0)
        logit_grid[i, j] = float(torch.cat(lds).float().mean())
        steered_cos = (cos_sum / N).cpu().numpy()                            # [6,29]
        for k in range(j + 1, 6):                                            # downstream only
            down[i, j, k, :] = steered_cos[k, :] - base_cos[k, :]
        print(f"  patch i={TOKEN_NAMES[i]:>12} j={TOKEN_NAMES[j]:<12}: "
              f"logit_diff {logit_grid[i, j]:+.4f} (shift {logit_grid[i, j]-baseline_logit_diff:+.4f})")

    # invariants
    assert np.isfinite(logit_grid[np.triu_indices(6, k=1)]).all(), "non-finite upper-triangle logit cell"
    z6 = down[..., PATCH_FROM_ENTRY]                  # entry == patch onset of k => ~0 (unaffected yet)
    assert np.nanmax(np.abs(z6)) < 1e-3, f"dcos@entry{PATCH_FROM_ENTRY} not ~0: {np.nanmax(np.abs(z6))}"

    # ---- save ----
    out_dir = Path(args.output_root) / regime
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    tag = args.task_pair
    shift_grid = logit_grid - baseline_logit_diff
    np.save(out_dir / f"{tag}_logit_grid.npy", logit_grid)
    np.save(out_dir / f"{tag}_logit_shift_grid.npy", shift_grid)
    np.save(out_dir / f"{tag}_downstream_dcos.npy", down)
    np.savetxt(out_dir / f"{tag}_logit_shift_grid.csv", shift_grid, delimiter=",",
               header="rows=i(switch-on token), cols=j(pin token); value=mean steered-baseline logit_diff",
               comments="")

    # peak cells
    fi, fj = np.unravel_index(np.nanargmax(shift_grid), shift_grid.shape)
    summary = {
        "task_pair": tag, "src_task": src_task, "tgt_task": tgt_task, "n_pairs": N,
        "token_names": TOKEN_NAMES, "positions_example_row0": dict(zip(TOKEN_NAMES, base_pos6[0])),
        "patch_from_entry": PATCH_FROM_ENTRY, "n_layers": n_layers,
        "baseline_logit_diff": baseline_logit_diff,
        "logit_grid": logit_grid.tolist(), "logit_shift_grid": shift_grid.tolist(),
        "peak_shift": float(shift_grid[fi, fj]),
        "peak_i": TOKEN_NAMES[fi], "peak_j": TOKEN_NAMES[fj],
        "mean_baseline_cos_by_pos_layer": base_cos.tolist(),
        "metric": "M1 logit(tgt_gold)-logit(src_gold) at qfin; M2 dcos(k,L)=cos(steered_k,tgt_k)-cos(base_k,tgt_k)",
    }
    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- quick inline 6x6 shift heatmap (full plotting in plot_patch_interval_sixtoken.py) ----
    vmax = float(np.nanmax(np.abs(shift_grid))) or 1e-6
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(shift_grid, origin="upper", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(6)); ax.set_xticklabels(TOKEN_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(TOKEN_NAMES)
    ax.set_xlabel("j  (token pinned to original)")
    ax.set_ylabel("i  (token switched to target)")
    ax.set_title(f"interval patch {src_task}→{tgt_task}  (n={N})\n"
                 f"mean Δ logit(tgt−src) vs no-patch baseline {baseline_logit_diff:+.2f}", fontsize=9)
    for i in range(6):
        for j in range(6):
            if not np.isnan(shift_grid[i, j]):
                ax.text(j, i, f"{shift_grid[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="steered − baseline logit_diff")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"{tag}_logit_shift_heatmap.png", dpi=140)
    plt.close(fig)

    print(f"DONE -> {out_dir}\n  peak shift {shift_grid[fi, fj]:+.4f} @ i={TOKEN_NAMES[fi]} j={TOKEN_NAMES[fj]}")


if __name__ == "__main__":
    main()
