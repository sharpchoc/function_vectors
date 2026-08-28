"""
Do ONLY the demo LABEL tokens drive the output? ISOLATED label-token patching (GPT-J-6B; baukit).
Follow-up to patch_interval_sixtoken.py.

Run the BASE/source prompt; at residual entries 6..28 overwrite token activations with the TARGET
prompt's, then read the output logit gap at the query-final token. Two modes per patch set:

  open      -- overwrite only the patched positions -> target; everything else recomputes freely
               (so the patched info can also be RELAYED to the output via in-between/query tokens).
  isolated  -- overwrite the patched positions -> target AND pin EVERY other token (all non-patched,
               non-output positions) back to its BASE value. Only the patched positions carry target,
               and the only route to the output is the DIRECT patched->output attention path.

Patch sets (positions overwritten ← target; output/query-final token is never patched):
  demo2_prelabel  -- {demo2 pre label}            (the "second pre-label" alternative)
  both_labels     -- {demo1 label, demo2 label}   (headline; isolated answers "do ONLY labels drive it")

Metric (logit-only): logit_diff = logit(tgt_gold_1) - logit(src_gold_1) at query-final, mean over all
pairs; recovery = (mean_patched - baseline) / (ceiling - baseline). baseline = source prompt; ceiling =
real target prompt. The gap (open recovery - isolated recovery) = share of the label effect that is
relayed through other tokens rather than read directly.
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
SRC_TGT = {
    "antonym_synonym": ("antonym", "synonym"),
    "next_number_digits_prev_number_digits": ("prev_number_digits", "next_number_digits"),
}

PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}

TOKEN_NAMES = ["demo1 label", "demo2 input", "demo2 pre label", "demo2 label",
               "query input", "query pre label"]   # t0..t5, prompt order
PATCH_FROM_ENTRY = 6   # patch residual entries 6..28 ("L6 and higher"; 0=embedding)

# patch set name -> role indices into the 6-position table (target overwrites these)
PATCH_SETS = [
    ("demo2_prelabel", [2]),
    ("both_labels", [0, 3]),
]
MODES = ["open", "isolated"]

# --- baukit-free position helpers (inlined, mirror patch_interval_sixtoken.py) ---
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
    p = argparse.ArgumentParser(description="Isolated label-token patching: do ONLY the labels drive the output?")
    p.add_argument("--task_pair", choices=sorted(TASK_PAIRS), default="antonym_synonym")
    p.add_argument("--max_pairs", type=int, default=None, help="Cap prompt pairs (smoke tests).")
    p.add_argument("--patch_from_entry", type=int, default=6,
                   help="First residual entry to patch (6 = 'L6 and above'; 0 = all layers incl. embedding).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--output_root", type=str,
                   default=str(LABEL_GEOMETRY_DIR / "twoshot" / "label_follow_patch"))
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

    block_names = model_config["layer_hook_names"]
    assert model_config["n_layers"] == len(block_names) == 28
    emb_name = "transformer.drop"
    layer_names = [emb_name] + list(block_names)
    n_layers = len(layer_names)
    name2idx = {nm: i for i, nm in enumerate(layer_names)}
    resid_dim = model_config["resid_dim"]
    patch_entries = list(range(PATCH_FROM_ENTRY, n_layers))   # 6..28
    n_pe = len(patch_entries)

    def single(w):
        return len(tokenizer(" " + w).input_ids) == 1

    shared_out = sorted(set(o2i_src) & set(o2i_tgt))
    label_words = [w for w in shared_out if single(w)]
    shared_in = sorted(set(i2o_src) & set(i2o_tgt))
    query_pool = list(shared_in)
    print(f"label words (shared single-tok output): {len(label_words)}; query pool: {len(query_pool)}")

    def single_inputs(o2i, L):
        return [x for x in o2i.get(L, []) if single(x)]

    # ---- build matched-label 2-shot prompt pairs (identical to patch_interval_sixtoken) ----
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
        try:
            src_in = [str(rng.choice(single_inputs(o2i_src, L1))), str(rng.choice(single_inputs(o2i_src, L2)))]
            tgt_in = [str(rng.choice(single_inputs(o2i_tgt, L1))), str(rng.choice(single_inputs(o2i_tgt, L2)))]
        except ValueError:
            continue
        forbidden = {L1, L2, *src_in, *tgt_in}
        cand_q = [q for q in query_pool if q not in forbidden and single(q)]
        if not cand_q:
            continue
        q = str(rng.choice(cand_q))

        out = {}
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
    print(f"built {N} prompt pairs")
    src_gold_t = torch.tensor(src_gold_id, device=device)
    tgt_gold_t = torch.tensor(tgt_gold_id, device=device)
    pad_id = tokenizer.pad_token_id

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
    tgt_chunks = make_chunks(tgt_ids, tgt_pos6, with_gold=True)

    def read_six(td, ch):
        B = ch["n"]
        pos = ch["pos6"]
        rows = torch.arange(B, device=device).unsqueeze(1)
        a = torch.empty((B, 6, n_layers, resid_dim), device=device, dtype=dtype)
        for li, nm in enumerate(layer_names):
            o = td[nm].output
            o = o[0] if isinstance(o, tuple) else o
            a[:, :, li, :] = o[rows, pos, :]
        return a

    def logit_diff_from(out_obj, ch):
        logits = out_obj.logits[:, -1, :]                  # qfin = last token (left-padded)
        ar = torch.arange(ch["n"], device=device)
        return logits[ar, ch["tgt_gold"]] - logits[ar, ch["src_gold"]]

    # ---- clean passes ----
    print("clean base pass (logit + full residual stack at entries 6..28, all positions)...")
    base_ld, base_full_chunks = [], []
    for ch in base_chunks:
        with TraceDict(model, layers=layer_names, retain_output=True) as td:
            out_obj = model(input_ids=ch["input_ids"], attention_mask=ch["attention_mask"])
            base_ld.append(logit_diff_from(out_obj, ch))
            B, seq = ch["n"], ch["input_ids"].shape[1]
            bf = torch.empty((B, seq, n_pe, resid_dim), device=device, dtype=dtype)
            for k, e in enumerate(patch_entries):
                o = td[layer_names[e]].output
                o = o[0] if isinstance(o, tuple) else o
                bf[:, :, k, :] = o
            base_full_chunks.append(bf)
    base_ld = torch.cat(base_ld, 0)

    print("clean target pass (logit + acts at the 6 positions)...")
    tgt_ld, tgt_acts = [], []
    for ch in tgt_chunks:
        with TraceDict(model, layers=layer_names, retain_output=True) as td:
            out_obj = model(input_ids=ch["input_ids"], attention_mask=ch["attention_mask"])
            tgt_ld.append(logit_diff_from(out_obj, ch))
            tgt_acts.append(read_six(td, ch))
    tgt_ld = torch.cat(tgt_ld, 0)
    tgt_act = torch.cat(tgt_acts, 0)               # [N,6,29,D]
    assert torch.isfinite(tgt_act).all()

    baseline_logit_diff = float(base_ld.float().mean())
    ceiling_logit_diff = float(tgt_ld.float().mean())
    print(f"baseline (source prompt) mean logit_diff = {baseline_logit_diff:+.4f}")
    print(f"ceiling  (target prompt) mean logit_diff = {ceiling_logit_diff:+.4f}")

    # ---- hooks ----
    def make_open_hook(tgt_slice, pos6, S, rows):
        def hook(output, layer_name):
            e = name2idx.get(layer_name, -1)
            if e < PATCH_FROM_ENTRY:
                return output
            h = output[0] if isinstance(output, tuple) else output
            for s_role in S:
                h[rows, pos6[:, s_role], :] = tgt_slice[:, s_role, e, :]
            return output
        return hook

    def make_isolated_hook(base_full, tgt_slice, pos6, S, rows):
        # freeze EVERY token except the output column to base, then set patched roles to target.
        def hook(output, layer_name):
            e = name2idx.get(layer_name, -1)
            if e < PATCH_FROM_ENTRY:
                return output
            h = output[0] if isinstance(output, tuple) else output
            h[:, :-1, :] = base_full[:, :-1, e - PATCH_FROM_ENTRY, :]   # pin all but output -> base
            for s_role in S:
                h[rows, pos6[:, s_role], :] = tgt_slice[:, s_role, e, :]  # patched roles -> target
            return output
        return hook

    def recovery(m):
        return (m - baseline_logit_diff) / ((ceiling_logit_diff - baseline_logit_diff) or 1e-9)

    results = {}    # (name, mode) -> mean logit_diff
    for name, S in PATCH_SETS:
        for mode in MODES:
            lds = []
            for ci, ch in enumerate(base_chunks):
                n, s = ch["n"], ch["s"]
                rows = torch.arange(n, device=device)
                tsl = tgt_act[s:s + n]
                if mode == "open":
                    hook = make_open_hook(tsl, ch["pos6"], S, rows)
                else:
                    hook = make_isolated_hook(base_full_chunks[ci], tsl, ch["pos6"], S, rows)
                with TraceDict(model, layers=layer_names, edit_output=hook, retain_output=False) as td:
                    out_obj = model(input_ids=ch["input_ids"], attention_mask=ch["attention_mask"])
                    lds.append(logit_diff_from(out_obj, ch))
            m = float(torch.cat(lds).float().mean())
            results[(name, mode)] = m
            print(f"  {name:>16} · {mode:<8} {[TOKEN_NAMES[i] for i in S]}: "
                  f"logit_diff {m:+.4f}  recovery {recovery(m)*100:5.1f}%")

    assert abs(results[("both_labels", "isolated")] - baseline_logit_diff) > 1e-3, "isolated hook had no effect"

    # ---- save ----
    out_dir = Path(args.output_root) / regime
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    tag = args.task_pair
    summary = {
        "task_pair": tag, "src_task": src_task, "tgt_task": tgt_task, "n_pairs": N,
        "patch_from_entry": PATCH_FROM_ENTRY, "token_names": TOKEN_NAMES,
        "baseline_logit_diff": baseline_logit_diff, "ceiling_logit_diff": ceiling_logit_diff,
        "conditions": {
            name: {mode: {"positions": [TOKEN_NAMES[i] for i in S],
                          "mean_logit_diff": results[(name, mode)],
                          "recovery": recovery(results[(name, mode)])}
                   for mode in MODES}
            for name, S in PATCH_SETS},
        "metric": "logit(tgt_gold)-logit(src_gold) at query-final; all pairs. "
                  "isolated = patched roles->target, all other non-output tokens pinned to base.",
    }
    with open(out_dir / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"{tag}_logitdiff.csv", "w") as f:
        f.write("condition,mean_logit_diff,recovery\n")
        f.write(f"baseline,{baseline_logit_diff:.4f},\n")
        for name, S in PATCH_SETS:
            for mode in MODES:
                m = results[(name, mode)]
                f.write(f"{name}.{mode},{m:.4f},{recovery(m):.4f}\n")
        f.write(f"target_ceiling,{ceiling_logit_diff:.4f},\n")

    # ---- inline grouped bar figure (full version in plot_patch_labelset_follow.py) ----
    order = [("baseline", None)] + [(n, mode) for n, _ in PATCH_SETS for mode in MODES] + [("target_ceiling", None)]
    labels = ["baseline"] + [f"{n}\n{mode}" for n, _ in PATCH_SETS for mode in MODES] + ["target\nceiling"]
    vals = ([baseline_logit_diff]
            + [results[(n, mode)] for n, _ in PATCH_SETS for mode in MODES]
            + [ceiling_logit_diff])
    cmap = {"baseline": "#888888", "target_ceiling": "#55a868", "open": "#c7a0a2", "isolated": "#c44e52"}
    colors = [cmap["baseline"]] + [cmap[mode] for _, _ in PATCH_SETS for mode in MODES] + [cmap["target_ceiling"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(baseline_logit_diff, color="#888888", ls="--", lw=1)
    ax.axhline(ceiling_logit_diff, color="#55a868", ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean logit(tgt_gold) − logit(src_gold) @ query-final")
    ax.set_title(f"Isolated label patch {src_task}→{tgt_task}  (n={N})\n"
                 f"do ONLY the labels drive the output? (isolated = all other tokens pinned to base)",
                 fontsize=10)
    for b, (nm, mode) in zip(bars, order):
        v = b.get_height()
        lab = f"{v:+.2f}" if mode is None else f"{v:+.2f}\n({recovery(v)*100:.0f}%)"
        ax.text(b.get_x() + b.get_width() / 2, v + (0.05 if v >= 0 else -0.05), lab,
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"{tag}_label_follow_bars.png", dpi=140)
    plt.close(fig)
    print(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()
