"""
Attention KNOCKOUT: does the query-final token (`qfin`, last `A:`) read task information DIRECTLY from
the demo-2 pre-label token (the `A:` before L2), or from the label tokens? (GPT-J-6B; transformers 4.49,
eager attention.)

We knock out a single attention edge — `qfin`'s query attending to a chosen key position — at EVERY
layer and head, by setting the pre-softmax attention score for that (query, key) entry to -inf, so after
softmax the weight on that key is 0 and the row renormalizes to sum 1 over the remaining keys. Only the
`qfin` query row is edited; all other tokens attend normally.

Conditions (key cut from qfin's attention; the output token itself is never a key here):
    clean              -- no knockout
    ko_demo2_prelabel  -- cut {demo2 pre label}            (TEST)
    ko_both_labels     -- cut {demo1 label, demo2 label}   (+control: the "reads from labels" alternative)
    ko_demo2_qcolon    -- cut {demo-2 "Q:" colon}          (-control: a structural token; should not hurt)

Metric (all prompts, no judge): on correct 2-shot prompts per task, first-token top-1 accuracy
(argmax == gold_1) and mean gold-token logit at qfin, clean vs each knockout (report the drop).
Tasks: antonym, synonym, next_number_digits, prev_number_digits (each on its own matched-label prompts).

Implementation: monkeypatch GPTJAttention._attn (faithful 4.49.0 copy + knockout); knockout indices are
stashed per-layer on `attn._ko` before each forward. No baukit needed (logits read from model output).
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
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers.models.gptj.modeling_gptj import GPTJAttention

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import word_pairs_to_prompt_data, create_prompt, get_token_meta_labels
from utils.paths import LABEL_GEOMETRY_DIR

TASKS = ["antonym", "synonym", "next_number_digits", "prev_number_digits"]
PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}
# roles in the 6-position table: 0 demo1 label, 1 demo2 input, 2 demo2 pre label, 3 demo2 label,
# 4 query input, 5 query pre label (= qfin)
CONDITIONS = ["clean", "ko_demo2_prelabel", "ko_both_labels", "ko_demo2_qcolon"]

# ----------------------- attention-knockout monkeypatch (transformers 4.49.0) -----------------------
def _attn_with_knockout(self, query, key, value, attention_mask=None, head_mask=None):
    # Faithful copy of GPTJAttention._attn (4.49.0) + a pre-softmax knockout block.
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    attn_weights = attn_weights / self.scale_attn
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    ko = getattr(self, "_ko", None)            # (q_idx [B], k_idx [B,K]) or None
    if ko is not None:
        q_idx, k_idx = ko
        B, H = attn_weights.shape[0], attn_weights.shape[1]
        rows = torch.arange(B, device=attn_weights.device)[:, None]      # [B,1]
        heads = torch.arange(H, device=attn_weights.device)[None, :]     # [1,H]
        mn = torch.finfo(attn_weights.dtype).min
        for j in range(k_idx.shape[1]):
            attn_weights[rows, heads, q_idx[:, None], k_idx[:, j][:, None]] = mn

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.to(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


def install_patch():
    GPTJAttention._attn = _attn_with_knockout


def set_knockout(model, q_idx, k_idx):
    for blk in model.transformer.h:
        blk.attn._ko = (q_idx, k_idx)


def clear_knockout(model):
    for blk in model.transformer.h:
        blk.attn._ko = None
# ----------------------------------------------------------------------------------------------------

# --- baukit-free position helpers (inlined, mirror patch_labelset_follow.py) ---
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
# -------------------------------------------------------------------------------


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
    p = argparse.ArgumentParser(description="qfin attention knockout: does qfin read task info from demo-2 pre-label?")
    p.add_argument("--max_pairs", type=int, default=None, help="Cap prompts per task (smoke tests).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_root", type=str,
                   default=str(LABEL_GEOMETRY_DIR / "twoshot" / "qfinal_attn_knockout"))
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    install_patch()

    print("Loading model...")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(args.model_name, device=args.device)
    model.eval()
    device = args.device
    assert model.config._attn_implementation == "eager", \
        f"need eager attention to edit scores, got {model.config._attn_implementation}"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id

    def single(w):
        return len(tokenizer(" " + w).input_ids) == 1

    def build_task(task):
        o2i, i2o = load_task(args.root_data_dir, task)
        labels = [w for w in sorted(o2i) if single(w)]
        qpool = [x for x in sorted(i2o) if single(x)]
        ids_list, pos_list, gold_list = [], [], []
        for w in labels:
            rng = stable_rng(args.seed, task, w)
            L1 = w
            cand_L2 = [x for x in labels if x != L1]
            if not cand_L2:
                continue
            L2 = str(rng.choice(cand_L2))
            si = [x for x in o2i[L1] if single(x)]
            sj = [x for x in o2i[L2] if single(x)]
            if not si or not sj:
                continue
            demo_in = [str(rng.choice(si)), str(rng.choice(sj))]
            forbidden = {L1, L2, *demo_in}
            cand_q = [q for q in qpool if q not in forbidden]
            if not cand_q:
                continue
            q = str(rng.choice(cand_q))
            pd = build_two_shot(demo_in, [L1, L2], q)
            token_labels, prompt_string = get_token_meta_labels(
                pd, tokenizer, query=q, prepend_bos=model_config["prepend_bos"])
            pos6 = get_six_positions(token_labels)
            ids = tokenizer(prompt_string).input_ids
            assert pos6[-1] == len(ids) - 1, "qfin not last token"
            ids_list.append(ids)
            pos_list.append(pos6)
            gold_list.append(tokenizer(" " + i2o[q]).input_ids[0])
            if args.max_pairs is not None and len(ids_list) >= args.max_pairs:
                break
        return ids_list, pos_list, gold_list

    def make_chunks(ids_list, pos_list, gold_list):
        chunks, s, n = [], 0, len(ids_list)
        while s < n:
            sub_ids = ids_list[s:s + args.batch_size]
            sub_pos = pos_list[s:s + args.batch_size]
            sub_gold = gold_list[s:s + args.batch_size]
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
            chunks.append({"input_ids": inp.to(device), "attention_mask": att.to(device),
                           "pos6": pos.to(device), "gold": torch.tensor(sub_gold, device=device), "n": B})
            s += B
        return chunks

    def k_for_condition(ch, cond):
        pos = ch["pos6"]
        if cond == "ko_demo2_prelabel":
            return pos[:, 2:3]
        if cond == "ko_both_labels":
            return pos[:, [0, 3]]
        if cond == "ko_demo2_qcolon":
            return (pos[:, 1] - 1).unsqueeze(1)
        raise ValueError(cond)

    def run_condition(chunks, cond):
        top1s, glogits = [], []
        for ch in chunks:
            if cond == "clean":
                clear_knockout(model)
            else:
                set_knockout(model, ch["pos6"][:, 5], k_for_condition(ch, cond))
            logits = model(input_ids=ch["input_ids"], attention_mask=ch["attention_mask"]).logits[:, -1, :]
            ar = torch.arange(ch["n"], device=device)
            top1s.append(logits.argmax(-1) == ch["gold"])
            glogits.append(logits[ar, ch["gold"]].float())
        clear_knockout(model)
        return (torch.cat(top1s).float().mean().item(), torch.cat(glogits).mean().item())

    # ---- knockout-correctness check (one small batch, output_attentions) ----
    def verify(chunks):
        ch = chunks[0]
        b = min(4, ch["n"])
        q = ch["pos6"][:b, 5]
        k = ch["pos6"][:b, 2:3]
        sub = {"input_ids": ch["input_ids"][:b], "attention_mask": ch["attention_mask"][:b]}
        set_knockout(model, q, k)
        out = model(**sub, output_attentions=True)
        clear_knockout(model)
        aw = out.attentions[12].float()                       # [b, H, seq, seq] at layer 12
        rows = torch.arange(b, device=device)
        cut = aw[rows, :, q, k[:, 0]]                          # [b, H] weight on the knocked-out key
        rowsum = aw[rows, :, q, :].sum(-1)                    # [b, H] should be ~1
        assert float(cut.abs().max()) < 1e-6, f"knocked key weight not 0: {float(cut.abs().max())}"
        assert float((rowsum - 1).abs().max()) < 1e-3, f"row not renormalized: {float((rowsum-1).abs().max())}"
        print(f"verify OK: cut weight max {float(cut.abs().max()):.2e}, row-sum dev {float((rowsum-1).abs().max()):.2e}")

    results = {}
    verified = False
    for task in TASKS:
        ids_list, pos_list, gold_list = build_task(task)
        chunks = make_chunks(ids_list, pos_list, gold_list)
        n = len(ids_list)
        if not verified:
            verify(chunks); verified = True
        results[task] = {"n": n, "conditions": {}}
        for cond in CONDITIONS:
            top1, glogit = run_condition(chunks, cond)
            results[task]["conditions"][cond] = {"top1": top1, "mean_gold_logit": glogit}
        c = results[task]["conditions"]
        base_t, base_g = c["clean"]["top1"], c["clean"]["mean_gold_logit"]
        for cond in CONDITIONS:
            c[cond]["d_top1"] = c[cond]["top1"] - base_t
            c[cond]["d_gold_logit"] = c[cond]["mean_gold_logit"] - base_g
        print(f"[{task}] n={n}  " + "  ".join(
            f"{cond}: top1={c[cond]['top1']:.3f}(Δ{c[cond]['d_top1']:+.3f}) "
            f"glogit={c[cond]['mean_gold_logit']:.2f}(Δ{c[cond]['d_gold_logit']:+.2f})" for cond in CONDITIONS))

    # ---- save ----
    out_dir = Path(args.output_root)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    summary = {"tasks": TASKS, "conditions": CONDITIONS, "knockout": "qfin query -> key edge, all layers/heads",
               "results": results}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "metrics.csv", "w") as f:
        f.write("task,condition,n,top1,d_top1,mean_gold_logit,d_gold_logit\n")
        for task in TASKS:
            for cond in CONDITIONS:
                c = results[task]["conditions"][cond]
                f.write(f"{task},{cond},{results[task]['n']},{c['top1']:.4f},{c['d_top1']:.4f},"
                        f"{c['mean_gold_logit']:.4f},{c['d_gold_logit']:.4f}\n")

    # ---- inline bar figure: top1 per task grouped by condition ----
    _plot(results, out_dir / "figures" / "qfinal_attn_knockout_bars.png")
    print(f"DONE -> {out_dir}")


COND_COLOR = {"clean": "#888888", "ko_demo2_prelabel": "#c44e52",
              "ko_both_labels": "#4c72b0", "ko_demo2_qcolon": "#55a868"}
COND_LABEL = {"clean": "clean", "ko_demo2_prelabel": "cut qfin→demo2 pre-label (test)",
              "ko_both_labels": "cut qfin→both labels (+ctrl)", "ko_demo2_qcolon": "cut qfin→demo2 'Q:' (−ctrl)"}


def _plot(results, out_png):
    tasks = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    width = 0.2
    for ax, metric, ylabel in [(axes[0], "top1", "first-token top-1 accuracy"),
                               (axes[1], "mean_gold_logit", "mean gold-token logit @ qfin")]:
        for ci, cond in enumerate(CONDITIONS):
            vals = [results[t]["conditions"][cond][metric] for t in tasks]
            ax.bar(np.arange(len(tasks)) + ci * width, vals, width,
                   color=COND_COLOR[cond], label=COND_LABEL[cond])
        ax.set_xticks(np.arange(len(tasks)) + 1.5 * width)
        ax.set_xticklabels(tasks, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Does qfin read task info directly from demo-2 pre-label? "
                 "(attention knockout at all layers; test vs ±controls)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
