#!/usr/bin/env python
"""Pre-flight gates for the Qwen2.5 port of the 69_task_run read/write-feature studies.

1. Assets: split, prompts, label_resid_means, perprompt_fvs, means.pt + selection for all tasks.
2. Prompt-format byte-equality across the three prompt builders used by the studies
   (f-string of ablate_fv_cue6 / sixshot_dummy_steer, get_token_meta_labels string of the
   read-ablation scripts, create_prompt(record_to_prompt_data) of the presence capture) for
   n in {0, 1, 6, 10}; label-token positions >= n.
3. Dummy-slot gate: the 6-shot and 1-shot '_' scaffolds find exactly n slots, each a token
   containing '_' (Qwen fuses " _\n\n" into one token).
4. FV consistency (needs the model, --with_model): cos(unit FV from means.pt + selection,
   unit mean of the per-prompt FVs) > 0.99 for every task, and the W_O linearity gate
   sum_h W_O^{l,h} a_h == o_proj output on one batch.
Prints a summary; exits non-zero on any failed gate.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import (QWEN25_MODEL, QWEN25_PROMPTS, QWEN25_READ_ARTIFACTS_DIR,  # noqa: E402
                             QWEN25_SELECTION_ROOT, QWEN25_SPLIT)
from src.utils.prompt_utils import create_prompt, get_token_meta_labels, word_pairs_to_prompt_data  # noqa: E402
from src.sandbox.ext_steerability.ablate_pc50_labeltokens import LABEL_RE  # noqa: E402
from src.sandbox.ext_steerability.ablate_fv_cue6 import build_items_nshot  # noqa: E402
from src.sandbox.ext_steerability.sixshot_dummy_steer import build_items_6shot  # noqa: E402
from src.sandbox.ext_steerability.steer_read_dir_methods import build_items as build_items_1shot  # noqa: E402
from src.sandbox.isolation_upper_bound.run_task import record_to_prompt_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with_model", action="store_true")
    ap.add_argument("--max_tasks", type=int, default=None)
    args = ap.parse_args()
    fails = []
    split = json.load(open(QWEN25_SPLIT))
    tasks = sorted(split["train_tasks"] + split["heldout_tasks"])
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    sel = json.load(open(QWEN25_SELECTION_ROOT / "pooled_sparse" / "selection.json"))
    sel_flat = torch.tensor(sel["selected_flat"])
    print(f"{len(tasks)} tasks; selection {len(sel_flat)} heads, inject_layer {sel['inject_layer']}")

    # 1. assets
    for t in tasks:
        for f in (QWEN25_PROMPTS / t / "train_prompts.json",
                  QWEN25_READ_ARTIFACTS_DIR / "label_resid_means" / f"{t}.pt",
                  QWEN25_READ_ARTIFACTS_DIR / "perprompt_fvs" / f"{t}.pt",
                  QWEN25_SELECTION_ROOT / t / "means.pt"):
            if not f.exists():
                fails.append(f"missing asset {f}")
    print(f"[1] assets: {len(fails)} missing")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN25_MODEL)
    tok.pad_token = tok.eos_token
    cfg = {"prepend_bos": False, "name_or_path": QWEN25_MODEL, "model_name": QWEN25_MODEL}

    # 2. prompt formats
    n_cmp, n_rec = 0, 0
    for t in tasks:
        recs = json.load(open(QWEN25_PROMPTS / t / "train_prompts.json"))
        n_rec += len(recs)
        for n in (0, 1, 6, 10):
            fitems = build_items_nshot(t, QWEN25_PROMPTS, tok, n)
            for rec, it in zip(recs, fitems):
                demos = rec["demos"][:n]
                q = {"input": str(rec["query"]["input"]),
                     "output": [str(x) for x in rec["query"]["output"]]
                     if isinstance(rec["query"]["output"], list) else str(rec["query"]["output"])}
                fstr = "".join(f"Q: {str(d['input'])}\nA: {str(d['output']).strip()}\n\n"
                               for d in demos) + f"Q: {q['input']}\nA:"
                assert tok(fstr).input_ids == it["ids"]
                # (b) read-ablation builder (n >= 1 only: word_pairs_to_prompt_data needs demos)
                if n >= 1:
                    wp = {"input": [str(d["input"]) for d in demos],
                          "output": [str(d["output"]) for d in demos]}
                    pd_ = word_pairs_to_prompt_data(wp, query_target_pair=q, prepend_bos_token=False,
                                                    shuffle_labels=False)
                    labels, pstr = get_token_meta_labels(pd_, tok, prepend_bos=False)
                    if pstr != fstr:
                        fails.append(f"[2b] {t} n={n} idx={rec['prompt_index']}: meta-label string != f-string")
                    pos = [i for i, _, lab in labels if LABEL_RE.match(lab)]
                    if len(pos) < n or len(tok(pstr).input_ids) != len(labels):
                        fails.append(f"[2b] {t} n={n} idx={rec['prompt_index']}: {len(pos)} label tokens / len mismatch")
                # (c) presence-capture builder
                r_n = dict(rec); r_n["demos"] = demos
                cstr = create_prompt(record_to_prompt_data(r_n, cfg))
                if cstr != fstr:
                    fails.append(f"[2c] {t} n={n} idx={rec['prompt_index']}: create_prompt != f-string\n  {cstr!r}\n  {fstr!r}")
                n_cmp += 1
    print(f"[2] prompt formats: {n_cmp} comparisons over {n_rec} records; fails so far {len(fails)}")

    # 3. dummy slots
    for t in tasks:
        try:
            it6 = build_items_6shot(t, QWEN25_PROMPTS, tok, real_labels=False)
            it1 = build_items_1shot(t, QWEN25_PROMPTS, tok)
        except AssertionError as e:
            fails.append(f"[3] {t}: {e}"); continue
        for it in it6:
            toks = tok.convert_ids_to_tokens([it["ids"][i] for i in it["inj_idx_list"]])
            if len(it["inj_idx_list"]) != 6 or not all("_" in s for s in toks):
                fails.append(f"[3] {t}: bad 6-shot slots {toks}")
        for it in it1:
            s = tok.convert_ids_to_tokens([it["ids"][it["inj_idx"]]])[0]
            if "_" not in s:
                fails.append(f"[3] {t}: bad 1-shot slot {s}")
    print(f"[3] dummy slots checked; fails so far {len(fails)}")

    # 4. FV consistency (model)
    if args.with_model:
        from src.sandbox.ext_steerability.ablate_pc50_labeltokens import load_model, model_dims
        from src.sandbox.ext_steerability.ablate_fv_cue6 import unit_fv
        from src.utils.model_utils import get_attn_out_proj
        model, _ = load_model(None, QWEN25_MODEL)
        n_layers, d, n_heads = model_dims(model)
        print(f"model dims: {n_layers} layers, d={d}, {n_heads} heads")

        class A: pass
        a = A(); a.means_root = QWEN25_SELECTION_ROOT
        coss, norms = [], []
        for t in tasks:
            u, nrm = unit_fv(t, a, model, sel_flat)
            pp = torch.load(QWEN25_READ_ARTIFACTS_DIR / "perprompt_fvs" / f"{t}.pt",
                            map_location="cpu", weights_only=False)["fv"].double().mean(0)
            c = float(u.double().cpu() @ (pp / pp.norm()))
            coss.append(c); norms.append(nrm)
            if c < 0.99:
                fails.append(f"[4] {t}: cos(unit_fv, mean perprompt fv) = {c:.4f}")
        coss = torch.tensor(coss); norms = torch.tensor(norms)
        print(f"[4] FV consistency: cos min {coss.min():.4f} median {coss.median():.4f}; "
              f"||v_A|| median {norms.median():.1f} min {norms.min():.1f} max {norms.max():.1f}")
        # linearity gate on one batch: o_proj(input) == sum_h W_O[:, h-slice] @ a_h
        it = build_items_nshot(tasks[0], QWEN25_PROMPTS, tok, 6)[:4]
        L = max(len(x["ids"]) for x in it)
        ids = torch.full((len(it), L), tok.eos_token_id); att = torch.zeros(len(it), L, dtype=torch.long)
        for r, x in enumerate(it):
            ids[r, :len(x["ids"])] = torch.tensor(x["ids"]); att[r, :len(x["ids"])] = 1
        got = {}
        proj = get_attn_out_proj(model, 12)
        h = proj.register_forward_hook(lambda m, i, o: got.update(inp=i[0].detach(), out=o.detach()))
        with torch.no_grad():
            model(input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False)
        h.remove()
        W = proj.weight.float().view(d, n_heads, d // n_heads)
        a_h = got["inp"].float().view(*got["inp"].shape[:-1], n_heads, d // n_heads)
        recon = torch.einsum("ohd,bthd->bto", W, a_h)
        rel = float((recon - got["out"].float()).norm() / got["out"].float().norm())
        print(f"[4] linearity: rel dev {rel:.2e}" + (" OK" if rel < 5e-2 else " FAIL"))
        if rel >= 5e-2:
            fails.append(f"[4] linearity rel dev {rel}")

    print(f"\n{'ALL GATES PASSED' if not fails else f'{len(fails)} FAILURES'}")
    for f in fails[:40]:
        print(" ", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
