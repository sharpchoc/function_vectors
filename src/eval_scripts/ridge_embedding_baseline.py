#!/usr/bin/env python
"""Embedding-only baseline for the (token position x layer) ridge sweep.

Same protocol as token_layer_fv_regression.py (branch worktree-labeltoken-fv-ridge) with
one change: X is the RAW TOKEN EMBEDDING wte[token_id] at each position — no transformer
blocks at all. GPT-J has no absolute position embeddings (rotary only, applied inside
attention), so the wte lookup is exactly the model's pre-attention representation. This
isolates how much of the layer-0 R^2 is pure token identity: the "free" information a
lookup table earns before any computation happens.

Everything else matches the layer sweep: same 69 tasks x 150 clean 10-shot prompts, same
52 positions (label31 + input21), Y = per-prompt FV for training, ridge alpha from
logspace(-1, 8, 19) by 5-fold CV over train tasks, scored per task against the TASK FV
with the split-pool-mean denominator. prompt_positions / ridge_fit_eval / r2_vs_taskfv
are copied verbatim from token_layer_fv_regression.py.

Runs on CPU: the weight file is only touched via mmap for the wte tensor.

Output: artifacts/69_task_run/token_layer_regressions/embedding.json (label31 positions)
        artifacts/69_task_run/token_layer_regressions/embedding_input21.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

_BOOT = Path(__file__).resolve().parents[2]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transformers import AutoTokenizer  # noqa: E402
from src.utils.prompt_utils import get_token_meta_labels, word_pairs_to_prompt_data  # noqa: E402
from src.utils.paths import ARTIFACTS_ROOT, REPO_ROOT  # noqa: E402

FVS = ARTIFACTS_ROOT / "69_task_run" / "perprompt_fvs"
LAB_RE = re.compile(r"^demonstration_(\d+)_label_token$")
INP_RE = re.compile(r"^demonstration_(\d+)_token$")
N_DEMOS = 10
POS_LABEL = [f"d{n}_{r}" for n in range(1, N_DEMOS + 1)
             for r in ("pre", "first", "last")] + ["query_cue"]
POS_INPUT = [f"d{n}_inp_{r}" for n in range(1, N_DEMOS + 1)
             for r in ("first", "last")] + ["query_inp_last"]
POS_ALL = POS_LABEL + POS_INPUT
ALPHAS = list(np.logspace(-1, 8, 19))
KFOLDS, CV_SEED = 5, 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts_root", type=Path,
                   default=REPO_ROOT / "dataset_files" / "isolation_prompts_ext")
    p.add_argument("--out_root", type=Path,
                   default=ARTIFACTS_ROOT / "69_task_run" / "token_layer_regressions")
    p.add_argument("--split_path", type=Path,
                   default=REPO_ROOT / "task_splits" / "extended_steerable_69_prunedfail.json")
    p.add_argument("--model_dir", type=Path, required=True,
                   help="local GPT-J snapshot dir (tokenizer files + pytorch_model.bin)")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args()


# --- copied verbatim from token_layer_fv_regression.py (pos_set fixed to all52) ---
def prompt_positions(rec, tokenizer):
    """(prompt_string, [token indices for all 52 positions]); no bos anywhere."""
    wp = {"input": [str(d["input"]) for d in rec["demos"]],
          "output": [str(d["output"]) for d in rec["demos"]]}
    qo = rec["query"]["output"]
    qo = [str(x) for x in qo] if isinstance(qo, list) else str(qo)
    q = {"input": str(rec["query"]["input"]), "output": qo}
    pd_ = word_pairs_to_prompt_data(wp, query_target_pair=q, prepend_bos_token=False,
                                    shuffle_labels=False)
    token_labels, prompt_string = get_token_meta_labels(pd_, tokenizer, query=q["input"],
                                                        prepend_bos=False)
    ids = tokenizer(prompt_string).input_ids
    assert len(ids) == len(token_labels)
    first, last = {}, {}
    inp_first, inp_last, q_inp_last = {}, {}, None
    for i, _, lab in token_labels:
        i = int(i)
        m = LAB_RE.match(lab)
        if m:
            n = int(m.group(1))
            first[n] = min(first.get(n, i), i)
            last[n] = max(last.get(n, -1), i)
            continue
        m = INP_RE.match(lab)          # demonstration_<n>_token = the input word
        if m:
            n = int(m.group(1))
            inp_first[n] = min(inp_first.get(n, i), i)
            inp_last[n] = max(inp_last.get(n, -1), i)
        elif lab == "query_demonstration_token":
            q_inp_last = i if q_inp_last is None else max(q_inp_last, i)
    assert sorted(first) == list(range(1, N_DEMOS + 1)), f"demos found: {sorted(first)}"
    assert sorted(inp_first) == list(range(1, N_DEMOS + 1)), \
        f"input tokens found for demos {sorted(inp_first)}"
    assert q_inp_last is not None, "no query_demonstration_token"

    idxs = []
    for n in range(1, N_DEMOS + 1):
        assert first[n] - 1 >= 0
        idxs += [first[n] - 1, first[n], last[n]]
    idxs.append(len(ids) - 1)
    for n in range(1, N_DEMOS + 1):
        idxs += [inp_first[n], inp_last[n]]
    idxs.append(q_inp_last)
    assert len(idxs) == len(POS_ALL)
    return ids, idxs


def r2_vs_taskfv(pred, task_fv, pool_mean):
    """Per-task R^2 with the constant task-FV target and split-pool-mean denominator."""
    resid = ((pred - task_fv) ** 2).sum()
    tot = ((task_fv - pool_mean) ** 2).sum() * pred.shape[0]
    return float(1 - resid / tot)


def ridge_fit_eval(X, Y, tr_idx, te_slices, tr_slices, task_fv, dtype, device):
    """Returns (results dict, best_alpha, pinned)."""
    Xtr = X[tr_idx].to(device=device, dtype=dtype)
    Ytr = Y[tr_idx].to(device=device, dtype=dtype)
    rng = np.random.RandomState(CV_SEED)
    tasks = sorted(tr_slices)
    order = rng.permutation(len(tasks))
    folds = [sorted(tasks[i] for i in f) for f in np.array_split(order, KFOLDS)]
    cv = torch.zeros(len(ALPHAS), dtype=dtype, device=device)
    row_of = {t: tr_slices[t] for t in tasks}
    for fold in folds:
        m = torch.zeros(len(Xtr), dtype=torch.bool)
        for t in fold:
            s, e = row_of[t]
            m[s:e] = True
        m = m.to(device)
        xf, yf = Xtr[~m], Ytr[~m]
        xbar, ybar = xf.mean(0), yf.mean(0)
        xc = xf - xbar
        evals, evecs = torch.linalg.eigh(xc.T @ xc)
        c = evecs.T @ (xc.T @ (yf - ybar))
        a_val = (Xtr[m] - xbar) @ evecs
        for ai, al in enumerate(ALPHAS):
            cv[ai] += (((a_val / (evals + al)) @ c + ybar - Ytr[m]) ** 2).sum()
    bi = int(torch.argmin(cv))
    best_alpha = float(ALPHAS[bi])

    xbar, ybar = Xtr.mean(0), Ytr.mean(0)
    xc = Xtr - xbar
    evals, evecs = torch.linalg.eigh(xc.T @ xc)
    c = evecs.T @ (xc.T @ (Ytr - ybar))

    def predict(xe):
        return ((xe - xbar) @ evecs / (evals + best_alpha)) @ c + ybar

    out = {}
    for name, slices in (("heldout", te_slices), ("train", tr_slices)):
        fvs = torch.stack([task_fv[t] for t in sorted(slices)]).to(device=device, dtype=dtype)
        pool_mean = fvs.mean(0)
        r2s = []
        for t in sorted(slices):
            s, e = slices[t]
            xe = (X[s:e] if name == "heldout" else Xtr[s:e]).to(device=device, dtype=dtype)
            r2s.append(r2_vs_taskfv(predict(xe), task_fv[t].to(device=device, dtype=dtype),
                                    pool_mean))
        out[name] = float(np.mean(r2s))
        out[name + "_per_task"] = {t: round(v, 4) for t, v in zip(sorted(slices), r2s)}
    return out, best_alpha, bi in (0, len(ALPHAS) - 1)
# --- end copied block ---


def load_wte(model_dir):
    sd = torch.load(model_dir / "pytorch_model.bin", map_location="cpu",
                    weights_only=True, mmap=True)
    wte = sd["transformer.wte.weight"].float().clone()
    del sd
    assert wte.shape[1] == 4096, wte.shape
    return wte


def main():
    args = parse_args()
    device = args.device
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    split = json.load(open(args.split_path))
    train_tasks, test_tasks = sorted(split["train_tasks"]), sorted(split["heldout_tasks"])
    args.out_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.padding_side = "right"
    wte = load_wte(args.model_dir)
    print(f"wte {tuple(wte.shape)} | 52 positions | {len(train_tasks)}+{len(test_tasks)} tasks",
          flush=True)

    def capture(task):
        recs = json.load(open(args.prompts_root / task / "train_prompts.json"))
        assert len(recs) == 150
        tok_ids = torch.zeros(150, len(POS_ALL), dtype=torch.long)
        for j, r in enumerate(recs):
            ids, idxs = prompt_positions(r, tokenizer)
            tok_ids[j] = torch.tensor([ids[i] for i in idxs])
        return wte[tok_ids]            # (150, 52, 4096) — pure embedding lookup

    # preallocated fp16 (like the layer sweep's capture): the container is memory-capped,
    # and fp32 X plus a torch.cat copy is ~18 GB — over the cap
    n_tasks = len(train_tasks) + len(test_tasks)
    X = torch.zeros(150 * n_tasks, len(POS_ALL), 4096, dtype=torch.float16)
    Yall, tr_slices, te_slices, task_fv = [], {}, {}, {}
    pos = 0
    for group, tasks in (("train", train_tasks), ("heldout", test_tasks)):
        for t in tasks:
            X[pos:pos + 150] = capture(t).half()
            f = torch.load(FVS / f"{t}.pt", map_location="cpu", weights_only=False)["fv"].float()
            Yall.append(f)
            (tr_slices if group == "train" else te_slices)[t] = (pos, pos + 150)
            pos += 150
            task_fv[t] = f.mean(0)
    n_train_rows = 150 * len(train_tasks)
    Y = torch.cat(Yall)
    tr_idx = torch.arange(n_train_rows)
    print(f"captured X {tuple(X.shape)}", flush=True)

    results = {}
    for pi, pname in enumerate(POS_ALL):
        res, alpha, pinned = ridge_fit_eval(X[:, pi].float(), Y, tr_idx, te_slices,
                                            tr_slices, task_fv, dtype, device)
        results[pname] = {"position": pname, "best_alpha": alpha, "alpha_pinned": pinned,
                          "r2_heldout_mean": round(res["heldout"], 4),
                          "r2_train_mean": round(res["train"], 4),
                          "r2_heldout_per_task": res["heldout_per_task"]}
        print(f"{pname:16s} heldout {res['heldout']:+.4f} train {res['train']:+.4f} "
              f"alpha {alpha:.3g}{' PINNED' if pinned else ''}", flush=True)

    common = {"layer": "embedding", "dtype": args.dtype,
              "scoring": "target = task FV; per-task R^2 vs split-pool mean; uniform over "
                         "dims; mean across tasks",
              "note": "X = wte[token_id] only — no transformer blocks (GPT-J has no "
                      "absolute position embeddings, so this is the pre-attention input)"}
    for fname, names in (("embedding.json", POS_LABEL), ("embedding_input21.json", POS_INPUT)):
        json.dump({**common, "positions": names,
                   "results": {n: results[n] for n in names}},
                  open(args.out_root / fname, "w"), indent=1)
        print(f"wrote {args.out_root / fname}", flush=True)


if __name__ == "__main__":
    main()
