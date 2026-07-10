#!/usr/bin/env python
"""Stream W: 1-shot preimage/FV projection-ablation causal test (GPT-J-6B; baukit).

For each of the 7 ridge held-out tasks, build 170 one-shot prompts (capture-pipeline-identical:
seed-42 split, demo sampled from the train split excluding the query, Q:/A: template,
'<|endoftext|>' BOS string prepend). At one site token per condition — cue1 (pre-label of the
demo), target1 (last label token of the demo), or final_cue (last prompt token) — remove the
residual-stream component along a unit direction u_b at start layer L and EVERY downstream block
b >= L (that token only):

    h[:, pos] -= (h[:, pos] . u_b) u_b        (fp32 math, cast back to model dtype)

Metric: delta log p (ablated - clean) of the FIRST answer token, read at the final position via
model.transformer(...) -> lm_head on the last column -> fp32 log_softmax.

Arms (direction sources; *_cf = same cells but a random OTHER task's FV, one draw per task):
  preimage_matched     cue1 -> tsvd bank pre_label_token_icl1, target1 -> last_label_token_icl1,
                       final_cue -> BOTH pre_label_token_icl2 (row final_cue_ctx, context-matched)
                       AND last_prompt_token_icl10 (row final_cue_icl10)
  preimage_icl10       pre_label_token_icl10 / last_label_token_icl10 / last_prompt_token_icl10
  fv                   the task FV direction itself, same unit vector at all 28 edit layers
  preimage_matched_cf, preimage_icl10_cf, fv_cf

Direction banks are the rank-16 TSVD preimages (fit_tsvd_preimages_multicell.py --tasks ...) of
the Stream S ridge maps; bank key edit_layer b <-> hook on transformer.h.{b} output <-> capture
entry b+1. Start layers L sweep edit layers 0..27; the embedding entry is never ablated.

Resumable: one npz per (task, arm); existing files are skipped unless --overwrite. Summaries
(per-task summary.csv, top-level combined_summary.csv over the npz present) rebuilt every run.

Sanity built in (per task): batched-vs-unbatched clean log p cross-check; a hooked forward with
L=28 (gate never fires) must reproduce the clean log p; decoded site tokens spot-checked.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.extract_targeted_residual_stream_activations import (
    make_prompt,
    sample_demo_indices,
    sample_query_indices,
    selected_token_records,
    stable_rng,
)
from src.eval_scripts.regress_activation_to_fv_fulldim_ridge import (
    DEFAULT_TEST_TASKS_EXCLUDE_CC_PC,
    load_function_vector,
    torch_load_trusted,
    write_json,
)
from utils.eval_utils import get_answer_id
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.paths import ARTIFACTS_ROOT, FV_FORMATION_DIR
from utils.prompt_utils import get_token_meta_labels, load_dataset

SITE_ROLES = ["pre_label_token", "last_label_token", "last_prompt_token"]
SITE_INDEX = {role: i for i, role in enumerate(SITE_ROLES)}
# row name -> which site token it ablates
ROW_SITE = {"cue1": "pre_label_token", "target1": "last_label_token",
            "final_cue": "last_prompt_token", "final_cue_ctx": "last_prompt_token",
            "final_cue_icl10": "last_prompt_token"}
# base arm -> [(row name, tsvd cell or None=raw FV direction)]
ARM_ROWS = {
    "preimage_matched": [("cue1", "pre_label_token_icl1"), ("target1", "last_label_token_icl1"),
                         ("final_cue_ctx", "pre_label_token_icl2"),
                         ("final_cue_icl10", "last_prompt_token_icl10")],
    "preimage_icl10": [("cue1", "pre_label_token_icl10"), ("target1", "last_label_token_icl10"),
                       ("final_cue", "last_prompt_token_icl10")],
    "fv": [("cue1", None), ("target1", None), ("final_cue", None)],
}
ALL_ARMS = ["preimage_matched", "preimage_icl10", "fv",
            "preimage_matched_cf", "preimage_icl10_cf", "fv_cf"]
N_EDIT_LAYERS = 28


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC))
    p.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS)
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_varicl_max4_top40")
    p.add_argument("--tsvd_root", type=Path,
                   default=ARTIFACTS_ROOT / "preimage_pairdiff_tsvdk16/train_varicl_max4_top40")
    p.add_argument("--output_root", type=Path, default=None,
                   help="Default: FV_FORMATION_DIR/oneshot_preimage_ablation/<fv_root basename>.")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--n_shots", type=int, default=1)
    p.add_argument("--max_train_prompts", type=int, default=130)
    p.add_argument("--max_test_prompts", type=int, default=40)
    p.add_argument("--start_layers", nargs="+", type=int, default=list(range(N_EDIT_LAYERS)))
    p.add_argument("--batch_size", type=int, default=170)
    p.add_argument("--max_prompts", type=int, default=None, help="Smoke cap on prompts/task.")
    p.add_argument("--prefixes", type=json.loads,
                   default={"input": "Q:", "output": "A:", "instructions": ""})
    p.add_argument("--separators", type=json.loads,
                   default={"input": "\n", "output": "\n\n", "instructions": ""})
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:
        return None


def build_cf_map(tasks, seed):
    cf = {}
    for task in tasks:
        others = sorted(t for t in tasks if t != task)
        rng = stable_rng("cf_assignment", seed, task)
        cf[task] = others[int(rng.integers(len(others)))]
    return cf


def build_prompts(task, dataset, tokenizer, model_config, args):
    """170 one-shot prompts mirroring the capture pipeline's query/demo sampling."""
    sampler = argparse.Namespace(seed=args.seed, n_shots=args.n_shots,
                                 max_train_prompts=args.max_train_prompts,
                                 max_test_prompts=args.max_test_prompts, max_valid_prompts=None)
    prompts = []
    for split in ("train", "test"):
        for query_idx in sample_query_indices(task, split, len(dataset[split]), sampler):
            demo_indices = sample_demo_indices(task, split, int(query_idx), dataset, sampler)
            prompt_data = make_prompt(dataset, split, int(query_idx), demo_indices,
                                      model_config, args.prefixes, args.separators)
            query = prompt_data["query_target"]["input"]
            token_labels, prompt_string = get_token_meta_labels(
                prompt_data, tokenizer, query=query, prepend_bos=model_config["prepend_bos"])
            recs = selected_token_records(token_labels, target_icl_example_index=1,
                                          token_roles=SITE_ROLES)
            pos = {r["token_role"]: r["token_position"] for r in recs}
            texts = {r["token_role"]: r["token_text"] for r in recs}
            ids = tokenizer(prompt_string, truncation=False, padding=False).input_ids
            assert pos["last_prompt_token"] == len(ids) - 1, \
                f"{task}: final site {pos['last_prompt_token']} != last index {len(ids) - 1}"
            answer = prompt_data["query_target"]["output"]       # already ' ' + str(v)
            answer_ids = get_answer_id(prompt_string, answer, tokenizer)
            prompts.append({"ids": ids, "pos": pos, "site_texts": texts, "split": split,
                            "query": query.strip(), "answer": answer.strip(),
                            "answer_id": int(answer_ids[0])})
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    return prompts


def make_chunks(prompts, tokenizer, batch_size, device):
    """Left-padded chunks with per-row site positions (offset by the pad amount)."""
    pad_id = tokenizer.pad_token_id
    chunks = []
    for s in range(0, len(prompts), batch_size):
        sub = prompts[s:s + batch_size]
        mx = max(len(x["ids"]) for x in sub)
        inp = torch.full((len(sub), mx), pad_id, dtype=torch.long)
        att = torch.zeros((len(sub), mx), dtype=torch.long)
        pos = torch.zeros((len(sub), len(SITE_ROLES)), dtype=torch.long)
        ans = torch.zeros(len(sub), dtype=torch.long)
        for r, pr in enumerate(sub):
            pad = mx - len(pr["ids"])
            inp[r, pad:] = torch.tensor(pr["ids"], dtype=torch.long)
            att[r, pad:] = 1
            for role, si in SITE_INDEX.items():
                pos[r, si] = pad + pr["pos"][role]
            ans[r] = pr["answer_id"]
        chunks.append({"input_ids": inp.to(device), "attention_mask": att.to(device),
                       "pos": pos.to(device), "answer_ids": ans.to(device), "n": len(sub)})
    return chunks


def load_directions(arm_cell, direction_task, args, device):
    """[28, 4096] fp32 unit rows (one per edit layer) + the pre-normalization norms."""
    if arm_cell is None:
        fv = load_function_vector(args.fv_root, direction_task).float()
        raw = fv.unsqueeze(0).expand(N_EDIT_LAYERS, -1).contiguous()
    else:
        bank_path = args.tsvd_root / arm_cell / "preimages" / f"{direction_task}_tsvd_preimage_bank.pt"
        bank = torch_load_trusted(bank_path, map_location="cpu")["preimages_by_edit_layer"]
        missing = [b for b in range(N_EDIT_LAYERS) if b not in bank]
        assert not missing, f"{bank_path} missing edit layers {missing}"
        raw = torch.stack([bank[b]["tsvd"].float() for b in range(N_EDIT_LAYERS)])
    norms = raw.norm(dim=1)
    unit = raw / norms.clamp_min(1e-12).unsqueeze(1)
    return unit.to(device), norms.numpy()


def main():
    args = parse_args()
    from baukit import TraceDict   # import late; precedent steer_label_cos_heatmap.py

    out_root = args.output_root or (FV_FORMATION_DIR / "oneshot_preimage_ablation" / args.fv_root.name)
    out_root.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    torch.set_grad_enabled(False)
    print("Loading model...", flush=True)
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(
        args.model_name, device=args.device, revision=args.revision)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    device = args.device
    layer_names = list(model_config["layer_hook_names"])     # transformer.h.0..27
    assert len(layer_names) == N_EDIT_LAYERS
    name_to_block = {nm: int(nm.split(".")[2]) for nm in layer_names}

    cf_map = build_cf_map(args.tasks, args.seed)
    write_json(out_root / "run_config.json", {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "cf_map": cf_map, "arm_rows": {a: ARM_ROWS[a] for a in ARM_ROWS},
        "git_commit": git_commit_hash(),
        "metric": "delta log p (ablated - clean) of the first answer token at the final position",
        "edit_layer_mapping": "start layer L ablates transformer.h.{b} outputs for all b >= L "
                              "(edit layers 0..27; embedding entry never touched)",
    })

    def forward_hidden(ch):
        return model.transformer(input_ids=ch["input_ids"],
                                 attention_mask=ch["attention_mask"]).last_hidden_state

    def run_logp(chunks, hook=None):
        """log p of each prompt's first answer token at the final position."""
        outs = []
        for ch in chunks:
            if hook is not None:
                with TraceDict(model, layers=layer_names, edit_output=hook):
                    hidden = forward_hidden(ch)
            else:
                hidden = forward_hidden(ch)
            logits = model.lm_head(hidden[:, -1, :]).float()
            logp = logits.log_softmax(dim=-1)
            rows = torch.arange(ch["n"], device=logits.device)
            outs.append(logp[rows, ch["answer_ids"]])
        return torch.cat(outs)

    for task in args.tasks:
        task_dir = out_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        todo = [a for a in args.arms
                if args.overwrite or not (task_dir / f"{a}_delta_logp.npz").exists()]
        if not todo:
            print(f"[{task}] all arms exist; skipping.", flush=True)
            continue

        t0 = time.time()
        dataset = load_dataset(task, root_data_dir=args.root_data_dir,
                               test_size=args.test_split, seed=args.seed)
        prompts = build_prompts(task, dataset, tokenizer, model_config, args)
        chunks = make_chunks(prompts, tokenizer, args.batch_size, device)
        n = len(prompts)

        # --- sanity: decoded site tokens (":" for the two pre-label-style sites) ---
        st = prompts[0]["site_texts"]
        for role in ("pre_label_token", "last_prompt_token"):
            assert st[role].strip() == ":", f"{task}: {role} decodes to {st[role]!r}, expected ':'"
        print(f"[{task}] n={n} | sites cue1={st['pre_label_token']!r} "
              f"target1={st['last_label_token']!r} final={st['last_prompt_token']!r}", flush=True)

        # --- clean pass + batched-vs-unbatched + no-op-gate checks ---
        clean_logp = run_logp(chunks)
        assert torch.isfinite(clean_logp).all(), f"{task}: non-finite clean log p"
        for i in range(min(3, n)):
            single = make_chunks([prompts[i]], tokenizer, 1, device)
            lp = run_logp(single)[0]
            diff = (lp - clean_logp[i]).abs().item()
            assert diff < 0.05, f"{task}: batched vs single log p differs by {diff:.4f} at {i}"

        def noop_hook(output, layer_name):
            if name_to_block[layer_name] < N_EDIT_LAYERS:   # gate at L=28: never fires
                return output
            return output

        noop_logp = run_logp(chunks, hook=noop_hook)
        assert torch.allclose(noop_logp, clean_logp, atol=1e-5), f"{task}: no-op gate != clean"

        # --- arm sweeps ---
        for arm in todo:
            base = arm[:-3] if arm.endswith("_cf") else arm
            direction_task = cf_map[task] if arm.endswith("_cf") else task
            rows = ARM_ROWS[base]
            delta = np.full((len(rows), N_EDIT_LAYERS, n), np.nan, dtype=np.float32)
            dir_norms = np.full((len(rows), N_EDIT_LAYERS), np.nan, dtype=np.float32)
            ta = time.time()
            for ri, (row, cell) in enumerate(rows):
                U, norms = load_directions(cell, direction_task, args, device)
                dir_norms[ri] = norms
                site = SITE_INDEX[ROW_SITE[row]]
                for L in args.start_layers:
                    s = 0
                    for ch in chunks:
                        pos_vec = ch["pos"][:, site]
                        rows_idx = torch.arange(ch["n"], device=device)

                        def hook(output, layer_name):
                            b = name_to_block[layer_name]
                            if b < L:
                                return output
                            h = output[0] if isinstance(output, tuple) else output
                            u = U[b]
                            v = h[rows_idx, pos_vec, :].float()
                            v = v - torch.outer(v @ u, u)
                            h[rows_idx, pos_vec, :] = v.to(h.dtype)
                            return output

                        lp = run_logp([ch], hook=hook)
                        delta[ri, L, s:s + ch["n"]] = (lp - clean_logp[s:s + ch["n"]]).cpu().numpy()
                        s += ch["n"]
            assert np.isfinite(delta[:, args.start_layers, :]).all(), f"{task}/{arm}: non-finite delta"
            np.savez(task_dir / f"{arm}_delta_logp.npz",
                     delta_logp=delta, clean_logp=clean_logp.cpu().numpy(),
                     row_names=np.array([r for r, _ in rows]),
                     row_cells=np.array([str(c) for _, c in rows]),
                     start_layers=np.array(args.start_layers),
                     split=np.array([p["split"] for p in prompts]),
                     query=np.array([p["query"] for p in prompts]),
                     answer=np.array([p["answer"] for p in prompts]),
                     answer_token_id=np.array([p["answer_id"] for p in prompts]),
                     direction_norms=dir_norms,
                     cf_task=np.array(cf_map[task] if arm.endswith("_cf") else ""))
            print(f"[{task}] {arm}: {len(rows)} rows x {len(args.start_layers)} layers "
                  f"in {time.time() - ta:.0f}s", flush=True)
        print(f"[{task}] done in {time.time() - t0:.0f}s", flush=True)

    # --- summaries over whatever npz exist ---
    combined = []
    for task in args.tasks:
        task_dir = out_root / task
        rows_out = []
        for arm in ALL_ARMS:
            f = task_dir / f"{arm}_delta_logp.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=False)
            delta, split = z["delta_logp"], z["split"]
            test_mask = split == "test"
            for ri, row in enumerate(z["row_names"]):
                for L in z["start_layers"]:
                    d = delta[ri, L]
                    d = d[np.isfinite(d)]
                    dt = delta[ri, L][test_mask]
                    dt = dt[np.isfinite(dt)]
                    if d.size == 0:
                        continue
                    rows_out.append({
                        "task": task, "arm": arm, "row": str(row), "start_layer": int(L),
                        "mean_delta_all": float(d.mean()),
                        "sem_all": float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else np.nan,
                        "n_all": int(d.size),
                        "mean_delta_test40": float(dt.mean()) if dt.size else np.nan,
                        "sem_test40": float(dt.std(ddof=1) / np.sqrt(dt.size)) if dt.size > 1 else np.nan,
                        "n_test40": int(dt.size)})
        if rows_out:
            with open(task_dir / "summary.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
                w.writeheader()
                w.writerows(rows_out)
            combined.extend(rows_out)
    if combined:
        with open(out_root / "combined_summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(combined[0].keys()))
            w.writeheader()
            w.writerows(combined)
        print(f"summaries written under {out_root}")
    print("DONE")


if __name__ == "__main__":
    main()
