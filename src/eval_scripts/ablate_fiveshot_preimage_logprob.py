#!/usr/bin/env python
"""Stream W-5shot: 5-shot preimage/FV projection-ablation causal test (GPT-J-6B; baukit).

Extension of the 1-shot Stream W study (ablate_oneshot_preimage_logprob.py — kept untouched;
this is a standalone sibling) to --n_shots 5 prompts, with combined multi-position rows.

For each of the 7 ridge held-out tasks, build 170 five-shot prompts (capture-pipeline-identical:
seed-42 split, demos sampled from the train split excluding the query, Q:/A: template,
'<|endoftext|>' BOS string prepend). At the row's site token(s) — cue_i (pre-label ':' of demo i),
target_i (last label token of demo i), final_cue (last prompt token) — remove the residual-stream
component along a unit direction u_b at start layer L and EVERY downstream block b >= L (those
tokens only):

    h[:, pos] -= (h[:, pos] . u_b) u_b        (fp32 math, cast back to model dtype)

Metric: delta log p (ablated - clean) of the FIRST answer token, read at the final position via
model.transformer(...) -> lm_head on the last column -> fp32 log_softmax.

Rows per arm (14 for n_shots=5): cue1..cue5, target1..target5, final_cue individually, plus
combined rows ablated simultaneously in one forward:
  all_targets           the 5 demo label tokens
  all_cues              the 5 demo ':' cue tokens
  all_cues_incl_final   the 5 demo cues + the final query cue

Arms (direction sources; *_cf = same cells but a random OTHER task's directions, one draw per
task — same cf assignment as the 1-shot study):
  preimage_matched   position-matched cells: cue_i <- pre_label_token_icl{i},
                     target_i <- last_label_token_icl{i}, final_cue <- pre_label_token_icl6
                     (the 5-shot query token IS a "pre label 6" by causal context).
                     Combined rows: each position keeps its own matched cell.
  preimage_icl10     cue_i <- pre_label_token_icl10, target_i <- last_label_token_icl10;
                     the INDIVIDUAL final_cue row keeps last_prompt_token_icl10 (1-shot
                     consistent); in all_cues_incl_final the final cue gets
                     pre_label_token_icl10 (same vector at every cue).
  fv                 the task FV direction itself, same unit vector at all 28 edit layers,
                     at every position of the row.

Direction banks are the rank-16 TSVD preimages (fit_tsvd_preimages_multicell.py) of the ridge
maps fit against the canonical train_varicl_top40 FVs; bank key edit_layer b <-> hook on
transformer.h.{b} output <-> capture entry b+1. Start layers L sweep edit layers 0..27; the
embedding entry is never ablated.

Memory: 5-shot prompts are ~110 tokens; defaults are --batch_size 85 and
TraceDict(retain_output=False) + use_cache=False so the sweep fits on 24 GB.

Resumable: one npz per (task, arm); existing files are skipped unless --overwrite. Summaries
(per-task summary.csv, top-level combined_summary.csv over ALL task dirs on disk) rebuilt every
run.

Sanity built in (per task): batched-vs-unbatched clean log p cross-check; a hooked forward with
L=28 (gate never fires) must reproduce the clean log p; every cue site (incl. final) must decode
to ':'; all 11 site positions must be strictly increasing in prompt order (guards the multi-site
hook against tokenization edge cases).
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

ALL_ARMS = ["preimage_matched", "preimage_icl10", "fv",
            "preimage_matched_cf", "preimage_icl10_cf", "fv_cf"]
N_EDIT_LAYERS = 28


def build_sites(n_shots):
    """Site names in prompt order: cue_i then target_i per demo, final_cue last."""
    sites = []
    for i in range(1, n_shots + 1):
        sites += [f"cue{i}", f"target{i}"]
    sites.append("final_cue")
    return sites


def build_arm_rows(n_shots):
    """base arm -> [(row name, [(site, tsvd cell or None=raw FV direction), ...])]."""
    S = n_shots
    m_cue = lambda i: (f"cue{i}", f"pre_label_token_icl{i}")
    m_tgt = lambda i: (f"target{i}", f"last_label_token_icl{i}")
    m_fin = ("final_cue", f"pre_label_token_icl{S + 1}")
    matched = (
        [(f"cue{i}", [m_cue(i)]) for i in range(1, S + 1)]
        + [(f"target{i}", [m_tgt(i)]) for i in range(1, S + 1)]
        + [("final_cue", [m_fin]),
           ("all_targets", [m_tgt(i) for i in range(1, S + 1)]),
           ("all_cues", [m_cue(i) for i in range(1, S + 1)]),
           ("all_cues_incl_final", [m_cue(i) for i in range(1, S + 1)] + [m_fin])]
    )
    i_cue = lambda i: (f"cue{i}", "pre_label_token_icl10")
    i_tgt = lambda i: (f"target{i}", "last_label_token_icl10")
    icl10 = (
        [(f"cue{i}", [i_cue(i)]) for i in range(1, S + 1)]
        + [(f"target{i}", [i_tgt(i)]) for i in range(1, S + 1)]
        + [("final_cue", [("final_cue", "last_prompt_token_icl10")]),
           ("all_targets", [i_tgt(i) for i in range(1, S + 1)]),
           ("all_cues", [i_cue(i) for i in range(1, S + 1)]),
           ("all_cues_incl_final", [i_cue(i) for i in range(1, S + 1)]
            + [("final_cue", "pre_label_token_icl10")])]
    )
    fv = [(name, [(site, None) for site, _ in spec]) for name, spec in matched]
    return {"preimage_matched": matched, "preimage_icl10": icl10, "fv": fv}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC))
    p.add_argument("--arms", nargs="+", default=ALL_ARMS, choices=ALL_ARMS)
    p.add_argument("--fv_root", type=Path,
                   default=ARTIFACTS_ROOT / "function_vectors/gpt-j/train_varicl_top40")
    p.add_argument("--tsvd_root", type=Path,
                   default=ARTIFACTS_ROOT / "preimage_pairdiff_tsvdk16/train_varicl_top40")
    p.add_argument("--output_root", type=Path, default=None,
                   help="Default: FV_FORMATION_DIR/fiveshot_preimage_ablation/<fv_root basename> "
                        "(required explicitly when --n_shots != 5).")
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--root_data_dir", type=str, default="dataset_files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_split", type=float, default=0.3)
    p.add_argument("--n_shots", type=int, default=5)
    p.add_argument("--max_train_prompts", type=int, default=130)
    p.add_argument("--max_test_prompts", type=int, default=40)
    p.add_argument("--start_layers", nargs="+", type=int, default=list(range(N_EDIT_LAYERS)))
    p.add_argument("--batch_size", type=int, default=85)
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


def build_cf_map(tasks, seed, pool=None):
    """Counterfactual task per task, drawn from the FIXED pool (default: the 7 held-out tasks)
    so the assignment is identical no matter how --tasks shards an invocation. Verbatim from
    the 1-shot study -> identical cf assignment."""
    pool = list(pool) if pool is not None else list(DEFAULT_TEST_TASKS_EXCLUDE_CC_PC)
    cf = {}
    for task in tasks:
        others = sorted(t for t in pool if t != task)
        assert others, f"cf pool has no alternative for {task}"
        rng = stable_rng("cf_assignment", seed, task)
        cf[task] = others[int(rng.integers(len(others)))]
    return cf


def build_prompts(task, dataset, tokenizer, model_config, args, sites):
    """170 n-shot prompts mirroring the capture pipeline's query/demo sampling, with one site
    position per cue/target of every demo plus the final prompt token."""
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
            pos, texts = {}, {}
            for i in range(1, args.n_shots + 1):
                recs = selected_token_records(
                    token_labels, target_icl_example_index=i,
                    token_roles=["pre_label_token", "last_label_token"])
                by_role = {r["token_role"]: r for r in recs}
                pos[f"cue{i}"] = by_role["pre_label_token"]["token_position"]
                texts[f"cue{i}"] = by_role["pre_label_token"]["token_text"]
                pos[f"target{i}"] = by_role["last_label_token"]["token_position"]
                texts[f"target{i}"] = by_role["last_label_token"]["token_text"]
            fin = selected_token_records(token_labels, target_icl_example_index=1,
                                         token_roles=["last_prompt_token"])
            assert len(fin) == 1 and fin[0]["token_role"] == "last_prompt_token"
            pos["final_cue"] = fin[0]["token_position"]
            texts["final_cue"] = fin[0]["token_text"]

            ids = tokenizer(prompt_string, truncation=False, padding=False).input_ids
            assert pos["final_cue"] == len(ids) - 1, \
                f"{task}: final site {pos['final_cue']} != last index {len(ids) - 1}"
            ordered = [pos[s] for s in sites]
            assert all(a < b for a, b in zip(ordered, ordered[1:])), \
                f"{task}: site positions not strictly increasing: {list(zip(sites, ordered))}"
            answer = prompt_data["query_target"]["output"]       # already ' ' + str(v)
            answer_ids = get_answer_id(prompt_string, answer, tokenizer)
            prompts.append({"ids": ids, "pos": pos, "site_texts": texts, "split": split,
                            "query": query.strip(), "answer": answer.strip(),
                            "answer_id": int(answer_ids[0])})
    if args.max_prompts is not None:
        prompts = prompts[:args.max_prompts]
    return prompts


def make_chunks(prompts, tokenizer, batch_size, device, sites):
    """Left-padded chunks with per-row site positions (offset by the pad amount)."""
    site_index = {s: i for i, s in enumerate(sites)}
    pad_id = tokenizer.pad_token_id
    chunks = []
    for s in range(0, len(prompts), batch_size):
        sub = prompts[s:s + batch_size]
        mx = max(len(x["ids"]) for x in sub)
        inp = torch.full((len(sub), mx), pad_id, dtype=torch.long)
        att = torch.zeros((len(sub), mx), dtype=torch.long)
        pos = torch.zeros((len(sub), len(sites)), dtype=torch.long)
        ans = torch.zeros(len(sub), dtype=torch.long)
        for r, pr in enumerate(sub):
            pad = mx - len(pr["ids"])
            inp[r, pad:] = torch.tensor(pr["ids"], dtype=torch.long)
            att[r, pad:] = 1
            for role, si in site_index.items():
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

    if args.output_root is None:
        assert args.n_shots == 5, "--output_root must be given explicitly when --n_shots != 5"
    out_root = args.output_root or (FV_FORMATION_DIR / "fiveshot_preimage_ablation"
                                    / args.fv_root.name)
    out_root.mkdir(parents=True, exist_ok=True)

    sites = build_sites(args.n_shots)
    site_index = {s: i for i, s in enumerate(sites)}
    arm_rows = build_arm_rows(args.n_shots)

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
        "cf_map": cf_map, "sites": sites,
        "arm_rows": {a: [[name, [list(sc) for sc in spec]] for name, spec in rows]
                     for a, rows in arm_rows.items()},
        "git_commit": git_commit_hash(),
        "metric": "delta log p (ablated - clean) of the first answer token at the final position",
        "edit_layer_mapping": (
            "start layer L ablates transformer.h.{b} outputs for all b >= L at every site "
            "position of the row (edit layers 0..27; embedding entry never touched)"),
    })

    def forward_hidden(ch):
        return model.transformer(input_ids=ch["input_ids"],
                                 attention_mask=ch["attention_mask"],
                                 use_cache=False).last_hidden_state

    def run_logp(chunks, hook=None):
        """log p of each prompt's first answer token at the final position."""
        outs = []
        for ch in chunks:
            if hook is not None:
                with TraceDict(model, layers=layer_names, edit_output=hook,
                               retain_output=False):
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
        prompts = build_prompts(task, dataset, tokenizer, model_config, args, sites)
        chunks = make_chunks(prompts, tokenizer, args.batch_size, device, sites)
        n = len(prompts)

        # --- sanity: every cue site (incl. final) decodes to ':'; targets recorded only ---
        st = prompts[0]["site_texts"]
        for i in range(1, args.n_shots + 1):
            assert st[f"cue{i}"].strip() == ":", \
                f"{task}: cue{i} decodes to {st[f'cue{i}']!r}, expected ':'"
        assert st["final_cue"].strip() == ":", \
            f"{task}: final_cue decodes to {st['final_cue']!r}, expected ':'"
        tgt_txt = " ".join(f"target{i}={st[f'target{i}']!r}" for i in range(1, args.n_shots + 1))
        print(f"[{task}] n={n} | prompt_len~{len(prompts[0]['ids'])} | {tgt_txt}", flush=True)

        # --- clean pass + batched-vs-unbatched + no-op-gate checks ---
        clean_logp = run_logp(chunks)
        assert torch.isfinite(clean_logp).all(), f"{task}: non-finite clean log p"
        for i in range(min(3, n)):
            single = make_chunks([prompts[i]], tokenizer, 1, device, sites)
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
            rows = arm_rows[base]
            # one direction bank per distinct cell in this arm (matched: 11; icl10: 3; fv: 1)
            cells = {cell for _, spec in rows for _, cell in spec}
            bank = {cell: load_directions(cell, direction_task, args, device) for cell in cells}
            delta = np.full((len(rows), N_EDIT_LAYERS, n), np.nan, dtype=np.float32)
            dir_norms = np.full((len(rows), len(sites), N_EDIT_LAYERS), np.nan, dtype=np.float32)
            ta = time.time()
            for ri, (row, spec) in enumerate(rows):
                row_dirs = [(site_index[site], bank[cell][0]) for site, cell in spec]
                for site, cell in spec:
                    dir_norms[ri, site_index[site]] = bank[cell][1]
                for L in args.start_layers:
                    s = 0
                    for ch in chunks:
                        rows_idx = torch.arange(ch["n"], device=device)
                        pos = ch["pos"]

                        def hook(output, layer_name):
                            b = name_to_block[layer_name]
                            if b < L:
                                return output
                            h = output[0] if isinstance(output, tuple) else output
                            for si, U in row_dirs:
                                u = U[b]
                                p = pos[:, si]
                                v = h[rows_idx, p, :].float()
                                v = v - torch.outer(v @ u, u)
                                h[rows_idx, p, :] = v.to(h.dtype)
                            return output

                        lp = run_logp([ch], hook=hook)
                        delta[ri, L, s:s + ch["n"]] = (lp - clean_logp[s:s + ch["n"]]).cpu().numpy()
                        s += ch["n"]
            assert np.isfinite(delta[:, args.start_layers, :]).all(), f"{task}/{arm}: non-finite delta"
            np.savez(task_dir / f"{arm}_delta_logp.npz",
                     delta_logp=delta, clean_logp=clean_logp.cpu().numpy(),
                     row_names=np.array([r for r, _ in rows]),
                     row_sites=np.array([json.dumps([s for s, _ in spec]) for _, spec in rows]),
                     row_cells=np.array([json.dumps([c for _, c in spec]) for _, spec in rows]),
                     site_names=np.array(sites),
                     site_texts_example=np.array(json.dumps(prompts[0]["site_texts"])),
                     start_layers=np.array(args.start_layers),
                     split=np.array([p["split"] for p in prompts]),
                     query=np.array([p["query"] for p in prompts]),
                     answer=np.array([p["answer"] for p in prompts]),
                     answer_token_id=np.array([p["answer_id"] for p in prompts]),
                     direction_norms=dir_norms,
                     n_shots=np.array(args.n_shots),
                     cf_task=np.array(cf_map[task] if arm.endswith("_cf") else ""))
            print(f"[{task}] {arm}: {len(rows)} rows x {len(args.start_layers)} layers "
                  f"in {time.time() - ta:.0f}s", flush=True)
        print(f"[{task}] done in {time.time() - t0:.0f}s", flush=True)

    # --- summaries over whatever npz exist (ALL task dirs on disk, not just args.tasks:
    # per-task invocations must not clobber the combined CSV with a subset) ---
    combined = []
    summary_tasks = sorted(d.name for d in out_root.iterdir()
                           if d.is_dir() and any(d.glob("*_delta_logp.npz")))
    for task in summary_tasks:
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
