#!/usr/bin/env python
"""[Moved into the repo 2026-09-23 from /workspace/msj_icl/icl_69/readfeat/qwen_capture_read.py; model-agnostic via --model_name and the path arguments; defaults = the Qwen2.5-7B-Instruct 96-task line.]
Qwen copy of capture_label_resid_means.py — per-layer task-mean residual at the last token of the
10th demo label (clean 10-shot prompts). 150-assert RELAXED (next_in_group/next_in_period have 52).
Output: <out_root>/<task>.pt {resid_means (n_layers, resid_dim), n_prompts, label_idx}.
"""
import argparse, json, sys
from pathlib import Path
import torch
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from baukit import TraceDict
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.sandbox.ext_steerability.capture_label_head_means import prompt_and_label_idx
from src.utils.paths import REPO_ROOT as REPO  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--prompts_root', default=str(REPO/'dataset_files/isolation_prompts_ext_qwen25'))
ap.add_argument('--out_root', default=str(REPO/'artifacts/qwen25_read/label_resid_means'))
ap.add_argument('--split_path', default=str(REPO/'task_splits/qwen25_ext_steerable_96_prunedfail.json'))
ap.add_argument('--model_name', default='Qwen/Qwen2.5-7B-Instruct')
ap.add_argument('--batch_size', type=int, default=8)
ap.add_argument('--shard_idx', type=int, default=0); ap.add_argument('--shard_n', type=int, default=1)
a = ap.parse_args()
a.prompts_root, a.out_root, a.split_path = Path(a.prompts_root), Path(a.out_root), Path(a.split_path)
split = json.load(open(a.split_path))
tasks = sorted(split['train_tasks'] + split['heldout_tasks'])[a.shard_idx::a.shard_n]
a.out_root.mkdir(parents=True, exist_ok=True)
print(f'{len(tasks)} tasks on this shard', flush=True)
model, tokenizer, model_config = load_gpt_model_and_tokenizer(a.model_name)
nL, resid = model_config['n_layers'], model_config['resid_dim']
tokenizer.padding_side = 'right'
layer_names = model_config['layer_hook_names']; assert len(layer_names) == nL
for task in tasks:
    out = a.out_root / f'{task}.pt'
    if out.exists(): print(f'{task}: exists, skip', flush=True); continue
    recs = json.load(open(a.prompts_root / task / 'train_prompts.json'))
    assert len(recs) >= 40, f'{task}: only {len(recs)} prompts'   # RELAXED from ==150
    # keep only prompts whose 10th-demo-label token is correctly located (skip rare tokenizer
    # edge cases, e.g. labels ending in punctuation like "Washington, D.C." -> ".C" vs ".")
    built = []; kept_recs = []
    for rec in recs:
        ps, li, ci = prompt_and_label_idx(rec, tokenizer)
        ids = tokenizer(ps).input_ids
        gold_last = tokenizer(' ' + str(rec['demos'][9]['output']).strip()).input_ids[-1]
        if ids[li] == gold_last and li != ci:
            built.append((ps, li, ci)); kept_recs.append(rec)
    n_skipped = len(recs) - len(built)
    assert len(built) >= 40, f'{task}: only {len(built)} valid prompts after gate'
    recs = kept_recs
    resid_sum = torch.zeros(nL, resid, dtype=torch.float64); n_seen = 0
    for start in range(0, len(built), a.batch_size):
        chunk = built[start:start + a.batch_size]
        sentences = [c[0] for c in chunk]
        label_idx = torch.tensor([c[1] for c in chunk], device=model.device)
        inputs = tokenizer(sentences, return_tensors='pt', padding=True).to(model.device)
        bidx = torch.arange(len(chunk), device=model.device)
        with torch.no_grad(), TraceDict(model, layers=layer_names, retain_input=False, retain_output=True) as td:
            model(**inputs)
        for li_, lname in enumerate(layer_names):
            outp = td[lname].output; outp = outp[0] if isinstance(outp, tuple) else outp
            resid_sum[li_] += outp[bidx, label_idx].double().sum(dim=0).cpu()
        n_seen += len(chunk)
    assert n_seen == len(recs)
    rm = (resid_sum / n_seen).float()
    torch.save({'task': task, 'resid_means': rm, 'n_prompts': n_seen,
                'label_idx': torch.tensor([c[1] for c in built]),
                'site': 'block OUTPUT hidden state at last token of 10th demo label (clean 10-shot)'}, out)
    print(f'{task}: {n_seen}/{n_seen+n_skipped} prompts (skipped {n_skipped}) | ||rm|| L7={rm[7].norm():.1f} L13={rm[13].norm():.1f} L0={rm[0].norm():.1f}', flush=True)
open(a.out_root / f'.capture_done_{a.shard_idx}', 'w').write('ok\n'); print('shard done', flush=True)
