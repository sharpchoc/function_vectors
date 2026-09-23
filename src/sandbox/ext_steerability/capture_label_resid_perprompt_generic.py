#!/usr/bin/env python
"""[Moved into the repo 2026-09-23 from /workspace/msj_icl/icl_69/readfeat/qwen_capture_read_perprompt.py; model-agnostic via --model_name and the path arguments; defaults = the Qwen2.5-7B-Instruct 96-task line.]
Qwen per-prompt read capture: block-OUTPUT residual at the 10th-demo-label token, ALL 28 layers,
per prompt. Relaxed >=40 assert + label-gate skip (Washington D.C.). Pairs to per-prompt FV by prompt_index.
Output: <out_root>/<task>.pt {acts (N,28,3584) fp16, label_idx (N,), prompt_index (N,)}. acts.mean(0)==task mean.
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
from src.sandbox.isolation_upper_bound.run_task import auto_batch
from src.sandbox.ext_steerability.capture_label_head_means import prompt_and_label_idx
from src.utils.paths import REPO_ROOT as REPO  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument('--prompts_root', default=str(REPO/'dataset_files/isolation_prompts_ext_qwen25'))
ap.add_argument('--out_root', default=str(REPO/'artifacts/qwen25_read/label_resid_perprompt'))
ap.add_argument('--split_path', default=str(REPO/'task_splits/qwen25_ext_steerable_96_prunedfail.json'))
ap.add_argument('--model_name', default='Qwen/Qwen2.5-7B-Instruct')
ap.add_argument('--batch_size', type=int, default=8)
ap.add_argument('--token_budget', type=int, default=2500)
ap.add_argument('--shard_idx', type=int, default=0); ap.add_argument('--shard_n', type=int, default=1)
a = ap.parse_args()
a.prompts_root, a.out_root, a.split_path = Path(a.prompts_root), Path(a.out_root), Path(a.split_path)
split = json.load(open(a.split_path))
tasks = sorted(split['train_tasks'] + split['heldout_tasks'])[a.shard_idx::a.shard_n]
a.out_root.mkdir(parents=True, exist_ok=True)
model, tokenizer, mc = load_gpt_model_and_tokenizer(a.model_name)
nL, resid = mc['n_layers'], mc['resid_dim']; tokenizer.padding_side = 'right'
layer_names = mc['layer_hook_names']
print(f'{len(tasks)} tasks this shard', flush=True)
for task in tasks:
    out = a.out_root / f'{task}.pt'
    if out.exists(): print(f'{task}: exists, skip', flush=True); continue
    recs = json.load(open(a.prompts_root / task / 'train_prompts.json')); assert len(recs) >= 40
    built = []; kept = []
    for rec in recs:
        ps, li, ci = prompt_and_label_idx(rec, tokenizer)
        ids = tokenizer(ps).input_ids
        gl = tokenizer(' ' + str(rec['demos'][9]['output']).strip()).input_ids[-1]
        if ids[li] == gl and li != ci: built.append((ps, li)); kept.append(rec['prompt_index'])
    assert len(built) >= 40, f'{task}: only {len(built)} valid'
    acts = torch.zeros(len(built), nL, resid, dtype=torch.float16)
    max_tok = max(len(tokenizer(c[0]).input_ids) for c in built)
    bs = auto_batch(max_tok, a.token_budget, a.batch_size)
    for start in range(0, len(built), bs):
        chunk = built[start:start+bs]
        sents = [c[0] for c in chunk]; lidx = torch.tensor([c[1] for c in chunk], device=model.device)
        inputs = tokenizer(sents, return_tensors='pt', padding=True).to(model.device)
        bidx = torch.arange(len(chunk), device=model.device)
        with torch.no_grad(), TraceDict(model, layers=layer_names, retain_input=False, retain_output=True) as td:
            model(**inputs)
        for li_, lname in enumerate(layer_names):
            outp = td[lname].output; outp = outp[0] if isinstance(outp, tuple) else outp
            acts[start:start+len(chunk), li_] = outp[bidx, lidx].half().cpu()
    torch.save({'task': task, 'acts': acts, 'label_idx': torch.tensor([c[1] for c in built]),
                'prompt_index': kept, 'n_prompts': len(built), 'n_skipped': len(recs)-len(built)}, out)
    # sanity vs task mean
    mp = a.out_root.parent / 'label_resid_means' / f'{task}.pt'
    chk = ''
    if mp.exists():
        tm = torch.load(mp, weights_only=False)['resid_means']
        d = (acts.float().mean(0) - tm).abs().max().item(); chk = f'| mean-vs-taskmean maxdev {d:.3f}'
    print(f'{task}: {len(built)}/{len(recs)} prompts (skip {len(recs)-len(built)}) {chk}', flush=True)
open(a.out_root / f'.done_{a.shard_idx}', 'w').write('ok\n'); print('shard done', flush=True)
