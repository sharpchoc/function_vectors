#!/usr/bin/env python
"""[Moved into the repo 2026-09-23 from /workspace/msj_icl/icl_69/readfeat/qwen_capture_perprompt_fvs.py; model-agnostic via --model_name and the path arguments; defaults = the Qwen2.5-7B-Instruct 96-task line.]
Qwen per-prompt FV capture. Per task, per prompt: at the final cue token, take the o_proj INPUT of
the 140 selected heads, sum W_O-projections -> per-prompt FV. Output <out_root>/<task>.pt
{sel_flat, raw (N,140,128) fp16, fv (N,3584) fp16, prompt_index}. Relaxed assert; get_attn_out_proj.
"""
import argparse, json, sys
from pathlib import Path
import torch
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from baukit import TraceDict
from src.sandbox.isolation_upper_bound.run_task import auto_batch, load_records, record_to_prompt_data
from src.utils.model_utils import load_gpt_model_and_tokenizer, set_seed, get_attn_out_proj
from src.utils.varicl_utils import split_activations_by_head
from src.utils.prompt_utils import create_prompt
from src.utils.paths import REPO_ROOT as REPO  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument('--tasks', nargs='+', required=True)
ap.add_argument('--prompts_root', default=str(REPO/'dataset_files/isolation_prompts_ext_qwen25'))
ap.add_argument('--selection_path', default=str(REPO/'artifacts/sandbox/ext_steerability_qwen25_96/pooled_sparse/selection.json'))
ap.add_argument('--out_root', default=str(REPO/'artifacts/qwen25_read/perprompt_fvs'))
ap.add_argument('--model_name', default='Qwen/Qwen2.5-7B-Instruct')
ap.add_argument('--seed', type=int, default=42); ap.add_argument('--capture_batch', type=int, default=8)
ap.add_argument('--token_budget', type=int, default=2500)
a = ap.parse_args()
a.prompts_root, a.selection_path, a.out_root = Path(a.prompts_root), Path(a.selection_path), Path(a.out_root)
set_seed(a.seed)
sel = json.load(open(a.selection_path)); sel_flat = sorted(sel['selected_flat'])
model, tokenizer, cfg = load_gpt_model_and_tokenizer(a.model_name); model.eval(); torch.set_grad_enabled(False)
nL, nH, resid = cfg['n_layers'], cfg['n_heads'], cfg['resid_dim']; head_dim = resid // nH
by_layer = {}
for f in sel_flat: by_layer.setdefault(f // nH, []).append(f % nH)
a.out_root.mkdir(parents=True, exist_ok=True)
for task in a.tasks:
    out = a.out_root / f'{task}.pt'
    if out.exists(): print(f'[{task}] exists, skip', flush=True); continue
    recs = load_records(a, task, 'train_prompts'); assert len(recs) >= 40
    raw = torch.zeros(len(recs), len(sel_flat), head_dim, dtype=torch.float16)
    fv = torch.zeros(len(recs), resid, dtype=torch.float32)
    old = tokenizer.padding_side; tokenizer.padding_side = 'right'; lin_checked = False
    try:
        sents_all = [create_prompt(record_to_prompt_data(r, cfg)) for r in recs]
        max_tok = max(len(tokenizer(s).input_ids) for s in sents_all)
        cap = auto_batch(max_tok, a.token_budget, a.capture_batch)
        for start in range(0, len(recs), cap):
            sents = sents_all[start:start+cap]
            inputs = tokenizer(sents, return_tensors='pt', padding=True).to(model.device)
            plens = inputs.attention_mask.sum(dim=1) - 1; bidx = torch.arange(len(sents), device=model.device)
            with torch.no_grad(), TraceDict(model, layers=cfg['attn_hook_names'], retain_input=True, retain_output=True) as td:
                model(**inputs)
            col = 0
            for li in range(nL):
                if li not in by_layer: continue
                lname = cfg['attn_hook_names'][li]; inp = td[lname].input; inp = inp[0] if isinstance(inp, tuple) else inp
                heads = split_activations_by_head(inp, cfg); cue = heads[bidx, plens]
                w = get_attn_out_proj(model, li).weight.detach(); wv = w.view(resid, nH, head_dim)
                if not lin_checked:
                    rebuilt = torch.einsum('bhd,ehd->be', cue.to(w.dtype), wv)
                    outp = td[lname].output; outp = outp[0] if isinstance(outp, tuple) else outp
                    ref = outp[bidx, plens]; dev = (rebuilt-ref).abs().max().item()/max(ref.abs().max().item(),1e-6)
                    assert dev < 5e-2, f'linearity gate L{li}: {dev:.3e}'; lin_checked = True
                for h in by_layer[li]:
                    act = cue[:, h]; raw[start:start+len(sents), col] = act.half().cpu()
                    fv[start:start+len(sents)] += torch.einsum('bd,ed->be', act.float(), wv[:, h].float()).cpu(); col += 1
            assert col == len(sel_flat)
    finally:
        tokenizer.padding_side = old
    torch.save({'task': task, 'sel_flat': sel_flat, 'raw': raw, 'fv': fv.half(),
                'prompt_index': [r['prompt_index'] for r in recs]}, out)
    print(f'[{task}] saved fv{tuple(fv.shape)}', flush=True)
open(a.out_root / '.fvdone', 'a').write(f'{a.tasks[0]}\n'); print('PERPROMPT FV DONE', flush=True)
