#!/usr/bin/env python
"""[Moved into the repo 2026-09-23 from /workspace/msj_icl/icl_69/readfeat/qwen_raw_mean_sweep.py; model-agnostic via --model_name and the path arguments; defaults = the Qwen2.5-7B-Instruct 96-task line.]
Qwen port of sweep_raw_mean_layers.py — raw label-token-mean steering at the '_' slot, swept over
all layers x alpha, on the 1-shot dummy scaffold. Injector uses get_decoder_block (Qwen block path).
build_items 150-assert relaxed. Faithful to the GPT-J study otherwise (T=1 sampled exact match, 150
prompts/task, ALPHAS=(0.5,1,2,4)*norm, shared-mean control from train tasks). Shardable by task.
"""
import argparse, json, sys, zlib
from pathlib import Path
import numpy as np, torch
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.model_utils import load_gpt_model_and_tokenizer, get_decoder_block
from src.utils.paths import REPO_ROOT as REPO  # noqa: E402
ALPHAS = (0.5, 1.0, 2.0, 4.0)

class Injector:
    """Add vec to block OUTPUT at masked positions (prefill only). Qwen block path via get_decoder_block."""
    def __init__(self, model, layers):
        self.vec = None; self.mask = None; self.layers = list(layers)
        self.handles = [get_decoder_block(model, l).register_forward_hook(self._hook) for l in self.layers]
    def _hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self.vec is None or self.mask is None or hs.shape[1] != self.mask.shape[1]: return None
        hs = hs.clone(); hs[self.mask] = (hs[self.mask].float() + self.vec).to(hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []

def build_items(task, prompts_root, tok):
    """Dummy-label slot = the token at len(tok(pre)) (Qwen fuses ' _\\n\\n' into one 'Ġ_ĊĊ'
    token; that fused token IS the dummy answer slot). Robust to single- or multi-token ' _'."""
    recs = json.load(open(prompts_root / task / 'train_prompts.json'))
    assert len(recs) >= 40   # RELAXED from ==150
    items = []
    for rec in recs:
        q = str(rec['query']['input']); gold = rec['query']['output']
        gold = str(gold[0] if isinstance(gold, list) else gold).strip()
        demo_inp = str(rec['demos'][0]['input']); assert demo_inp != q
        pre = f'Q: {demo_inp}\nA:'
        ids = tok(f'{pre} _\n\nQ: {q}\nA:').input_ids
        inj_idx = len(tok(pre).input_ids)          # the dummy-answer slot token
        assert '_' in tok.convert_ids_to_tokens([ids[inj_idx]])[0], f"{task}: slot token has no '_' ({tok.convert_ids_to_tokens([ids[inj_idx]])})"
        items.append({'ids': ids, 'inj_idx': inj_idx, 'gold': gold, 'gold_len': len(tok(' ' + gold).input_ids)})
    return items

def batches_by_len(items, budget, cap):
    order = sorted(range(len(items)), key=lambda i: len(items[i]['ids']))
    bs, cur, cur_max = [], [], 0
    for i in order:
        L = len(items[i]['ids']); m = max(cur_max, L)
        if cur and (len(cur) + 1) * m > budget or len(cur) >= cap:
            bs.append(cur); cur, cur_max = [], 0; m = L
        cur.append(i); cur_max = m
    if cur: bs.append(cur)
    return bs

ap = argparse.ArgumentParser()
ap.add_argument('--resid_means_root', default=str(REPO/'artifacts/qwen25_read/label_resid_means'))
ap.add_argument('--prompts_root', default=str(REPO/'dataset_files/isolation_prompts_ext_qwen25'))
ap.add_argument('--out_root', default=str(REPO/'artifacts/qwen25_read/raw_mean_steering'))
ap.add_argument('--split_path', default=str(REPO/'task_splits/qwen25_ext_steerable_96_prunedfail.json'))
ap.add_argument('--model_name', default='Qwen/Qwen2.5-7B-Instruct')
ap.add_argument('--layers', type=int, nargs='+', default=list(range(28)))
ap.add_argument('--token_budget', type=int, default=2500); ap.add_argument('--batch_cap', type=int, default=16)
ap.add_argument('--shard_idx', type=int, default=0); ap.add_argument('--shard_n', type=int, default=1)
ap.add_argument('--shared_mean', action='store_true', default=True)
ap.add_argument('--no_shared_mean', dest='shared_mean', action='store_false')
a = ap.parse_args()
for k in ('resid_means_root','prompts_root','out_root','split_path'): setattr(a, k, Path(getattr(a,k)))
split = json.load(open(a.split_path))
group = {t: 'train' for t in split['train_tasks']}; group.update({t: 'heldout' for t in split['heldout_tasks']})
tasks = sorted(group)[a.shard_idx::a.shard_n]
a.out_root.mkdir(parents=True, exist_ok=True)
print(f'{len(tasks)} tasks this shard, layers {a.layers[0]}..{a.layers[-1]}', flush=True)
model, tok, model_config = load_gpt_model_and_tokenizer(a.model_name)
tok.padding_side = 'left'
injectors = {l: Injector(model, [l]) for l in a.layers}
shared = None
if a.shared_mean:
    shared = torch.stack([torch.load(a.resid_means_root/f'{t}.pt', map_location='cpu', weights_only=False)['resid_means']
                          for t in split['train_tasks']]).mean(0)
    print(f"shared-mean from {len(split['train_tasks'])} train tasks; ||L7||={shared[7].norm():.1f}", flush=True)
for task in tasks:
    out_path = a.out_root / f'{task}.json'
    if out_path.exists(): print(f'{task}: exists, skip', flush=True); continue
    items = build_items(task, a.prompts_root, tok)
    rm = torch.load(a.resid_means_root/f'{task}.pt', map_location='cpu', weights_only=False)['resid_means']
    assert rm.shape[0] > max(a.layers)
    res = {'task': task, 'group': group[task], 'n_prompts': len(items),
           'vector_norms': {f'L{l}': round(float(rm[l].norm()),3) for l in a.layers},
           'site': 'raw label-token mean residual, injected additively at its own layer',
           'alphas': list(ALPHAS), 'conditions': {}}
    def run(cname, vec, layer):
        inj = injectors[layer]; inj.vec = None if vec is None else vec
        preds = [None]*len(items)
        for bi, b in enumerate(batches_by_len(items, a.token_budget, a.batch_cap)):
            lens = [len(items[i]['ids']) for i in b]; L = max(lens)
            ids = torch.full((len(b), L), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros(len(b), L, dtype=torch.long); mask = torch.zeros(len(b), L, dtype=torch.bool)
            for r, i in enumerate(b):
                n = lens[r]; off = L - n
                ids[r, off:] = torch.tensor(items[i]['ids']); att[r, off:] = 1
                mask[r, off + items[i]['inj_idx']] = True
            inj.mask = mask.cuda()
            max_new = min(max(items[i]['gold_len'] for i in b) + 3, 16)
            torch.manual_seed(zlib.crc32(f'{task}|{cname}|{bi}'.encode()))
            with torch.no_grad():
                gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), do_sample=True,
                                     temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=max_new,
                                     pad_token_id=tok.eos_token_id)
            inj.mask = None
            for r, i in enumerate(b):
                preds[i] = tok.decode(gen[r, L:], skip_special_tokens=True).split('\n')[0].strip()
        inj.vec = None
        acc = float(np.mean([p == it['gold'] for p, it in zip(preds, items)]))
        res['conditions'][cname] = {'acc': round(acc,4), 'preds': preds}
        print(f'{task} | {cname}: acc={acc:.3f}', flush=True)
    run('baseline', None, a.layers[0])
    for l in a.layers:
        v = rm[l].cuda()
        for al in ALPHAS: run(f'L{l}_a{al}', al*v, l)
    if shared is not None:
        res['shared_norms'] = {f'L{l}': round(float(shared[l].norm()),3) for l in a.layers}
        for l in a.layers:
            sv = shared[l].cuda()
            for al in ALPHAS: run(f'sharedL{l}_a{al}', al*sv, l)
    res['golds'] = [it['gold'] for it in items]
    json.dump(res, open(out_path, 'w'))
open(a.out_root / f'.sweep_done_{a.shard_idx}', 'w').write('ok\n'); print('shard done', flush=True)
