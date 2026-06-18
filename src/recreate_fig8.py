"""
Recreate FV Figure 8 (few-shot ICL accuracy vs. #shots) for a given model, by GENERATING
each answer until a newline (not just scoring the first token) and STORING every response.

Method:
  For each trial we sample (max_shots + 1) distinct examples: max_shots demonstrations + 1
  query. For each shot count k = 0..max_shots we build the k-shot prompt = (first k of the
  same demonstrations) + query. So shots 1..9 are nested prefixes of the 10-shot prompt for
  that query (i.e. derived from the same draw). The model GENERATES greedily until it emits
  "\n" (answer complete); we take the text before the newline as the response and score it
  with the repo's normalized exact-match metric.

Outputs per (model, task):
  results/recreate_fig8/<model_tag>/<task>.json          -- summary accuracy curve + baseline
  results/recreate_fig8/<model_tag>/<task>_responses.jsonl -- every response, for manual review
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.model_utils import load_gpt_model_and_tokenizer, set_seed
from utils.prompt_utils import ICLDataset, word_pairs_to_prompt_data, create_prompt
from utils.eval_utils import exact_match_score

ROOT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset_files')


def model_tag(name):
    return name.split('/')[-1].lower()


def list_tasks(folder):
    d = os.path.join(ROOT_DATA, folder)
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith('.json'))


def load_all_pairs(folder, task):
    return ICLDataset(os.path.join(ROOT_DATA, folder, task + '.json'))


def majority_label_baseline(ds):
    """Paper baseline: (# most-common output string) / (# total). Model-independent."""
    outputs = ds['output']
    _, counts = np.unique(np.array(outputs, dtype=object), return_counts=True)
    return float(counts.max() / len(outputs))


def build_trials(ds, n_trials, max_shots, prepend_bos, rng):
    """Return a flat list of prompt items, one per (trial, shot-count)."""
    n_pool = len(ds)
    items = []
    for t in range(n_trials):
        idx = rng.choice(n_pool, max_shots + 1, replace=False)
        demos_all = ds[list(idx[:max_shots])]          # {'input':[...], 'output':[...]}
        # NB: int-indexing ICLDataset builds a 1-row Series that upcasts mixed dtypes (e.g.
        # float input forces an int output gold like 1 -> 1.0). List-indexing preserves the
        # per-column dtype, so extract the query that way to keep golds faithful.
        q = ds[[int(idx[max_shots])]]
        query = {'input': q['input'][0], 'output': q['output'][0]}
        for k in range(max_shots + 1):
            word_pairs = {'input': demos_all['input'][:k], 'output': demos_all['output'][:k]}
            prompt_data = word_pairs_to_prompt_data(
                word_pairs, query_target_pair=query, prepend_bos_token=prepend_bos)
            prompt = create_prompt(prompt_data)
            gold = prompt_data['query_target']['output']
            gold = gold[0] if isinstance(gold, list) else gold
            items.append({'trial': t, 'shots': k,
                          'query_input': query['input'], 'gold': gold, 'prompt': prompt})
    return items


@torch.inference_mode()
def run_task(folder, task, model, tokenizer, model_config, n_trials, max_shots,
             batch_size, max_new_tokens, seed, out_dir, do_sample=False, token_budget=48000):
    ds = load_all_pairs(folder, task)
    prepend_bos = False if model_config['prepend_bos'] else True
    rng = np.random.default_rng(seed)
    items = build_trials(ds, n_trials, max_shots, prepend_bos, rng)

    old_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'                     # left-pad for batched generation
    lens = [len(tokenizer(it['prompt']).input_ids) for it in items]
    order = np.argsort(lens)

    # token-budget batching: sorted by length, cap (batch_count * max_len_in_batch) so the
    # KV cache fits regardless of prompt length (GPT-J full-MHA is the tight case).
    batches, cur = [], []
    for i in order:
        cur.append(i)
        if len(cur) * lens[i] > token_budget or len(cur) >= batch_size:
            batches.append(cur); cur = []
    if cur:
        batches.append(cur)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      pad_token_id=tokenizer.pad_token_id,
                      stop_strings=["\n"], tokenizer=tokenizer)
    try:
        for bidx in batches:
            prompts = [items[i]['prompt'] for i in bidx]
            enc = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
            out = model.generate(**enc, **gen_kwargs)
            gen = out[:, enc.input_ids.shape[1]:]        # left-pad => uniform input length
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for row, i in enumerate(bidx):
                resp = texts[row].split('\n')[0].strip() # text before the first newline
                items[i]['response'] = resp
                items[i]['correct'] = bool(exact_match_score(resp, items[i]['gold']))
    finally:
        tokenizer.padding_side = old_side

    # ---- aggregate accuracy per shot count ----
    acc, nper = [], []
    for k in range(max_shots + 1):
        flags = [it['correct'] for it in items if it['shots'] == k]
        acc.append(float(np.mean(flags))); nper.append(len(flags))

    tag = model_tag(model_config['name_or_path'])
    # ---- store every response for manual inspection ----
    with open(os.path.join(out_dir, task + '_responses.jsonl'), 'w') as f:
        for it in items:
            f.write(json.dumps({'trial': it['trial'], 'shots': it['shots'],
                                'query_input': it['query_input'], 'gold': it['gold'],
                                'response': it.get('response', ''),
                                'correct': it.get('correct', False)}) + '\n')

    summary = {
        'task': task, 'folder': folder, 'model': model_config['name_or_path'],
        'metric': 'generation_exact_match', 'n_trials': n_trials, 'n_pool': len(ds),
        'shots': list(range(max_shots + 1)), 'accuracy': acc, 'n_per_shot': nper,
        'majority_baseline': majority_label_baseline(ds),
    }
    with open(os.path.join(out_dir, task + '.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--folder', required=True, choices=['abstractive', 'ambiguous', 'extractive'])
    ap.add_argument('--tasks', nargs='*', default=None)
    ap.add_argument('--n_trials', type=int, default=200)
    ap.add_argument('--max_shots', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--token_budget', type=int, default=48000)
    ap.add_argument('--max_new_tokens', type=int, default=32)
    ap.add_argument('--do_sample', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      '..', 'results', 'recreate_fig8'))
    args = ap.parse_args()

    tasks = args.tasks or list_tasks(args.folder)
    out_dir = os.path.join(args.out_dir, model_tag(args.model_name), args.folder)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.seed)
    model, tokenizer, cfg = load_gpt_model_and_tokenizer(args.model_name, device='cuda')
    model.eval()

    print(f"[{model_tag(args.model_name)}] {args.folder}: {len(tasks)} tasks, "
          f"n_trials={args.n_trials}, max_shots={args.max_shots}, bs={args.batch_size}")
    t0 = time.time()
    for ti, task in enumerate(tasks):
        ts = time.time()
        r = run_task(args.folder, task, model, tokenizer, cfg, args.n_trials, args.max_shots,
                     args.batch_size, args.max_new_tokens, args.seed, out_dir, args.do_sample,
                     args.token_budget)
        print(f"  [{ti+1}/{len(tasks)}] {task:28s} "
              f"acc@0={r['accuracy'][0]:.2f} acc@{args.max_shots}={r['accuracy'][-1]:.2f} "
              f"base={r['majority_baseline']:.2f} ({time.time()-ts:.1f}s)", flush=True)
    print(f"Done {args.folder} for {model_tag(args.model_name)} in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
