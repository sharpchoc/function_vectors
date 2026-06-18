"""
Render the stored ICL inferences into easy-to-read Markdown "cards": for each example we show
the full PROMPT in a block, the MODEL RESPONSE in a colored block (green if correct, red if
wrong), and the GOLD answer in its own distinct-colored block. Colors come from fenced ```diff
blocks, which GitHub / VS Code render with syntax highlighting.

The prompt text isn't stored in the JSONL, but it's fully reconstructable (deterministic seed),
so we replay the exact sampling with recreate_fig8.build_trials and pair it with the responses
(written in the same order). Prompts are model-independent (same seed/data), so we rebuild once
per task and reuse for both models.

Usage:
  python responses_to_md.py                                   # defaults below, all models/tasks
  python responses_to_md.py --shots 0 1 5 10 --examples 6     # which shot counts / how many each
  python responses_to_md.py --model qwen3-8b-base --folder ambiguous --tasks round truncate
"""
import os, sys, json, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recreate_fig8 import build_trials, load_all_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results', 'recreate_fig8')

N_TRIALS, MAX_SHOTS, SEED = 200, 10, 42   # must match the sweep


def reconstruct_prompts(folder, task):
    """Replay the exact (deterministic) sampling to recover each trial's prompt text."""
    ds = load_all_pairs(folder, task)
    rng = np.random.default_rng(SEED)
    items = build_trials(ds, N_TRIALS, MAX_SHOTS, prepend_bos=True, rng=rng)
    return items   # ordered trial-major, shot-minor — same order as the JSONL


def block(label, text, fence='text'):
    return [f"**{label}:**", f"```{fence}", text, "```", ""]


def render_pair(folder, task, model, shots_show, n_examples):
    base = os.path.join(RESULTS, model, folder)
    resp_path = os.path.join(base, task + '_responses.jsonl')
    summ_path = os.path.join(base, task + '.json')
    if not os.path.exists(resp_path):
        return None
    resp = [json.loads(l) for l in open(resp_path)]
    items = reconstruct_prompts(folder, task)
    summ = json.load(open(summ_path)) if os.path.exists(summ_path) else {}

    # pair reconstructed prompt with stored response (same order); verify alignment
    assert len(items) == len(resp), f"length mismatch {len(items)} vs {len(resp)}"
    for it, r in zip(items, resp):
        if str(it['query_input']) != str(r['query_input']) or it['shots'] != r['shots']:
            print(f"  WARN alignment off in {model}/{folder}/{task} (regen the sweep?)")
            break

    out = [f"# `{task}` — {model} ({folder})\n",
           f"- **Model:** {summ.get('model', model)}",
           f"- **Metric:** generation exact-match (greedy, stop at newline)",
           f"- **Trials per shot:** {summ.get('n_trials','?')} · "
           f"**Majority baseline:** {summ.get('majority_baseline', float('nan')):.3f}",
           f"- **Color key:** green = model correct · red = model wrong · "
           f"cyan `@@…@@` = gold answer\n"]

    # quick accuracy curve
    acc = summ.get('accuracy', [])
    if acc:
        out.append("**Accuracy by shots:** " +
                   " · ".join(f"{k}:{a:.2f}" for k, a in enumerate(acc)) + "\n")
    out.append("---\n")

    for k in shots_show:
        krows = [(it, r) for it, r in zip(items, resp) if r['shots'] == k]
        if not krows:
            continue
        ka = sum(r['correct'] for _, r in krows) / len(krows)
        # show a mix of correct and incorrect
        wrong = [x for x in krows if not x[1]['correct']]
        right = [x for x in krows if x[1]['correct']]
        nw = min(len(wrong), n_examples // 2)
        nr = min(len(right), n_examples - nw)
        nw = min(len(wrong), n_examples - nr)
        shown = sorted(right[:nr] + wrong[:nw], key=lambda x: x[1]['trial'])

        out.append(f"## {k}-shot examples — accuracy {ka:.2f} "
                   f"(showing {len(shown)} of {len(krows)})\n")
        for it, r in shown:
            ok = r['correct']
            mark = '✅ correct' if ok else '❌ wrong'
            out.append(f"### Trial {r['trial']} · {k}-shot · {mark}\n")
            out += block("Prompt", it['prompt'], 'text')
            resp_line = (('+ ' if ok else '- ') + str(r['response']).strip()) or '(empty)'
            out += block("Model response", resp_line, 'diff')
            out += block("Gold answer", f"@@ {str(r['gold']).strip()} @@", 'diff')
            out.append("---\n")
    md_path = os.path.join(base, task + '.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(out))
    return md_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--folder', default=None)
    ap.add_argument('--tasks', nargs='*', default=None)
    ap.add_argument('--shots', nargs='*', type=int, default=[0, 1, 5, 10])
    ap.add_argument('--examples', type=int, default=6)
    args = ap.parse_args()

    models = [args.model] if args.model else sorted(os.listdir(RESULTS))
    n = 0
    for model in models:
        mdir = os.path.join(RESULTS, model)
        if not os.path.isdir(mdir):
            continue
        folders = [args.folder] if args.folder else sorted(
            d for d in os.listdir(mdir) if os.path.isdir(os.path.join(mdir, d)))
        for folder in folders:
            fdir = os.path.join(mdir, folder)
            tasks = args.tasks or sorted(
                os.path.basename(p)[:-len('_responses.jsonl')]
                for p in glob.glob(os.path.join(fdir, '*_responses.jsonl')))
            for task in tasks:
                if render_pair(folder, task, model, args.shots, args.examples):
                    n += 1
            print(f"  {model}/{folder}: rendered {len(tasks)} tasks")
    print(f"Total task .md rendered: {n} "
          f"(shots={args.shots}, {args.examples} examples each)")


if __name__ == '__main__':
    main()
