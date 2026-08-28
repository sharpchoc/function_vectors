"""
Plot the recreated Figure 8 (and the ambiguous-task companion) with TWO model lines per
panel (GPT-J and Qwen3-8B-Base) plus the dotted majority-label baseline.

Reads <GENERAL_DIR>/recreate_fig8/<model_tag>/<task>.json produced by recreate_fig8.py.
"""
import os, sys, json, math, argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.paths import AMBIGUOUS_DIR, GENERAL_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = str(GENERAL_DIR / 'recreate_fig8')

# The 25 abstractive tasks shown in the paper's Figure 8 (filename order matches the paper grid)
PAPER_FIG8 = [
    'antonym', 'capitalize', 'capitalize_first_letter', 'capitalize_last_letter',
    'capitalize_second_letter', 'commonsense_qa', 'country-capital', 'country-currency',
    'english-french', 'english-german', 'english-spanish', 'landmark-country',
    'lowercase_first_letter', 'lowercase_last_letter', 'national_parks', 'next_capital_letter',
    'park-country', 'person-instrument', 'person-occupation', 'person-sport',
    'present-past', 'product-company', 'sentiment', 'singular-plural', 'synonym',
]

MODELS = [  # (tag, legend label, color)
    ('qwen3-8b-base', 'Qwen3-8B (base)', 'tab:red'),
    ('gpt-j-6b', 'GPT-J 6B', 'tab:blue'),
]


def load(tag, folder, task):
    p = os.path.join(RESULTS, tag, folder, task + '.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def make_grid(tasks, folder, out_png, ncols=4, title=None):
    n = len(tasks)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 1.9 * nrows),
                             squeeze=False)
    handles_labels = None
    for idx, task in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        baseline = None
        for tag, label, color in MODELS:
            r = load(tag, folder, task)
            if r is None:
                continue
            ax.plot(r['shots'], r['accuracy'], '-', color=color, label=label, linewidth=1.6)
            baseline = r['majority_baseline'] if baseline is None else baseline
        if baseline is not None:
            ax.axhline(baseline, linestyle=':', color='black', linewidth=1.0,
                       label='majority baseline')
        ax.set_title(task, fontsize=8)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(0, max(r['shots']) if r else 10)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Number of Shots', fontsize=6)
        ax.set_ylabel('Accuracy', fontsize=6)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()
    # blank any unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    if handles_labels:
        fig.legend(*handles_labels, loc='lower center', ncol=3, fontsize=9,
                   bbox_to_anchor=(0.5, -0.01))
    if title:
        fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    print('wrote', out_png)


def all_tasks(folder):
    d = os.path.join(RESULTS, MODELS[0][0], folder)
    return sorted(f[:-5] for f in os.listdir(d)
                  if f.endswith('.json') and not f.endswith('_responses.jsonl'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', choices=['abstractive', 'ambiguous'], required=True)
    args = ap.parse_args()
    if args.which == 'abstractive':
        make_grid(PAPER_FIG8, 'abstractive', str(GENERAL_DIR / 'figures' / 'fig8_abstractive_2models.png'),
                  title='Figure 8 recreation — few-shot ICL accuracy (abstractive tasks)')
    else:
        tasks = all_tasks('ambiguous')
        make_grid(tasks, 'ambiguous', str(AMBIGUOUS_DIR / 'figures' / 'fig_ambiguous_2models.png'),
                  title='Few-shot ICL accuracy (ambiguous tasks)')


if __name__ == '__main__':
    main()
