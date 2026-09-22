"""Merge the strict-review verdicts (review10 + the audit9 sample) -> results/code_styles/strict_review/: verdicts.csv (one row per doc:
ok_opus, ok_gpt5, accepted, items cited by each, items cited by both), by_family.csv, summary.json, items_by_family.png."""
import json, collections, sys
import pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/root/.claude/jobs/1f45be64/tmp'); from regen_common import REVIEW_ITEMS
T = '/root/.claude/jobs/1f45be64/tmp/'; R = '/workspace/function_vectors/results/code_styles/strict_review/'; import os; os.makedirs(R, exist_ok=True)
V = collections.defaultdict(dict)
for p in [T + 'audit9/sample_review.jsonl', T + 'review10/verdicts.jsonl']:
    for l in open(p):
        r = json.loads(l); V[r['doc_id']][r['reviewer'].split('/')[1]] = r
rows = []
for d, v in V.items():
    if len(v) < 2: continue
    o, g = v['claude-opus-5'], v['gpt-5']; io = {REVIEW_ITEMS.get(i['item'], 'item0') for i in o['issues']}; ig = {REVIEW_ITEMS.get(i['item'], 'item0') for i in g['issues']}
    rows.append(dict(doc_id=d, family=o['family'], ok_opus=o['ok'], ok_gpt5=g['ok'], accepted=o['ok'] and g['ok'], items_opus=' '.join(sorted(io)), items_gpt5=' '.join(sorted(ig)), items_both=' '.join(sorted(io & ig)),
                     n_issues_opus=len(o['issues']), n_issues_gpt5=len(g['issues'])))
df = pd.DataFrame(rows).sort_values('doc_id'); df.to_csv(R + 'verdicts.csv', index=False)
fam = df.groupby('family').agg(n=('doc_id', 'size'), accepted=('accepted', 'sum'), ok_opus=('ok_opus', 'sum'), ok_gpt5=('ok_gpt5', 'sum')); fam['acc_rate'] = (fam.accepted / fam.n).round(3)
for it in ['padding', 'wrong_task', 'invalid', 'inconsistent_convention', 'fake_decision', 'family_rule', 'meaning_change', 'extra_change', 'scope_creep', 'too_few_or_bunched']:
    fam['both_' + it] = df.groupby('family').items_both.apply(lambda s: s.str.contains(it).sum())
fam.sort_values('acc_rate').to_csv(R + 'by_family.csv')
both = collections.Counter(x for s in df.items_both for x in s.split()); opus = collections.Counter(x for s in df.items_opus for x in s.split()); gpt = collections.Counter(x for s in df.items_gpt5 for x in s.split())
summ = dict(n_docs=len(df), accepted=int(df.accepted.sum()), acc_rate=round(df.accepted.mean(), 3), ok_opus=int(df.ok_opus.sum()), ok_gpt5=int(df.ok_gpt5.sum()), both_reject=int((~df.ok_opus & ~df.ok_gpt5).sum()),
            items_cited_by_both=dict(both.most_common()), items_opus=dict(opus.most_common()), items_gpt5=dict(gpt.most_common()), families_acc_rate=fam.acc_rate.to_dict())
json.dump(summ, open(R + 'summary.json', 'w'), indent=1)
top = ['padding', 'wrong_task', 'invalid', 'inconsistent_convention', 'fake_decision', 'meaning_change']
M = (fam[['both_' + t for t in top]].div(fam.n, axis=0)).loc[fam.sort_values('acc_rate').index]
fig, ax = plt.subplots(figsize=(8, max(6, .22 * len(M)))); im = ax.imshow(M.values, aspect='auto', cmap='Reds', vmin=0, vmax=1); ax.set_yticks(range(len(M))); ax.set_yticklabels([f'{f} ({fam.loc[f, "acc_rate"]:.0%})' for f in M.index], fontsize=7)
ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=30, ha='right', fontsize=8); ax.set_title('share of never-reviewed docs where BOTH reviewers cite the item (family: acceptance rate)', fontsize=9); plt.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig(R + 'items_by_family.png', dpi=150)
print(json.dumps({k: v for k, v in summ.items() if k != 'families_acc_rate'}, indent=1)); print(fam.sort_values('acc_rate')[['n', 'accepted', 'acc_rate', 'both_padding', 'both_wrong_task', 'both_invalid']].to_string())
