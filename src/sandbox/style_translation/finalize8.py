"""Merge regen8 accepted records into pairs / raw for a FAMILY SUBSET (families as argv; accepted files = regen8/*_accepted.jsonl).
Same semantics as finalize5 (one record per old doc; replacement ids must be unique; families stay at 200 docs)."""
import json, sys, glob
sys.path.insert(0, '/workspace/function_vectors')
from src.sandbox.style_translation.code_build import RAW, PAIRS
T = '/root/.claude/jobs/1f45be64/tmp/regen8/'; fams = set(sys.argv[1:])
acc = [json.loads(l) for p in glob.glob(T + '*_accepted.jsonl') for l in open(p)]
seen = {}; dropped = []
for a in acc:
    if a['fam'] not in fams: continue
    nid = a['rec']['doc_id']
    if a['old_doc_id'] in seen or nid in {x['rec']['doc_id'] for x in seen.values()}: dropped.append(a['old_doc_id']); continue
    seen[a['old_doc_id']] = a
changed = {}
for fam in sorted(fams):
    recs = json.load(open(PAIRS / f'{fam}.json')); raw = {x['doc_id']: x for x in json.load(open(RAW / f'{fam}.json'))}; ids = {x['doc_id'] for x in recs}; n = 0
    for a in seen.values():
        if a['fam'] != fam: continue
        old_id, c = a['old_doc_id'], a['rec']
        if c['doc_id'] != old_id and c['doc_id'] in ids and c['doc_id'] not in seen: dropped.append(old_id); continue
        if c['doc_id'] == old_id:
            i = next(i for i, x in enumerate(recs) if x['doc_id'] == old_id); recs[i] = c; raw[old_id] = c
        else:
            recs = [x for x in recs if x['doc_id'] != old_id] + [c]
            if old_id in raw: raw[old_id]['pass'] = False
            raw[c['doc_id']] = c
        ids = {x['doc_id'] for x in recs}; n += 1
    recs = sorted(recs, key=lambda x: x['doc_id']); assert len(recs) == 200 and len(ids) == 200, (fam, len(recs), len(ids))
    json.dump(recs, open(PAIRS / f'{fam}.json', 'w'), ensure_ascii=False, indent=0); json.dump(sorted(raw.values(), key=lambda x: x['doc_id']), open(RAW / f'{fam}.json', 'w'), ensure_ascii=False, indent=0); changed[fam] = n
print('written', sum(changed.values()), 'docs in', len(changed), 'families; dropped', len(dropped)); print(changed)
