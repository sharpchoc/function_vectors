"""Merge every accepted record (regen6 <suffix>_accepted.jsonl files + strip5_accepted.jsonl) into pairs / raw, one write per family.
    python tmp/finalize5.py r1_accepted.jsonl r2_accepted.jsonl ... strip5_accepted.jsonl"""
import json, sys
sys.path.insert(0, '/workspace/function_vectors')
from src.sandbox.style_translation.code_build import RAW, PAIRS
TMP = '/root/.claude/jobs/1f45be64/tmp'
acc = []
for f in sys.argv[1:]:
    acc += [json.loads(l) for l in open(f'{TMP}/{f}')]
seen = {}; new_ids = set(); dropped = []
for a in acc:                                                     # one record per old doc; a replacement task id claimed by two drivers keeps the first
    nid = a['rec']['doc_id']
    if a['old_doc_id'] in seen or nid in new_ids:
        dropped.append(a['old_doc_id']); continue
    seen[a['old_doc_id']] = a; new_ids.add(nid)
existing = {}
for a in list(seen.values()):                                     # a replacement id that already exists as an untouched doc of the family
    fam = a['fam']
    if fam not in existing:
        existing[fam] = {x['doc_id'] for x in json.load(open(PAIRS / f'{fam}.json'))}
    nid = a['rec']['doc_id']
    if nid != a['old_doc_id'] and nid in existing[fam] and nid not in seen:
        dropped.append(a['old_doc_id']); del seen[a['old_doc_id']]
print('accepted records', len(acc), 'kept', len(seen), 'dropped (duplicate replacement id)', len(dropped))
json.dump(sorted(dropped), open(f'{TMP}/finalize5_dropped.json', 'w'), indent=1)
fams = sorted({a['fam'] for a in seen.values()}); changed = {}
for fam in fams:
    recs = json.load(open(PAIRS / f'{fam}.json')); raw = {x['doc_id']: x for x in json.load(open(RAW / f'{fam}.json'))}; n = 0
    for a in seen.values():
        if a['fam'] != fam: continue
        old_id, c = a['old_doc_id'], a['rec']; n += 1
        if c['doc_id'] == old_id:
            i = next(i for i, x in enumerate(recs) if x['doc_id'] == old_id); recs[i] = c; raw[old_id] = c
        else:
            recs = [x for x in recs if x['doc_id'] != old_id] + [c]
            if old_id in raw: raw[old_id]['pass'] = False
            raw[c['doc_id']] = c
    recs = sorted(recs, key=lambda x: x['doc_id']); assert len(recs) == 200 and len({x['doc_id'] for x in recs}) == 200, (fam, len(recs))
    json.dump(recs, open(PAIRS / f'{fam}.json', 'w'), ensure_ascii=False, indent=0)
    json.dump(sorted(raw.values(), key=lambda x: x['doc_id']), open(RAW / f'{fam}.json', 'w'), ensure_ascii=False, indent=0)
    changed[fam] = n
print('written', sum(changed.values()), 'docs in', len(changed), 'families:', changed)
json.dump(changed, open(f'{TMP}/finalize5_changed.json', 'w'), indent=1)
