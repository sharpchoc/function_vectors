import json, sys, glob, statistics, collections
sys.path.insert(0, '/workspace/function_vectors'); sys.path.insert(0, '/workspace/function_vectors/src')
from src.sandbox.ext_styleprops.properties import PROPS
from src.sandbox.style_translation.translate_english import audit_nat
from src.sandbox.style_translation.families import FAMILIES
D = '/workspace/function_vectors/dataset_files/style_translation/pairs'
print(f"{'family':14s} {'pairs':>5s} {'spare':>5s} {'k_en med':>8s} {'k_en min':>8s} {'nat audit bad':>13s} {'alt other-fam bad':>17s} {'outside-span diff':>17s} {'fluent':>6s} {'style_ok(judge)':>15s} {'k_judge~k_en':>12s}")
tot = 0
for f in FAMILIES:
    recs = json.load(open(f'{D}/{f.name}.json')); tot += len(recs)
    spare = sum(1 for r in recs if r['doc_id'] not in {d['doc_id'] for d in json.load(open(f'/workspace/function_vectors/dataset_files/style_translation/final/{f.name}.json'))})
    ks = [r['k_en'] for r in recs]
    nat_bad = sum(bool(audit_nat(r['text_nat'])) for r in recs)
    # alt twin: every OTHER family must still be all-nat (skip all_caps: case-based detectors)
    alt_bad = 0
    if f.name != 'all_caps':
        for r in recs:
            t = r['text_alt']
            for g in PROPS:
                if g == f.name: continue
                if any(t[o.start:o.end] != o.nat for o in PROPS[g].find_opps(t)):
                    alt_bad += 1; break
    def strip(t, spans):
        out, pos = [], 0
        for a, b in spans: out.append(t[pos:a]); pos = b
        out.append(t[pos:]); return ''.join(out)
    diff = sum(strip(r['text_nat'], [o['nat_span'] for o in r['opps']]) != strip(r['text_alt'], [o['alt_span'] for o in r['opps']]) for r in recs)
    fl = sum(r['verify_en']['fluent'] for r in recs); st = sum(r['verify_en']['style_consistent'] for r in recs)
    close = sum(abs(r['verify_en']['k_found'] - r['k_en']) <= 1 for r in recs) / len(recs)
    print(f"{f.name:14s} {len(recs):5d} {spare:5d} {statistics.median(ks):8.0f} {min(ks):8d} {nat_bad:13d} {alt_bad:17d} {diff:17d} {fl:6d} {st:15d} {close:12.2f}")
print('total pairs', tot)
if len(sys.argv) > 1:
    print('\n===== one pair per family: the opportunity spans (nat -> alt) =====')
    for f in FAMILIES:
        r = json.load(open(f'{D}/{f.name}.json'))[3]
        print(f"\n[{f.name}] {r['doc_id']} k_en={r['k_en']}: " + ' | '.join(f"{o['nat']!r}->{o['alt']!r}" for o in r['opps'][:7]))
        if f.name in ('double_space', 'quote_punct', 'oxford_comma'):
            print('  NAT:', r['text_nat'][:420]); print('  ALT:', r['text_alt'][:420])
