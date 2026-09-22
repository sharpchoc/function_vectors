"""Strict two-reviewer review (Opus 5 + GPT-5, AND rule, the regeneration checklist) of every pool document never reviewed before
(no 'reviewed_by' field, not in audit9/sample_review.jsonl). Resumable; verdicts -> review10/verdicts.jsonl; 200 doc-workers (400 calls in flight)."""
import json, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, '/workspace/function_vectors'); sys.path.insert(0, '/root/.claude/jobs/1f45be64/tmp')
from regen_common import review_both
from src.sandbox.style_translation import code_build as CB
from src.sandbox.style_translation.code_build import load_key, PAIRS
from src.sandbox.style_translation.code_families import CODE_FAMILY
T = '/root/.claude/jobs/1f45be64/tmp/'; OUT = T + 'review10/verdicts.jsonl'; key = load_key()
pool = json.load(open('/workspace/function_vectors/results/code_styles/code_pool.json'))['pool']
done = {json.loads(l)['doc_id'] for l in open(T + 'audit9/sample_review.jsonl')}
try: done |= {json.loads(l)['doc_id'] for l in open(OUT)}
except FileNotFoundError: pass
jobs = []
for f in pool:
    for r in json.load(open(PAIRS / f'{f}.json')):
        if not r.get('reviewed_by') and r['doc_id'] not in done: jobs.append((f, r))
print(f'docs to review {len(jobs)} (skipping {len(done)} already reviewed)', flush=True)
t0 = time.time(); n = ok = 0; lock = threading.Lock()
def work(f, r):
    F = CODE_FAMILY[f]; task = {'id': r['doc_id'].split('__')[1], 'title': r['topic'], 'spec': r['text_es']}
    vs = review_both(key, F, task, r, r['doc_id'], 0, 'strict_review', OUT); return all(v['ok'] for v in vs)
with ThreadPoolExecutor(128) as ex:
    futs = [ex.submit(work, f, r) for f, r in jobs]
    for fu in as_completed(futs):
        with lock:
            n += 1; ok += bool(fu.result())
            if n % 100 == 0: print(f'[{n}/{len(jobs)} {int(time.time()-t0)}s] accepted {ok} ({ok/n:.1%})', flush=True)
usage = {f'{m}/{k}': (round(v, 2) if k == 'cost' else v) for (m, k), v in sorted(CB.USAGE.items())}
print(f'REVIEW10 DONE: {n} docs, accepted {ok} ({ok/max(n,1):.1%}), {int(time.time()-t0)}s | USAGE {usage}', flush=True)
