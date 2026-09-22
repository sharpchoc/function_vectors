"""regen8 (2026-09-22, approved plan Part 2): regenerate the documents that failed the strict review, few retries.
Per document: designed task (reserved per doc; fallback tasks claimed under a file lock; the pool is extended by Gemini when short) ->
TWO natural-twin candidates in parallel (Opus 5, GEN prompt + the family's real reviewer objections as "do not" examples) -> builder
(alternative twin by exact rule where RULE_ALT has one, else the Opus rewrite) -> free pre-checks (regen_common.reasons + code_static_checks)
-> the best survivor (most counted occurrences, earliest 5th) -> ONE reviewer, GPT-5 (user decision) -> accepted, else ONE revision with
the feedback, else the next task (max TASKS_PER_DOC). Accepted -> <suffix>_accepted.jsonl (finalize5.py merges); verdicts -> <suffix>_review.jsonl.
    python tmp/regen8.py --suffix p1 --docs docs.json [--workers 128] [--pilot N]"""
import argparse, fcntl, json, os, random, sys, threading, time, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, '/workspace/function_vectors'); sys.path.insert(0, '/root/.claude/jobs/1f45be64/tmp')
from regen_common import TMP, REVISE, NOPAD, REVIEW_ITEMS, log_lock, review_one, reasons
from src.sandbox.style_translation import code_build as CB
from src.sandbox.style_translation.code_build import finish_pair, gen_hint, gen_length, load_key, RAW, PAIRS, _chat, GEN
from src.sandbox.style_translation.code_families import CODE_FAMILY
from src.sandbox.style_translation.code_static_checks import precheck
from src.sandbox.style_translation.code_free import counted
import design_tasks7 as DT
GEN_MODEL = 'anthropic/claude-opus-5'; REVIEWER = ('openai/gpt-5', {'effort': 'medium'}, None)
CB.REFUSAL_FALLBACK = 'google/gemini-3.1-pro-preview'
TASKS_PER_DOC, N_CAND = 3, 2
OBJ = json.load(open(f'{TMP}/regen8/objections.json'))
PY2 = {'py2_print', 'py2_iter', 'py2_except'}
BASH_NOTE = '\nTESTS: write every test so that it is expressible with the POSIX `[ ]` builtin: no `=~` regex tests, no glob patterns, no `&&` / `||` inside a test (use two tests), no unquoted `<` / `>`; quote every variable operand.'
PY2_NOTE = ('\nCOMPATIBILITY: apart from the construct itself, use only syntax that is valid in BOTH Python 2 and Python 3: no f-strings, no type '
            'hints, no walrus, no `print(a, b)` with several arguments, no `nonlocal`; format strings with % or .format.')


def gen_prompt(F, task):
    obj = OBJ.get(F.name, [])
    extra = ('\nREJECTED EXAMPLES from this family (a strict reviewer failed earlier solutions for exactly these; never do this):\n' + '\n'.join('- ' + x for x in obj)) if obj else ''
    p = GEN.format(lang=F.tgt_lang, hint=gen_hint(F, True), spec=task['spec'], length=gen_length(F, True))
    return p.replace('\nLength ', extra + (PY2_NOTE if F.name in PY2 else '') + (BASH_NOTE if F.name == 'bash_test' else '') + '\nLength ', 1)


class Claims:
    """Fallback designed tasks per family, claimed under a file lock shared by all drivers; extends the pool with Gemini when short."""
    def __init__(self, key): self.key = key; self.path = f'{TMP}/regen8/claims.json'; self.lock = threading.Lock()
    def take(self, fam, exclude):
        with self.lock, open(self.path + '.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            claims = json.load(open(self.path)) if os.path.exists(self.path) else {}
            used = set(claims.get(fam, [])) | exclude
            pool = json.load(open(RAW / f'tasks_designed_{fam}.json'))
            free = [t for t in pool if t['id'] not in used]
            if len(free) < 5:
                try: DT.design(self.key, fam, len(pool) + 40, set(), v2=True); pool = json.load(open(RAW / f'tasks_designed_{fam}.json')); free = [t for t in pool if t['id'] not in used]
                except Exception as e: print(f'{fam}: pool extension failed: {str(e)[:120]}', flush=True)
            if not free: return None
            t = free[0]; claims.setdefault(fam, []).append(t['id']); json.dump(claims, open(self.path, 'w')); return t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--suffix', required=True); ap.add_argument('--docs', required=True, help='json list of doc_ids (old ids) to regenerate')
    ap.add_argument('--workers', type=int, default=128); ap.add_argument('--pilot', type=int, default=0); ap.add_argument('--deadline_min', type=float, default=240); args = ap.parse_args()
    key = load_key(); claims = Claims(key); os.makedirs(f'{TMP}/regen8', exist_ok=True)
    review_log = f'{TMP}/regen8/{args.suffix}_review.jsonl'; accepted_path = f'{TMP}/regen8/{args.suffix}_accepted.jsonl'
    docs = json.load(open(args.docs))
    try: done = {json.loads(l)['old_doc_id'] for l in open(accepted_path)}
    except FileNotFoundError: done = set()
    docs = [d for d in docs if d not in done]
    if args.pilot: random.Random(8).shuffle(docs); docs = docs[:args.pilot]
    by_fam = collections.defaultdict(list)
    for d in docs: by_fam[d.split('__')[0]].append(d)
    existing = {f: {r['doc_id'].split('__')[1] for r in json.load(open(RAW / f'{f}.json'))} for f in by_fam}   # task ids already used by kept docs
    reserved = {}
    for f, ds in by_fam.items():                                     # one reserved designed task per doc (not yet used by any existing doc)
        pool = [t for t in json.load(open(RAW / f'tasks_designed_{f}.json')) if t['id'] not in existing[f]]
        random.Random(f).shuffle(pool)
        for d, t in zip(ds, pool): reserved[d] = t
    t_start = time.time(); deadline = t_start + args.deadline_min * 60; stat = collections.Counter(); per_fam = collections.defaultdict(collections.Counter)

    def candidate(F, task, seed):
        nat = _chat(key, gen_prompt(F, task), 0.9, max_tokens=16000, model=GEN_MODEL, reasoning={'effort': 'medium'})
        return build(F, task, nat)

    def build(F, task, nat):
        try:
            c = finish_pair(key, F, task, nat, GEN_MODEL, free_occ=True, rule_alt=True)
            rs, fb = reasons(F, c)
            if not rs:
                sc, sfb = precheck(F, c); rs += sc; fb += sfb
        except Exception as e:
            msg = str(e)[:300]
            with CB._USAGE_LOCK: CB.REJECTS[msg.split(':')[0][:60]] += 1
            return None, ['guard:' + msg.split(':')[0][:30]], [f'[automatic guard] {msg}'], nat
        return c, rs, fb, nat

    def score(c):
        o = c['opps']; return (len(o), -o[4]['nat_span'][0] if len(o) > 4 else 0)

    def attempt_task(F, task, log):
        # round 1: two candidates in parallel
        with ThreadPoolExecutor(N_CAND) as ex: cands = list(ex.map(lambda s: candidate(F, task, s), range(N_CAND)))
        stat['gen'] += N_CAND
        ok = [x for x in cands if x[0] is not None and not x[1]]
        for c, rs, fb, nat in cands:
            if rs: log.append([task['id'], 1, 'fresh', rs])
        if ok:
            c, rs, fb, nat = max(ok, key=lambda x: score(x[0]))
        else:                                                        # no candidate passed the free checks: revise the better one once
            c, rs, fb, nat = max(cands, key=lambda x: (x[0] is not None, len(x[0]['opps']) if x[0] else 0))
            if nat is None: return None
            c, rs, fb, nat = revise(F, task, nat, fb, log, 2)
            if c is None or rs: return None
        v = review_one(key, REVIEWER[0], REVIEWER[1], REVIEWER[2], F, task, c, task['id'], 1, 'fresh', review_log); stat['review'] += 1
        if v['ok']: log.append([task['id'], 1, 'fresh', []]); return c
        log.append([task['id'], 1, 'fresh', ['review:' + ','.join(sorted({REVIEW_ITEMS.get(i['item'], 'item0') for i in v['issues']}))]])
        fb2 = [f"(round 1) [reviewer gpt-5, item {i['item']}] line: `{i['line']}` — {i['problem']}" for i in v['issues']]
        c, rs, fb, nat = revise(F, task, nat, fb2, log, 2)
        if c is None or rs: return None
        v = review_one(key, REVIEWER[0], REVIEWER[1], REVIEWER[2], F, task, c, task['id'], 2, 'revise', review_log); stat['review'] += 1
        if v['ok']: log.append([task['id'], 2, 'revise', []]); return c
        log.append([task['id'], 2, 'revise', ['review:' + ','.join(sorted({REVIEW_ITEMS.get(i['item'], 'item0') for i in v['issues']}))]]); return None

    def revise(F, task, prev_nat, feedback, log, rnd):
        issues = '\n'.join(f'- {x}' for x in feedback[-30:])
        prompt = REVISE.format(lang=F.tgt_lang, hint=gen_hint(F, True), nopad=NOPAD, length=gen_length(F, True), spec=task['spec'], prev=prev_nat.rstrip(), issues=issues)
        if F.name in PY2: prompt = prompt.replace('\nLength ', PY2_NOTE + '\nLength ', 1)
        nat = _chat(key, prompt, 0.7, max_tokens=16000, model=GEN_MODEL, reasoning={'effort': 'medium'}); stat['gen'] += 1
        c, rs, fb, nat = build(F, task, nat)
        if rs: log.append([task['id'], rnd, 'revise', rs])
        return c, rs, fb, nat

    def work(d):
        fam = d.split('__')[0]; F = CODE_FAMILY[fam]; log = []; t0 = time.time(); c = None; task = None; tried = set()
        for k in range(TASKS_PER_DOC):
            if time.time() > deadline: break
            task = reserved.get(d) if k == 0 and d in reserved else claims.take(fam, tried | existing[fam])
            if task is None: break
            tried.add(task['id']); c = attempt_task(F, task, log)
            if c is not None: break
        out = dict(doc=d, fam=fam, ok=c is not None, task=task['id'] if (c is not None and task) else None, log=log, secs=round(time.time() - t0), first=bool(c is not None and len(log) == 1))
        if c is not None:
            c['gen_model'] = GEN_MODEL; c['reviewed_by'] = [REVIEWER[0]]; c['task_designed_by'] = task.get('designed_by'); c['regen'] = 'regen8'
            with log_lock:
                with open(accepted_path, 'a') as fh: fh.write(json.dumps({'fam': fam, 'old_doc_id': d, 'rec': c, 'rounds': len(log), 'replaced': task['id'], 'log': log}, ensure_ascii=False) + '\n')
        return out

    print(f'{args.suffix}: docs {len(docs)} over {len(by_fam)} families | reserved tasks {len(reserved)}', flush=True); results = []
    with ThreadPoolExecutor(args.workers) as ex:
        for fu in as_completed([ex.submit(work, d) for d in docs]):
            r = fu.result(); results.append(r); per_fam[r['fam']]['ok'] += r['ok']; per_fam[r['fam']]['first'] += r['first']; per_fam[r['fam']]['n'] += 1
            if len(results) % 25 == 0 or args.pilot:
                ok = sum(x['ok'] for x in results); first = sum(x['first'] for x in results)
                cost = sum(v for (m, k), v in CB.USAGE.items() if k == 'cost')
                print(f"[{len(results)}/{len(docs)} {int(time.time() - t_start)}s] ok {ok} ({ok / len(results):.0%}) first-attempt {first} ({first / len(results):.0%}) | gens {stat['gen']} reviews {stat['review']} | ${cost:.0f} (${cost / max(ok, 1):.2f}/accepted)", flush=True)
    usage = {f'{m}/{k}': (round(v, 2) if k == 'cost' else v) for (m, k), v in sorted(CB.USAGE.items())}
    ok = sum(x['ok'] for x in results); first = sum(x['first'] for x in results)
    json.dump({'results': results, 'usage': usage, 'builder_rejects': dict(CB.REJECTS), 'per_family': {f: dict(c) for f, c in per_fam.items()}, 'secs': round(time.time() - t_start)}, open(f'{TMP}/regen8/{args.suffix}_log.json', 'w'), indent=1)
    print(f"REGEN8 {args.suffix} DONE: {ok}/{len(results)} ok ({ok / max(len(results), 1):.0%}), first-attempt {first / max(len(results), 1):.0%}, {int(time.time() - t_start)}s | USAGE {usage} | REJECTS {dict(CB.REJECTS)}", flush=True)
    for f, c in sorted(per_fam.items()): print(f"  {f:22s} ok {c['ok']:3d}/{c['n']:3d}  first {c['first']:3d}", flush=True)


if __name__ == '__main__':
    main()
