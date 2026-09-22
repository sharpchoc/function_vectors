"""2026-09-21 family-tailored task design (user request): a THIRD model (Gemini 3.1 Pro, neither generator nor reviewer) designs programming
tasks whose natural solution exhibits the family's construct many times without contrived code. Input: the still-failed docs of regen6 and
the reviewers' actual objections on them. Output: dataset_files/style_translation/code/tasks_designed_<family>.json (ids d001..)."""
import json, sys, random, collections, re
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '/workspace/function_vectors')
from src.sandbox.style_translation import code_build as CB
from src.sandbox.style_translation.code_build import _chat, load_key, RAW, PAIRS, gen_hint
from src.sandbox.style_translation.code_families import CODE_FAMILY
TMP = '/root/.claude/jobs/1f45be64/tmp'; MODEL = 'google/gemini-3.1-pro-preview'; random.seed(11)
PROMPT = """You design programming tasks for a paired code-style dataset. Each task will be solved in {lang} by another model as ONE realistic
file of 40-80 lines; the file is then mechanically rewritten into a twin that differs ONLY in one coding convention:

Family `{fam}` — natural convention: {nat}. Alternative convention: {alt}.
Rewrite rule applied to the solution: "{rewrite}".
What the solution has to exhibit: "{hint}".

HARD REQUIREMENT ON THE SOLUTION (so choose tasks accordingly): at least 8 occurrences of the construct must arise from the task's OWN
logic — each on its own line, spread over the whole file (several in the first half) — with NO contrived code: no unused variables, no
checks that never fire, no helpers that exist only to add an occurrence, no example/demo blocks, no comments about the convention.
Two strict independent reviewers reject anything that looks manufactured. Generic textbook exercises ("find the max of a list") FAILED
for this family because they do not naturally need the construct. Design problems from domains where the construct is the natural tool.

These are real reviewer objections from failed attempts in this family — design tasks whose honest solution cannot run into them:
{objections}

Also make the mechanical rewrite SAFE: the convention must be applicable to every occurrence without ambiguity, and the solution should not
need the affected tokens inside string literals, comments or printed messages (a rename or literal change must never alter program output).

Write {n} DIFFERENT tasks. Each: a short title and a spec of 3-7 sentences in the style of a coding-exercise statement: the exact
function / script name(s) and signature(s), inputs and outputs, the concrete rules / constants / cases the logic must handle (this is what
makes the construct necessary — spell them out), and 1-2 input->output examples. The spec must NOT mention the coding convention, the
number of occurrences, or style. A small module with 2-4 related functions and a short driver is fine.{avoid}
{ident}Return ONLY a JSON array: [{{"title": "...", "spec": "..."}}, ...]"""

def objections(fam, sf):
    out = []
    for t in ['r1', 'r2', 'r3', 'r4', 's1', 's2', 's3', 'd1', 'strip5']:
        try:
            for l in open(f'{TMP}/{t}_review.jsonl'):
                r = json.loads(l)
                if r['family'] == fam and r['doc_id'] in sf:
                    out += [f"- (item {i['item']}) `{i['line'].strip()[:90]}` — {i['problem'][:220]}" for i in r['issues']]
        except FileNotFoundError: pass
    random.shuffle(out); return '\n'.join(out[:14]) or '- (none recorded: the solutions simply had fewer than 5 honest occurrences)'

IDENT = """IDENTIFIER RULE FOR THIS FAMILY (critical): any identifier that the spec writes as code stays IDENTICAL in both twins, so a spec that
names many identifiers produces a half-converted twin that reviewers reject. Therefore the spec may contain exactly ONE piece of code: the
entry function written with a SINGLE-WORD name and single-word parameters, e.g. `summarize(records)` or `report(rows, limit)`. Everything else
— helper functions, fields, inputs, outputs, dictionary keys, column names — must be described in PLAIN ENGLISH PROSE with no backticks, no
quotes around names, no snake_case / camelCase / prefixed words and no `name(` call forms. Examples may show literal input and output values
(lists, numbers, tuples, plain strings) but outputs should be tuples / lists / numbers / sentences, NOT dictionaries with multi-word keys.
"""
def design(key, fam, n_total, sf, v2=False):
    F = CODE_FAMILY[fam]; path = RAW / f'tasks_designed_{fam}.json'
    pool = json.load(open(path)) if path.exists() else []
    obj = objections(fam, sf)
    while len(pool) < n_total:
        n = min(25, n_total - len(pool) + 3)
        avoid = ('\nDo NOT repeat or paraphrase these existing titles: ' + '; '.join(t['title'] for t in pool)) if pool else ''
        for attempt in range(4):
            try:
                raw = _chat(key, PROMPT.format(lang=F.tgt_lang, fam=fam, nat=F.nat_label, alt=F.alt_label, rewrite=F.rewrite, hint=gen_hint(F, True), objections=obj, n=n, avoid=avoid, ident=(IDENT if v2 else '')),
                            1.0, max_tokens=30000, model=MODEL, reasoning={'effort': 'high'}, timeout=600)
                js = json.loads(raw[raw.index('['): raw.rindex(']') + 1]); js = [t for t in js if isinstance(t, dict) and t.get('spec') and t.get('title')]
                assert len(js) >= max(3, n // 2), len(js)
                seen = {t['title'].strip().lower() for t in pool}
                for t in js:
                    if t['title'].strip().lower() in seen: continue
                    seen.add(t['title'].strip().lower()); pool.append({'id': f'd{len(pool) + 1:03d}', 'title': t['title'].strip(), 'spec': t['spec'].strip(), 'designed_by': MODEL} | ({'v2': True} if v2 else {}))
                json.dump(pool, open(path, 'w'), indent=1, ensure_ascii=False); break
            except Exception as e:
                print(f'{fam} attempt {attempt} failed: {str(e)[:200]}', flush=True)
        else:
            break
        print(f'{fam}: {len(pool)} / {n_total}', flush=True)
    return fam, len(pool)

if __name__ == '__main__':
    key = load_key(); sf = set(json.load(open(f'{TMP}/regen6_still_failed.json')))
    cnt = collections.Counter(d.split('__')[0] for d in sf)
    want = {f: max(10, int(round(2.5 * c))) for f, c in cnt.items()}
    V2 = len(sys.argv) > 1 and sys.argv[1] == 'v2'
    if V2: want = {f: (len(json.load(open(RAW / f'tasks_designed_{f}.json'))) if (RAW / f'tasks_designed_{f}.json').exists() else 0) + int(n) for f, n in zip(sys.argv[2::2], sys.argv[3::2])}
    print(want, flush=True)
    with ThreadPoolExecutor(len(want)) as ex:
        for fam, n in ex.map(lambda fn: design(key, fn[0], fn[1], sf, V2), want.items()): print('DONE', fam, n, flush=True)
    print('USAGE', {f'{m}/{k}': round(v, 3) for (m, k), v in CB.USAGE.items()}, flush=True)
