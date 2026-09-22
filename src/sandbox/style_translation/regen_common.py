"""Shared pieces of the padding regeneration drivers (2026-09-21): reviewer prompt (14-item checklist + ext_ok note), REVISE prompt, verdict
parsing, the two independent reviewers called concurrently, and the acceptance checks on a builder record."""
import json, re, threading, time
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.insert(0, '/workspace/function_vectors')
from src.sandbox.style_translation.code_free import LINE_ALIGN_FAMILIES, counted, fifth_frac
from src.sandbox.style_translation.code_line_align import line_blocks, check_a
from src.sandbox.style_translation import code_build as CB
from src.sandbox.style_translation.code_build import gen_hint, degenerate, valid_source, padding_lines, leak_lines, _chat, GEN

TMP = '/root/.claude/jobs/1f45be64/tmp'
GEN_MODEL = 'anthropic/claude-opus-5'
REVIEWERS = [('anthropic/claude-opus-5', {'effort': 'medium'}, 0.0), ('openai/gpt-5', {'effort': 'medium'}, None)]
REVIEW = """You are an independent, strict reviewer of a paired code dataset. A pair = a programming TASK, a NATURAL solution written in a
family's natural convention, and an ALTERNATIVE twin that must differ from the natural solution ONLY in that one convention.

Family: {fam} ({lang}). NATURAL convention: {nat}. ALTERNATIVE convention: {alt}.
Rewrite rule that was supposed to produce the alternative twin from the natural one: "{rewrite}".
Generation hint the natural twin had to satisfy: "{hint}".

Procedure: first restate in one sentence which document is the natural twin, which the alternative, and what the two conventions are.
Then inspect EVERY line of BOTH documents and EVERY occurrence of the construct in both twins — not only the first lines. Be strict rather
than lenient: valid-but-unnatural code is a FAIL; a line whose only reason to exist is to raise the occurrence count is a FAIL; when in
doubt, fail. Quote the offending line verbatim for every issue.

Fail the document on ANY of these items (use the item number in the verdict):
1. Meaning change in the alternative twin (values, string contents, numbers, logic).
2. Anything changed besides the family's convention: identifiers, comments, strings, whitespace, statement order.
3. Invalid syntax or code that would not run, in either twin.
4. Convention applied inconsistently: occurrences left in the other style, mixed forms, unconverted constructs.
5. Fake / forced decisions: stacked negations (`not x not in y`, `!!x`); two or more occurrences of the construct on one line where the
   later one is dictated by the earlier one (the hint asks for one occurrence per line).
6. Padding of any kind: filler / placeholder / dummy / illustrative lines, discarded results (`_ = ...`, `let _ = ...`), comments announcing
   examples or counting occurrences, `elif True`, `for k in range(1)`, unused variables, repeated boilerplate, checks that can never fire,
   helper code that exists only to raise the occurrence count.
7. Meta-commentary naming the convention or the style requirement (e.g. "# using is None here", "// error propagation 3", "# snake_case as required").
8. Rename collisions with existing names or language builtins (shadowing a builtin, two things with the same name).
9. Family-rule violations against the rewrite rule above (e.g. a pre-declared `x = None` before an if/else for a ternary family, a hoisted
   temporary before `match` for rust_question, a partially converted construct).
10. Scope creep: the alternative touching constructs outside the family's definition (comments, docstrings, string literals, identifiers, other operators).
11. Code that does not solve the stated task, or whose function / script name or signature differs from the task's.
12. Fewer than 5 genuine occurrences of the construct in the natural twin, or the occurrences bunched at the end of the document.
13. Stray artefacts: markdown fences, control tokens, duplicated definitions, prose outside the code.
14. Line correspondence: the twins must stay aligned line by line except exactly where the convention itself changes the line structure.

NOTE on items 6 and 11: the generation hint explicitly PERMITS a realistic, complete solution — a small module with several related functions
and a short driver / main that actually uses them. Do not flag such genuinely useful, actually-called code as padding merely because the task
did not ask for it; flag it under item 6 only if it is never called, can never fire, has no effect, duplicates other code, or is obviously
contrived for the construct. The required function must still exist with the task's name and signature and give the task's example results.

Return ONLY strict JSON, no markdown fences, no prose outside it:
{{"restatement": "<one sentence>", "ok": true|false, "issues": [{{"item": <1-14>, "line": "<verbatim offending line>", "problem": "<one sentence>"}}]}}
"ok" must be false whenever "issues" is non-empty.

TASK:
{spec}

NATURAL twin ({nat}):
{nat_code}

ALTERNATIVE twin ({alt}):
{alt_code}
"""
REVISE = """You previously wrote the {lang} solution below for the task. Independent reviewers and automatic checks rejected it for the
issues listed at the end. REVISE the document so that EVERY listed issue is fixed, without introducing any new padding: keep the solution
honest, realistic and complete; do not add code, branches, checks, variables or comments whose only purpose is to create an occurrence of
the construct; remove anything that only exists for that purpose; never mention the style, the convention or the exercise in a comment.
The document must still solve the task with the stated name and signature.
STYLE REQUIREMENT (still applies): {hint}.
{nopad}Length {length} lines. Plain code only: no markdown fences, no prose before or after.

TASK:
{spec}

PREVIOUS DOCUMENT:
{prev}

ISSUES (all must be fixed):
{issues}
"""
NOPAD = GEN[GEN.index('NO PADDING'):GEN.index('Length {length}')]
REVIEW_ITEMS = {0: 'unparsed', 1: 'meaning_change', 2: 'extra_change', 3: 'invalid', 4: 'inconsistent_convention', 5: 'fake_decision', 6: 'padding', 7: 'meta_comment',
                8: 'rename_collision', 9: 'family_rule', 10: 'scope_creep', 11: 'wrong_task', 12: 'too_few_or_bunched', 13: 'artefact', 14: 'line_correspondence'}
log_lock = threading.Lock()


def parse_verdict(raw):
    t = raw.strip(); t = re.sub(r'^\s*```[a-zA-Z]*\s*', '', t); t = re.sub(r'\s*```\s*$', '', t)
    js = json.loads(t[t.find('{'):t.rfind('}') + 1])
    assert isinstance(js.get('ok'), bool) and isinstance(js.get('issues'), list)
    js['issues'] = [{'item': int(x.get('item', 0)) if str(x.get('item', '')).strip().isdigit() else 0, 'line': str(x.get('line', ''))[:300], 'problem': str(x.get('problem', ''))[:400]}
                    for x in js['issues'] if isinstance(x, dict)]
    if js['issues']:
        js['ok'] = False
    return js


def review_one(key, model, reasoning, temp, F, task, rec, tag, rnd, kind, log_path, free_occ=True):
    prompt = REVIEW.format(fam=F.name, lang=F.tgt_lang, nat=F.nat_label, alt=F.alt_label, rewrite=F.rewrite, hint=gen_hint(F, free_occ), spec=task['spec'].strip(),
                           nat_code=rec['text_nat'].rstrip(), alt_code=rec['text_alt'].rstrip())
    verdict, raw, err = None, None, None; t0 = time.time()
    for attempt in range(4):
        try:
            raw = _chat(key, prompt, temp, max_tokens=12000, model=model, reasoning=reasoning, timeout=500)
            verdict = parse_verdict(raw); err = None; break
        except Exception as e:
            err = f'{type(e).__name__}: {str(e)[:200]}'; time.sleep(3 * (attempt + 1))
    entry = {'doc_id': rec['doc_id'], 'family': F.name, 'task_id': task['id'], 'tag': tag, 'round': rnd, 'kind': kind, 'reviewer': model, 'time': time.strftime('%H:%M:%S'),
             'secs': round(time.time() - t0), 'ok': bool(verdict and verdict['ok']),
             'issues': verdict['issues'] if verdict else [{'item': 0, 'line': '', 'problem': 'unparseable verdict: ' + str(err)}],
             'restatement': verdict.get('restatement') if verdict else None, 'n_opps': len(rec['opps'])}
    if verdict is None:
        entry['raw'] = (raw or '')[:2000]
    with log_lock:
        with open(log_path, 'a') as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return entry


def review_both(key, F, task, rec, tag, rnd, kind, log_path, free_occ=True):
    with ThreadPoolExecutor(2) as ex:
        return list(ex.map(lambda r: review_one(key, r[0], r[1], r[2], F, task, rec, tag, rnd, kind, log_path, free_occ), REVIEWERS))


def reasons(F, r):
    """Acceptance checks on a builder record; returns (reasons, feedback lines)."""
    fam = F.name; fb = []
    if not r or not r.get('pass') or not r.get('free_filter'):
        n = None
        if r:
            try:
                n = len(counted(fam, dict(r, opps=r.get('opps_all') or r['opps'])))
            except Exception:
                n = None
        fb.append(f'[automatic check] the pair yields only {n if n is not None else "too few"} counted occurrences of the construct (at least 5 needed, each the first on its line, the 5th before 85 % of the file)' if n is not None and n < 5
                  else '[automatic check] the alternative twin could not be aligned to the natural one as a local edit (the rewrite must change ONLY the convention)')
        return ['builder_reject'], fb
    out = []; nat, alt = r['text_nat'], r['text_alt']; o = r['opps']
    if len(o) < 5:
        out.append('<5'); fb.append(f'[automatic check] only {len(o)} counted occurrences (at least 5 needed)')
    elif o[4]['nat_span'][0] >= fifth_frac(fam, r) * len(nat):
        out.append('5th>85%'); fb.append('[automatic check] the 5th occurrence starts after 85 % of the file: spread the occurrences over the file')
    if re.search(r'<ctrl\d+>', nat + alt) or degenerate(nat) or degenerate(alt):
        out.append('guard'); fb.append('[automatic check] stray artefact / duplicated top-level definition')
    pad = padding_lines(nat, F.tgt_lang)
    if pad:
        out.append('padding'); fb += [f'[automatic padding guard, {n}] line {i}: `{l.strip()}`' for i, n, l in pad[:5]]
    leak = leak_lines(nat, F.tgt_lang, fam)
    if leak:
        out.append('leak'); fb += [f'[automatic style-leak guard: a comment must never mention the style / convention / requirement or name the convention] line {i}: `{l.strip()}`' for i, n, l in leak[:5]]
    if not valid_source(F.tgt_lang, nat) or (fam not in ('py2_print', 'py2_except') and not valid_source(F.tgt_lang, alt)):
        out.append('invalid'); fb.append('[automatic check] a twin does not parse')
    if r['shared_fraction'] < .6:
        out.append('shared<.6'); fb.append('[automatic check] twins share < 60 % of tokens')
    if o and nat[:o[0]['nat_span'][0]] != alt[:o[0]['alt_span'][0]]:
        out.append('k0'); fb.append('[automatic check] the twins differ before the first occurrence of the construct')
    if fam in LINE_ALIGN_FAMILIES and o:
        blocks, _ = line_blocks(fam, dict(r))
        for x in o:
            full = next((b for b in blocks if b['nat_span'] == x['nat_span']), None)
            if full is None:
                out.append('span_not_block'); break
            ok, label = check_a(fam, F.tgt_lang, full)
            if not ok:
                out.append('a:' + label); fb.append(f'[automatic check] an occurrence block fails the family rule ({label}): `{x["nat"].strip()[:80]}`'); break
    return out, fb
