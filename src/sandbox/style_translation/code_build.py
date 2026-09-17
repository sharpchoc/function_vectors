#!/usr/bin/env python
"""Cheap k = 4 check corpus for the CODE conventions (code_families.py). Per language a shared pool of small tasks (Gemini); per family
and task: Gemini writes the NATURAL-style solution (>= 8 decision points spread over 20-35 lines), then rewrites it changing ONLY the
convention; the twins are aligned by a token diff (non-equal blocks = opportunities, adjacent blocks merged). Kept if >= 5 opportunities,
the 5th starts before 75 % of the code, and >= 60 % of the tokens are shared. Records -> dataset_files/style_translation/pairs/<family>.json
(text_es = task spec, langs = {src: Task, tgt: <Language>}); raw cache dataset_files/style_translation/code/<family>.json;
task pools dataset_files/style_translation/code/tasks_<lang>.json. Then cue tokens + prompts (Qwen) filtered to k in {0, 4}."""
import argparse
import argparse, difflib, json, re, subprocess, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.gen_spanish import load_key, URL
from src.sandbox.style_translation.code_families import CODE_FAMILIES, CODE_FAMILY
from src.sandbox.style_translation.models import paths as model_paths

PAIRS = STYLE_TRANSLATION_DATA / "pairs"; RAW = STYLE_TRANSLATION_DATA / "code"; PY = sys.executable
TASK_PROMPT = {
 "default": """Produce {n} short, self-contained programming tasks for {lang}, each solvable in 20-35 lines of code. Vary the domains widely
(text processing, arithmetic, records/dictionaries, dates, parsing, simple simulations, geometry, inventory, scheduling, statistics, games).
Return ONLY a JSON list of objects {{"id": "t001", "title": "...", "spec": "..."}} where spec is 2-4 sentences stating what to implement,
the inputs and outputs, the exact function/script name and signature to use, and two example calls with their expected results.""",
 "SQL": """Produce {n} short SQL tasks. Each spec describes a small schema (2-3 tables with columns) and asks for 3-5 queries (selects with
filters, joins, aggregates, inserts/updates) that a 20-35 line SQL script can satisfy. Return ONLY a JSON list of objects {{"id": "t001", "title": "...", "spec": "..."}}.""",
 "CSS": """Produce {n} short CSS tasks. Each spec describes a small page (5-8 elements with class/id names) and the visual styling required
(colors given as hex, spacing, borders, fonts) that a 20-35 line stylesheet can satisfy. Return ONLY a JSON list of objects {{"id": "t001", "title": "...", "spec": "..."}}.""",
 "Bash": """Produce {n} short Bash scripting tasks (file handling, string processing, argument checking, simple loops), each solvable in a 20-35 line
script, with the inputs/outputs and 2 example invocations. Return ONLY a JSON list of objects {{"id": "t001", "title": "...", "spec": "..."}}.""",
}
GEN = """Write a complete {lang} solution to the task below.
STYLE REQUIREMENT (this is the point of the exercise): {hint}. Apply it consistently, and make sure the places where it applies are spread
over the WHOLE solution, including the second half — not clustered at the top.
Length 20-35 lines. Plain code only: no markdown fences, no prose before or after.

TASK:
{spec}"""
REWRITE = """Below is a {lang} program. Rewrite it so that you {rewrite}.
EVERYTHING ELSE must stay byte-for-byte identical: same logic, names, spacing, comments, blank lines, line order. Do not fix or improve anything.
Return only the code, no markdown fences, no prose.

{code}"""
TOK = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)
from src.sandbox.style_translation.code_whitespace_alt import WS_FAMILIES, same_ast, transform as ws_transform
from src.sandbox.style_translation.code_free import filter_free, AFFECTED
# extra generation hints for the families whose decisions can be forced by earlier code (free-opportunity rebuild, 2026-09-16):
# the NATURAL solution must contain many FRESH choice points (new names / new loops / new blocks), not re-mentions
EXTRA_HINT = {
    "py_snake_camel": "introduce at least 8 DIFFERENT multi-word variable, parameter and function names, each a new name (do not just re-use two or three names)",
    "js_camel_snake": "introduce at least 8 DIFFERENT multi-word variable, parameter and function names, each a new name (do not just re-use two or three names)",
    "py_const_naming": "define at least 6 DIFFERENT module-level constants, each with a distinct multi-word name",
    "py_class_naming": "define at least 7 DIFFERENT small classes (each with a distinct multi-word name), spread through the whole solution",
    "py_private": "assign at least 7 DIFFERENT private attributes in __init__ and other methods, each with a distinct multi-word name",
    "py_bool_prefix": "introduce at least 7 DIFFERENT boolean variables / parameters, each with a distinct multi-word name",
    "py_loop_vars": "write at least 7 DIFFERENT for-loops (nested or sequential) spread through the whole solution, each with its own loop variable",
    "js_hungarian": "introduce at least 8 DIFFERENT variables and parameters, each with a distinct multi-word name",
    "py_abbrev": "introduce at least 8 DIFFERENT variables and parameters, each a distinct name (do not re-use the same few names)",
    "py_self_name": "define a class with at least 7 methods (each `def method(self, ...)`), spread through the whole solution",
    "py_indent": "use at least 8 DIFFERENT indented blocks (if / for / while / def / try / with), several of them nested, spread through the whole solution",
    "py_tabs": "use at least 8 DIFFERENT indented blocks (if / for / while / def / try / with), several of them nested, spread through the whole solution",
}


def _chat(key, content, temperature, max_tokens=2000):
    body = {"model": "google/gemini-2.5-flash", "temperature": temperature, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
    r = requests.post(URL, json=body, timeout=150, headers={"Authorization": f"Bearer {key}"}); r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    out = re.sub(r"^\s*```[a-zA-Z0-9+#-]*\s*\n", "", out.strip()); out = re.sub(r"\n\s*```\s*$", "", out)
    return out.rstrip() + "\n"


def tasks_for(key, lang, n=60, batch=60):
    """Task pool of at least n tasks, grown in batches of `batch` (each batch told to avoid the existing titles); ids t001.. are stable."""
    f = RAW / f"tasks_{lang}.json"
    pool = json.load(open(f)) if f.exists() else []
    while len(pool) < n:
        avoid = "; ".join(t["title"] for t in pool[-120:])
        extra = f"\nDo NOT repeat or closely paraphrase any of these existing task titles: {avoid}" if pool else ""
        for attempt in range(4):
            try:
                raw = _chat(key, TASK_PROMPT.get(lang, TASK_PROMPT["default"]).format(n=batch, lang=lang) + extra, 0.9, 12000)
                js = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
                js = [t for t in js if isinstance(t, dict) and t.get("spec")]
                assert len(js) >= 30, len(js)
                seen = {t["title"].strip().lower() for t in pool}
                for t in js:
                    if t["title"].strip().lower() in seen:
                        continue
                    seen.add(t["title"].strip().lower()); t["id"] = f"t{len(pool) + 1:03d}"; pool.append(t)
                RAW.mkdir(parents=True, exist_ok=True); json.dump(pool, open(f, "w"), indent=1); break
            except Exception as e:
                print(f"tasks {lang} batch attempt {attempt} failed: {e}", flush=True); time.sleep(3)
        else:
            raise RuntimeError(f"could not grow the task pool for {lang}")
        print(f"task pool {lang}: {len(pool)}", flush=True)
    return pool[:max(n, len(pool))]


def align(nat, alt, merge_gap=2):
    """Opportunities from a token diff. Returns (opps, shared_fraction) or (None, 0) if the rewrite is not a local edit."""
    ta, tb = TOK.findall(nat), TOK.findall(alt)
    if "".join(ta) != nat or "".join(tb) != alt:
        return None, 0.0
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    blocks = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    shared = sum(i2 - i1 for tag, i1, i2, _, _ in sm.get_opcodes() if tag == "equal") / max(len(ta), 1)
    if not blocks:
        return [], shared
    merged = [list(blocks[0])]
    for b in blocks[1:]:
        if b[0] - merged[-1][1] <= merge_gap and b[2] - merged[-1][3] <= merge_gap:
            merged[-1][1], merged[-1][3] = b[1], b[3]
        else:
            merged.append(list(b))
    offa = [0]; [offa.append(offa[-1] + len(t)) for t in ta]
    offb = [0]; [offb.append(offb[-1] + len(t)) for t in tb]
    opps = []
    for k, (i1, i2, j1, j2) in enumerate(merged):
        opps.append({"k": k, "nat": nat[offa[i1]:offa[i2]], "alt": alt[offb[j1]:offb[j2]], "nat_span": [offa[i1], offa[i2]], "alt_span": [offb[j1], offb[j2]]})
    return opps, shared


def degenerate(text):
    """Generator output to reject (2026-09-17, bugs 3a/3b): a markdown fence inside the code, a repeated top-level definition (the same
    function / class defined more than once = padded re-implementations), or a control-token artefact."""
    if "```" in text or re.search(r"<ctrl\d+>", text):
        return "fence/artefact"
    heads = re.findall(r"^(?:def|class|function|fn|const|async function)\s+([A-Za-z_]\w*)", text, re.M)
    if any(n > 1 for n in Counter(heads).values()):
        return "repeated definition"
    return None


def build_one(key, fam, task):
    for attempt in range(3):
        try:
            hint = fam.gen_hint + ("; ALSO: " + EXTRA_HINT[fam.name] if fam.name in EXTRA_HINT else "")
            nat = _chat(key, GEN.format(lang=fam.tgt_lang, hint=hint, spec=task["spec"]), 0.9)
            if nat.count("\n") < 8:
                raise ValueError("too short")
            if degenerate(nat):
                raise ValueError(degenerate(nat))
            if fam.name in WS_FAMILIES:                            # bug 4 (2026-09-17): whitespace-only families derive alt by rule
                alt = ws_transform(fam.name, nat)
                if alt is None or not same_ast(nat, alt):
                    raise ValueError("whitespace rule failed")
            else:
                alt = _chat(key, REWRITE.format(lang=fam.tgt_lang, rewrite=fam.rewrite, code=nat), 0.2)
                if degenerate(alt):
                    raise ValueError("alt " + degenerate(alt))
            opps, shared = align(nat, alt)
            ok = opps is not None and len(opps) >= 5 and shared >= 0.6 and opps[4]["nat_span"][0] < 0.75 * len(nat) and all(o["nat"] != o["alt"] for o in opps)
            rec = {"doc_id": f"{fam.name}__{task['id']}", "family": fam.name, "topic": task["title"], "angle": "code", "text_es": task["spec"].strip(),
                   "langs": {"src": "Task", "tgt": fam.tgt_lang}, "text_nat": nat, "text_alt": alt, "opps": opps or [], "k_en": len(opps or []),
                   "shared_fraction": round(shared, 3), "pass": bool(ok), "verify": None}
            if fam.name in WS_FAMILIES:
                rec["alt_rule"] = True
            if ok and fam.name in AFFECTED:                       # free-opportunity rule: the pair must keep >= 5 genuine choice points
                fr = filter_free(rec, fam.name); ok = fr is not None; rec["pass"] = bool(ok)
                if fr is not None:
                    rec.update(text_alt=fr["text_alt"], opps=fr["opps"], opps_all=fr["opps_all"], k_en=fr["k_en"], free_filter=True)
            if ok or attempt == 2:
                return rec
        except Exception as e:
            if attempt == 2:
                print(f"{fam.name}/{task['id']} FAILED: {e}", flush=True); return None
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in CODE_FAMILIES])
    ap.add_argument("--n_tasks", type=int, default=50); ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--ks", default="0,4"); ap.add_argument("--skip_prompts", action="store_true")
    ap.add_argument("--target", type=int, default=None, help="stop generating once this many passing pairs exist (chunks of 60 tasks); pairs file capped at --target")
    args = ap.parse_args()
    key = load_key(); RAW.mkdir(parents=True, exist_ok=True)
    pools = {lang: tasks_for(key, lang, n=args.n_tasks + 10) for lang in sorted({CODE_FAMILY[f].tgt_lang for f in args.families})}
    for lang, ts in pools.items():
        print(f"task pool {lang}: {len(ts)}", flush=True)
    for name in args.families:
        fam = CODE_FAMILY[name]; raw_path = RAW / f"{name}.json"
        raw = {r["doc_id"]: r for r in json.load(open(raw_path))} if raw_path.exists() else {}
        def apply_free():                                        # the free-opportunity rule, applied to every stored raw doc (old and new)
            if name in AFFECTED:
                for r in raw.values():
                    fr = filter_free(r, name) if (r.get("opps_all") or r.get("opps")) else None
                    r["pass"] = fr is not None
                    if fr is not None:
                        r.update(text_alt=fr["text_alt"], opps=fr["opps"], opps_all=fr["opps_all"], k_en=fr["k_en"], free_filter=True)
        apply_free()
        todo = [t for t in pools[fam.tgt_lang][: args.n_tasks] if f"{name}__{t['id']}" not in raw]
        chunks = [todo[i:i + 60] for i in range(0, len(todo), 60)] if args.target else [todo]
        for chunk in chunks:
            n_pass = sum(r["pass"] for r in raw.values())
            if args.target and n_pass >= args.target:
                break
            if not chunk:
                continue
            with ThreadPoolExecutor(args.workers) as ex:
                for fu in as_completed([ex.submit(build_one, key, fam, t) for t in chunk]):
                    r = fu.result()
                    if r: raw[r["doc_id"]] = r
            json.dump(sorted(raw.values(), key=lambda r: r["doc_id"]), open(raw_path, "w"), ensure_ascii=False, indent=0)
            apply_free()
            print(f"{name}: {sum(r['pass'] for r in raw.values())} passing after {len(raw)} docs", flush=True)
        apply_free(); json.dump(sorted(raw.values(), key=lambda r: r["doc_id"]), open(raw_path, "w"), ensure_ascii=False, indent=0)
        pairs = [r for r in sorted(raw.values(), key=lambda r: r["doc_id"]) if r["pass"]][: args.target or None]
        json.dump(pairs, open(PAIRS / f"{name}.json", "w"), ensure_ascii=False, indent=0)
        ks = sorted(r["k_en"] for r in raw.values())
        print(f"{name:20s} built={len(raw):3d} opps median={ks[len(ks)//2] if ks else 0:3d} pass={len(pairs):3d} shared={sum(r['shared_fraction'] for r in raw.values())/max(len(raw),1):.2f}", flush=True)
    if args.skip_prompts:
        return
    fams = [f for f in args.families if (PAIRS / f"{f}.json").exists() and json.load(open(PAIRS / f"{f}.json"))]
    subprocess.run([PY, "src/sandbox/style_translation/cue_tokens.py", "--model", "qwen25_base", "--families", *fams], check=True, cwd=_BOOT)
    subprocess.run([PY, "src/sandbox/style_translation/build_prompts.py", "--model", "qwen25_base", "--K", "5", "--families", *fams], check=True, cwd=_BOOT)
    if args.ks != "all":
        keep = {int(k) for k in args.ks.split(",")}; MP = model_paths("qwen25_base")
        for f in fams:
            p = MP["prompts"] / f"{f}.json"; items = [it for it in json.load(open(p)) if it["k"] in keep]
            json.dump(items, open(p, "w")); print(f"{f}: {len(items)} prompts kept")


if __name__ == "__main__":
    main()
