#!/usr/bin/env python
"""Summarise the code-translation corpus (code_translate.py): pass rates per family and language, the unresolved documents, and a review page
with random passed examples (source program next to both Python twins). Outputs results/code_styles/translation/{pass_rates.csv, unresolved.csv,
summary.json} and the HTML page path given by --html."""
import argparse
import csv
import difflib
import glob
import html
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.style_translation.logprob_margin import PAIRS
from src.sandbox.style_translation.code_translate import LANGS

OUT = ROOT / "results/code_styles/translation"; OUT.mkdir(parents=True, exist_ok=True)
STYLE = """<style>
:root { --bg:#f7f6f2; --fg:#1d1c1a; --muted:#6b675f; --rule:#dcd8cf; --panel:#fffdf9; --accent:#8a4b1f; --hl:#fff1c2; --body:'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif; --mono:'IBM Plex Mono', Menlo, Consolas, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#16171a; --fg:#ebe8e1; --muted:#a19c92; --rule:#33353a; --panel:#1f2125; --accent:#e0a06a; --hl:#4a3f14; color-scheme:dark } }
:root[data-theme="dark"] { --bg:#16171a; --fg:#ebe8e1; --muted:#a19c92; --rule:#33353a; --panel:#1f2125; --accent:#e0a06a; --hl:#4a3f14; color-scheme:dark }
body { background:var(--bg); color:var(--fg); font-family:var(--body); font-size:15px; line-height:1.5; margin:0; padding-block:28px 56px; padding-inline:16px; }
main { max-width:1500px; margin:0 auto; display:flex; flex-direction:column; gap:36px; min-width:0; }
h1 { font-size:1.8rem; margin:0 0 6px; text-wrap:balance; } h2 { font-size:1.15rem; margin:0 0 8px; } h3 { font-size:.85rem; color:var(--muted); font-weight:500; margin:0 0 6px; text-transform:uppercase; letter-spacing:.06em; }
p { margin:0 0 8px; max-width:90ch; } .note { color:var(--muted); font-size:.86rem; } .task { font-size:.95rem; }
.cols { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:12px; } @media (max-width:1000px) { .cols { grid-template-columns:1fr; } }
pre { background:var(--panel); border:1px solid var(--rule); padding:10px; margin:0; overflow:auto; max-height:60vh; font-family:var(--mono); font-size:.74rem; line-height:1.4; }
code span { display:block; white-space:pre; } .hl { background:var(--hl); }
.tw { overflow-x:auto; } table { border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:.86rem; } td, th { padding:4px 10px; border-bottom:1px solid var(--rule); text-align:right; } td:first-child, th:first-child { text-align:left; }
.low { color:#a33a2e; } details { border:1px solid var(--rule); padding:8px 12px; } summary { cursor:pointer; }
</style>"""


def code_block(s, marks=None):
    rows = []
    for l in s.splitlines():
        rows.append('<span' + (' class="hl"' if marks and l in marks else '') + '>' + html.escape(l) + '</span>')
    return '<pre><code>' + '\n'.join(rows) + '</code></pre>'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "artifacts/style_translation/qwen25_code/translation/v1")
    ap.add_argument("--html", type=Path, required=True); ap.add_argument("--per_lang", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); rng = random.Random(args.seed)
    fams = {r["family"]: r for r in csv.DictReader(open(ROOT / "results/code_styles/explainer/families.csv"))}
    rows = []; unresolved = []; passed = []; attempts = []
    for f in sorted(glob.glob(str(args.src / "*.json"))):
        R = json.load(open(f)); fam = Path(f).stem
        row = dict(family=fam, n=len(R), passed=sum(r["passed"] for r in R), mean_attempts=round(sum(r["attempts"] for r in R) / max(len(R), 1), 2))
        for l in LANGS:
            v = [r for r in R if r["lang"] == l]; row[l] = f"{sum(r['passed'] for r in v)}/{len(v)}"
        rows.append(row)
        for r in R:
            if r["passed"]:
                passed.append(r)
            else:
                J = r.get("judges") or {}; unresolved.append(dict(family=fam, doc_id=r["doc_id"], lang=r["lang"], attempts=r["attempts"], **{n: j.get("notes", "")[:200] for n, j in J.items()}))
    with open(OUT / "pass_rates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["family", "n", "passed", "mean_attempts"] + LANGS); w.writeheader(); w.writerows(rows)
    with open(OUT / "unresolved.csv", "w", newline="") as fh:
        keys = sorted({k for u in unresolved for k in u}); w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(unresolved)
    tot = sum(r["n"] for r in rows); ok = sum(r["passed"] for r in rows)
    by_lang = {l: (sum(int(r[l].split("/")[0]) for r in rows), sum(int(r[l].split("/")[1]) for r in rows)) for l in LANGS}
    summ = dict(families=len(rows), documents=tot, passed=ok, pass_rate=round(ok / max(tot, 1), 4), by_language={l: f"{a}/{b}" for l, (a, b) in by_lang.items()}, unresolved=len(unresolved))
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
    # review page: per language, random passed examples from different families
    secs = []
    for l in LANGS:
        pool = [r for r in passed if r["lang"] == l]; rng.shuffle(pool); seen = set(); picked = []
        for r in pool:
            if r["family"] in seen:
                continue
            seen.add(r["family"]); picked.append(r)
            if len(picked) == args.per_lang:
                break
        for r in picked:
            rec = {x["doc_id"]: x for x in json.load(open(PAIRS / f"{r['family']}.json"))}[r["doc_id"]]
            nat, alt = rec["text_nat"], rec["text_alt"]
            dn = [x for x in difflib.unified_diff(nat.splitlines(), alt.splitlines(), lineterm="", n=0) if x.startswith(("-", "+")) and not x.startswith(("---", "+++"))]
            mn = {x[1:] for x in dn if x.startswith("-")}; ma = {x[1:] for x in dn if x.startswith("+")}
            fi = fams.get(r["family"], {})
            secs.append(f'<section><h2>{l}: {r["family"]} ({html.escape(fi.get("natural", ""))} versus {html.escape(fi.get("alternative", ""))}), {r["doc_id"]}</h2>'
                        f'<p class="task"><strong>Task.</strong> {html.escape(rec["text_es"])}</p>'
                        f'<p class="note">Attempts {r["attempts"]}; Python {nat.count(chr(10)) + 1} lines, {l} {r["code"].count(chr(10)) + 1} lines; twins differ on {len(mn)} lines (highlighted).</p>'
                        f'<div class="cols"><div><h3>{l} source program (shared by both twins)</h3>{code_block(r["code"])}</div><div><h3>Python, natural convention</h3>{code_block(nat, mn)}</div><div><h3>Python, alternative convention</h3>{code_block(alt, ma)}</div></div></section>')
    tbl = ''.join(f'<tr><td>{r["family"]}</td><td>{r["n"]}</td><td class="{"low" if r["passed"] / max(r["n"], 1) < .7 else ""}">{r["passed"]}</td>' + ''.join(f'<td>{r[l]}</td>' for l in LANGS) + f'<td>{r["mean_attempts"]}</td></tr>' for r in rows)
    unr = ''.join(f'<tr><td>{u["family"]}</td><td>{u["doc_id"]}</td><td>{u["lang"]}</td><td style="text-align:left">{html.escape(u.get("gemini", "") or "")}</td><td style="text-align:left">{html.escape(u.get("claude", "") or "")}</td></tr>' for u in unresolved)
    page = ('<title>Code Translation Corpus</title>\n<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">\n' + STYLE +
            f'\n<main>\n<header><h1>Code-translation corpus, review</h1><p>Each Python convention document\'s natural twin translated by GPT-5 into one of four languages under a fixed style guide, accepted only when Gemini 2.5 Pro and Claude Sonnet 4.5 both pass it (definitions, logic, runnable, style). {summ["families"]} families, {tot} documents, {ok} passed ({summ["pass_rate"]:.0%}); by language: ' + ', '.join(f'{l} {v}' for l, v in summ["by_language"].items()) + '.</p></header>'
            f'<section><h2>Pass rates by family</h2><div class="tw"><table><thead><tr><th>family</th><th>documents</th><th>passed</th>' + ''.join(f'<th>{l}</th>' for l in LANGS) + '<th>mean attempts</th></tr></thead><tbody>' + tbl + '</tbody></table></div></section>'
            + '\n'.join(secs) +
            f'<section><h2>Unresolved documents ({len(unresolved)})</h2><details><summary>Show the list with the judges\' last notes</summary><div class="tw"><table><thead><tr><th>family</th><th>document</th><th>language</th><th>Gemini</th><th>Claude</th></tr></thead><tbody>{unr}</tbody></table></div></details></section>\n</main>\n')
    args.html.write_text(page); print(json.dumps(summ))


if __name__ == "__main__":
    main()
