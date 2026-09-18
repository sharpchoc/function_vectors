#!/usr/bin/env python
"""Deterministic alternative twin for sql_keyword_case (bug 15, 2026-09-18, user decision): the LLM rewrite lower-cased identifiers, string
literals and comments as well, so the alternative twin is rebuilt BY RULE from the natural one, as for the whitespace families and bash_subst.

Rule: every word outside string literals ('...' with '' / \\' escapes), quoted identifiers ("...", `...`, [...]) and comments (-- to end of
line, /* ... */) that is written in UPPER CASE and belongs to KEYWORDS (SQL reserved words, clause words, types and the usual built-in
functions — the list covers everything the natural twins use in upper case, see `census`) is lower-cased; everything else is copied
byte for byte. Words that are not keywords (table aliases such as `AT`, `UP`, `INV`, table names in capitals, `@Variables`) keep their case.

`transform(nat)` -> alt text. `changes(nat)` -> the (start, end) offsets of the words it lower-cases. `mixed_case(nat)` -> occurrences of
RESERVED keywords written in lower / mixed case outside strings and comments (the natural twin then mixes keyword case; such twins are
rejected at generation time by `sql_ok`). `sql_ok(nat, alt)` -> the builder's acceptance: alt is transform(nat), the round trip
(upper-casing the changed words again) restores nat, the same token sequence outside the changed words, at least one change, no mixed case.
`alias_collisions(nat)` -> keyword-listed words used as identifiers (followed by `.` or introduced by `AS`), for the report.

    python -m src.sandbox.style_translation.code_sql_alt --apply [--write]     re-derive the stored alt twins (dry run without --write)
    python -m src.sandbox.style_translation.code_sql_alt <file.sql>           print the alternative rendering of one file
"""
import re

# reserved words and clause words: a lower-case occurrence outside strings / comments means the natural twin mixes keyword case
RESERVED = set("""
SELECT FROM WHERE JOIN INNER LEFT RIGHT FULL OUTER CROSS NATURAL ON AS AND OR NOT IN IS NULL GROUP BY ORDER HAVING LIMIT OFFSET
INSERT INTO VALUES UPDATE SET DELETE CREATE TABLE PRIMARY KEY FOREIGN REFERENCES INDEX VIEW DROP ALTER ADD COLUMN DISTINCT
CASE WHEN THEN ELSE END BETWEEN LIKE ILIKE EXISTS UNION ALL ASC DESC DEFAULT UNIQUE CHECK CONSTRAINT WITH TRUNCATE BEGIN COMMIT
ROLLBACK TRANSACTION USING RETURNING OVER PARTITION INTERSECT EXCEPT ANY SOME TOP FETCH DECLARE IF TRUE FALSE INTERVAL EXTRACT
DUPLICATE REPLACE IGNORE CONFLICT CASCADE RESTRICT AUTO_INCREMENT AUTOINCREMENT MERGE MATCHED RECURSIVE ESCAPE COLLATE NULLS
EXPLAIN GRANT REVOKE PROCEDURE FUNCTION TRIGGER CALL EXEC EXECUTE TEMPORARY LATERAL WINDOW FILTER WITHIN
""".split())
# built-in functions, types and interval / extract units: lower-cased when written in upper case, not flagged when lower-case
FUNCTIONS = set("""
COUNT SUM AVG MIN MAX NOW COALESCE CAST CONVERT DATE TIME TIMESTAMP DATETIME INT INTEGER BIGINT SMALLINT TINYINT VARCHAR NVARCHAR CHAR
NCHAR TEXT BOOLEAN BOOL DECIMAL NUMERIC FLOAT REAL DOUBLE MONEY BIT SERIAL UUID BLOB JSON ARRAY ENUM
CURRENT_DATE CURRENT_TIME CURRENT_TIMESTAMP CURDATE CURTIME GETDATE SYSDATE STRFTIME JULIANDAY DATE_ADD DATE_SUB DATEDIFF DATEADD
DATEPART DATE_FORMAT DATE_TRUNC TIMESTAMPDIFF UNIX_TIMESTAMP FROM_UNIXTIME TO_CHAR TO_DATE TO_NUMBER TO_TIMESTAMP EOMONTH MONTHNAME
DAYNAME DAYOFWEEK DAYOFMONTH DAYOFYEAR WEEKDAY WEEKOFYEAR
DAY MONTH YEAR HOUR MINUTE SECOND WEEK QUARTER DAYS MONTHS YEARS HOURS MINUTES SECONDS WEEKS EPOCH DOW DOY MILLISECOND
LENGTH LEN CHAR_LENGTH CHARACTER_LENGTH LOWER UPPER TRIM LTRIM RTRIM SUBSTR SUBSTRING CONCAT CONCAT_WS ROUND FLOOR CEIL CEILING ABS MOD
POWER SQRT EXP LN LOG SIGN PI RANDOM RAND NULLIF IFNULL ISNULL NVL IIF GREATEST LEAST ROW_NUMBER RANK DENSE_RANK LAG LEAD NTILE
FIRST_VALUE LAST_VALUE STRING_AGG GROUP_CONCAT ARRAY_AGG LISTAGG SCOPE_IDENTITY LAST_INSERT_ID LAST_INSERT_ROWID IDENTITY INSTR
POSITION CHARINDEX PATINDEX STUFF FORMAT PRINTF TYPEOF TOTAL MD5 NEWID AGE TRUNC LPAD RPAD REVERSE SPLIT_PART REGEXP_REPLACE
REGEXP_MATCHES STRPOS STR ASCII CHR SOUNDEX INITCAP RETURNS RETURN
""".split())
KEYWORDS = RESERVED | FUNCTIONS
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUM = re.compile(r"[0-9][0-9A-Za-z_.]*")
_TOK = re.compile(r"\w+|\s+|[^\w\s]", re.UNICODE)                  # = code_build.TOK


def _scan(text):
    """Yield (start, end, kind) over the whole text: kind 'code' for a run of ordinary code, 'str' for a string literal, 'ident' for a
    quoted identifier, 'comment' for a comment. Consecutive runs cover the text exactly."""
    i = 0; n = len(text); cs = 0
    def flush(j):
        return (cs, j, "code") if j > cs else None
    out = []
    while i < n:
        c = text[i]
        if c == "'":
            f = flush(i); out.append(f) if f else None
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2; continue
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        j += 2; continue
                    break
                j += 1
            j = min(j + 1, n); out.append((i, j, "str")); i = cs = j; continue
        if c in '"`':
            f = flush(i); out.append(f) if f else None
            j = text.find(c, i + 1); j = n if j < 0 else j + 1
            out.append((i, j, "ident")); i = cs = j; continue
        if c == "[" and i + 1 < n and (text[i + 1].isalpha() or text[i + 1] == "_"):
            j = text.find("]", i + 1); eol = text.find("\n", i + 1); eol = n if eol < 0 else eol
            if 0 < j < eol:
                f = flush(i); out.append(f) if f else None
                out.append((i, j + 1, "ident")); i = cs = j + 1; continue
        if text.startswith("--", i):
            f = flush(i); out.append(f) if f else None
            j = text.find("\n", i); j = n if j < 0 else j
            out.append((i, j, "comment")); i = cs = j; continue
        if text.startswith("/*", i):
            f = flush(i); out.append(f) if f else None
            j = text.find("*/", i + 2); j = n if j < 0 else j + 2
            out.append((i, j, "comment")); i = cs = j; continue
        i += 1
    f = flush(n); out.append(f) if f else None
    return out


def _words(text):
    """(start, end, word) of every identifier-like word in the CODE runs (numbers and @variables skipped)."""
    out = []
    for s, e, kind in _scan(text):
        if kind != "code":
            continue
        i = s
        while i < e:
            c = text[i]
            if c.isdigit():
                m = _NUM.match(text, i); i = m.end(); continue
            if c == "@" or c == ":" or c == "$":                   # @Variable / :param / $1 are never keywords
                m = _WORD.match(text, i + 1); i = (m.end() if m else i + 1); continue
            if c.isalpha() or c == "_":
                m = _WORD.match(text, i); out.append((m.start(), m.end(), m.group())); i = m.end(); continue
            i += 1
    return out


def _is_upper(w):
    return w == w.upper() and any(ch.isalpha() for ch in w)


def changes(text):
    """(start, end) of the upper-case keyword words to lower-case."""
    return [(s, e) for s, e, w in _words(text) if _is_upper(w) and w in KEYWORDS]


def transform(text):
    out = text
    for s, e in reversed(changes(text)):
        out = out[:s] + out[s:e].lower() + out[e:]
    return out


def mixed_case(text):
    """RESERVED keywords written in lower or mixed case outside strings / comments: (start, word)."""
    return [(s, w) for s, e, w in _words(text) if not _is_upper(w) and w.upper() in RESERVED]


def alias_collisions(text):
    """Keyword-listed upper-case words that the code uses as identifiers: followed by `.`, or introduced by `AS` and not a CAST type."""
    hits = []
    for s, e, w in _words(text):
        if _is_upper(w) and w in KEYWORDS:
            after = text[e:].lstrip()[:1]; before = text[max(0, s - 12):s]
            if after == "." or (re.search(r"\bAS\s+$", before) and after not in "()"):   # `CAST(x AS DATE)` is the type, not an alias
                hits.append((s, w))
    return hits


def census(texts):
    """Upper-case words of the corpus outside strings / comments: {word: count}, split into lowered (in KEYWORDS) and kept."""
    from collections import Counter
    low, kept = Counter(), Counter()
    for t in texts:
        for s, e, w in _words(t):
            if _is_upper(w):
                (low if w in KEYWORDS else kept)[w] += 1
    return low, kept


def sql_ok(nat, alt):
    """Builder acceptance: alt == transform(nat); the round trip restores nat; identical token sequence apart from the lower-cased words;
    at least one change; the natural twin has no lower / mixed-case reserved keyword."""
    if alt != transform(nat) or mixed_case(nat):
        return False
    ch = changes(nat)
    if not ch:
        return False
    back = alt
    for s, e in reversed(ch):
        back = back[:s] + back[s:e].upper() + back[e:]
    if back != nat:
        return False
    ta, tb = _TOK.findall(nat), _TOK.findall(alt)
    return len(ta) == len(tb) and all(x == y or (x.lower() == y and x in KEYWORDS) for x, y in zip(ta, tb))


def apply(write=False, families=("sql_keyword_case",)):
    """Re-derive the alt twin of every stored pair (and its raw doc) by rule, re-align, re-derive the counted list (align -> counted).
    Nothing is dropped: documents below 5 counted opportunities are reported (the caller regenerates them)."""
    import json
    from src.utils.paths import STYLE_TRANSLATION_DATA
    from src.sandbox.style_translation.code_build import align
    from src.sandbox.style_translation.code_free import counted
    PAIRS, RAW = STYLE_TRANSLATION_DATA / "pairs", STYLE_TRANSLATION_DATA / "code"
    report = {}
    for fam in families:
        recs = json.load(open(PAIRS / f"{fam}.json")); raw_p = RAW / f"{fam}.json"
        raw = {r["doc_id"]: r for r in json.load(open(raw_p))} if raw_p.exists() else {}
        n_old = n_new = 0; short = []; mixed = []; nokw = 0; bad = []; changed_alt = 0; coll = []
        for i, r in enumerate(recs):
            nat = r["text_nat"]; alt = transform(nat)
            if not sql_ok(nat, alt):
                if mixed_case(nat):
                    mixed.append((r["doc_id"], sorted({w for _, w in mixed_case(nat)})))
                else:
                    bad.append(r["doc_id"]); continue
            if alias_collisions(nat):
                coll.append((r["doc_id"], sorted({w for _, w in alias_collisions(nat)})))
            changed_alt += alt != r["text_alt"]
            opps, shared = align(nat, alt); assert opps
            full = {k: v for k, v in r.items() if k not in ("opps_all", "free_filter")}; full.update(text_alt=alt, opps=opps)
            keep = counted(fam, full); n_old += len(r["opps"]); n_new += len(keep)
            nokw += sum(1 for o in keep if not any(_is_upper(w) and w in KEYWORDS for w in _WORD.findall(o["nat"])))
            new = dict(r); new.update(text_alt=alt, opps=[dict(o, k=j) for j, o in enumerate(keep)], opps_all=opps, k_en=len(keep),
                                      shared_fraction=round(shared, 3), free_filter=True, alt_rule=True)
            new.pop("_str_bounds", None); recs[i] = new
            if len(keep) < 5:
                short.append((r["doc_id"], len(keep)))
            if r["doc_id"] in raw:
                raw[r["doc_id"]].update({k: new[k] for k in ("text_alt", "opps", "opps_all", "k_en", "shared_fraction", "free_filter", "alt_rule")})
        if write:
            json.dump(recs, open(PAIRS / f"{fam}.json", "w"), ensure_ascii=False, indent=0)
            if raw:
                json.dump(sorted(raw.values(), key=lambda r: r["doc_id"]), open(raw_p, "w"), ensure_ascii=False, indent=0)
        ks = sorted(len(r["opps"]) for r in recs)
        report[fam] = dict(n=len(recs), counted_before=n_old, counted_after=n_new, alt_changed=changed_alt, pieces_without_keyword=nokw,
                           short=short, mixed_case=mixed, round_trip_failed=bad, alias_collisions=coll, k_median=ks[len(ks) // 2])
        print(f"{fam}: docs {len(recs)}, alt changed {changed_alt}, counted {n_old} -> {n_new} (median {ks[len(ks) // 2]}), pieces without keyword {nokw}, "
              f"short {short}, mixed-case {mixed}, round-trip failed {bad}, alias collisions {coll}", flush=True)
    return report


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    if sys.argv[1:2] == ["--apply"]:
        apply(write="--write" in sys.argv)
    else:
        src = open(sys.argv[1]).read(); out = transform(src); print(out, end="")
        print("sql_ok:", sql_ok(src, out), "mixed:", mixed_case(src), file=sys.stderr)
