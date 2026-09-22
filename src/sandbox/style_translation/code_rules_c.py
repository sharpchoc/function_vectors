"""Exact rule-based alternative twins, GROUP C (non-Python families), 2026-09-22 (regen8 preparation).

Each rule: alt = <family>(nat, spec="") applies the family's rewrite (code_families.py) to EVERY covered occurrence and to nothing else,
or returns None when it cannot guarantee that (ambiguous construct, unsupported syntax, nothing to convert). Every rule works on a lexed
view of the source (string literals, template literals, regex literals and comments are separate segments; only `code` segments — and the
`${...}` expressions inside template literals — are rewritten, except the comment rule and the identifier rule, which also renames the
identifiers mentioned in comments as the reviewed pairs do), and every result is re-checked with the language parser
(code_build.valid_source / bash_ok). Validation against the stored pairs: /root/.claude/jobs/1f45be64/tmp/rules8/report_c.md.

    js_quotes       "..." -> '...'  (inner \\" unescaped, inner ' escaped; template literals untouched)
    js_semicolons   statement-terminating ; removed (end of line or before }, not inside ( ) / [ ]); declines on ASI hazards or two
                    statements on one line
    js_var          const / let -> var
    js_strict_eq    === / !== -> == / !=
    js_camel_snake  camelCase identifiers BOUND in the file (function names, parameters, variables, catch names, destructured names)
                    -> snake_case at every occurrence (code, template expressions, comments); task-pinned identifiers unchanged;
                    declines when an unbound camelCase identifier that is not a JS builtin remains (property names, keys, foreign globals)
    php_array       [ ... ] array literals -> array( ... )   (index access and [$a, $b] = destructuring untouched)
    r_assignment    <-  ->  =   (declines on <<- / -> in code, or an assignment inside call arguments)
    c_comment_style // text -> /* text */ (line comments and trailing comments; a second // on the same comment line starts a new block)
    bash_test       [[ expr ]] -> [ expr ]  EXCEPT tests that [ ] cannot express (=~, &&, ||, unquoted < >, glob patterns, ( ),
                    unquoted string operands): those stay [[ ]] in BOTH twins (non-opportunities)
    sql_bool_case   true / false -> TRUE / FALSE outside strings and comments
    hex_constants   decimal integer literals of the family's set -> 0xFF style (hex_constants_pow2: every power of two >= 16 and 2^n-1 >= 15)
"""
import re

from src.sandbox.style_translation.code_build import valid_source, bash_ok

# ----------------------------------------------------------------------------------------------------------------------- lexers
_REGEX_PREV = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_WORDS = {"return", "typeof", "case", "do", "else", "in", "of", "instanceof", "new", "delete", "void", "throw", "yield", "await"}


def js_segments(text):
    """[(kind, start, end)] kinds: code, dq, sq, template, regex, lcomment, bcomment. A template literal is one segment (${} included)."""
    out = []; i = 0; n = len(text); cs = 0
    def flush(j):
        nonlocal cs
        if j > cs: out.append(("code", cs, j))
    while i < n:
        c = text[i]; nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            flush(i); j = text.find("\n", i); j = n if j < 0 else j; out.append(("lcomment", i, j)); i = cs = j; continue
        if c == "/" and nxt == "*":
            flush(i); j = text.find("*/", i + 2); j = n if j < 0 else j + 2; out.append(("bcomment", i, j)); i = cs = j; continue
        if c in "\"'":
            flush(i); j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n); out.append(("dq" if c == '"' else "sq", i, j)); i = cs = j; continue
        if c == "`":
            flush(i); j = i + 1; depth = 0
            while j < n:
                if text[j] == "\\": j += 2; continue
                if text[j] == "$" and j + 1 < n and text[j + 1] == "{": depth += 1; j += 2; continue
                if text[j] == "}" and depth: depth -= 1; j += 1; continue
                if text[j] == "`" and not depth: break
                j += 1
            j = min(j + 1, n); out.append(("template", i, j)); i = cs = j; continue
        if c == "/":
            k = i - 1
            while k >= 0 and text[k] in " \t": k -= 1
            prev = text[k] if k >= 0 else ""
            m = re.search(r"([A-Za-z_$][\w$]*)$", text[max(0, k - 12):k + 1]) if k >= 0 else None
            if prev == "" or prev == "\n" or prev in _REGEX_PREV or (m and m.group(1) in _REGEX_WORDS):
                j = i + 1; cls = False
                while j < n and text[j] != "\n":
                    ch = text[j]
                    if ch == "\\": j += 2; continue
                    if ch == "[": cls = True
                    elif ch == "]": cls = False
                    elif ch == "/" and not cls: break
                    j += 1
                if j < n and text[j] == "/":
                    j += 1
                    while j < n and text[j].isalpha(): j += 1
                    flush(i); out.append(("regex", i, j)); i = cs = j; continue
        i += 1
    flush(n)
    return out


def _generic_segments(text, line_comment=None, block=("/*", "*/"), quotes=("'", '"'), sql_quotes=False, hash_comments=False):
    """Simple lexer for PHP / R / Bash / SQL: code, string, lcomment, bcomment segments."""
    out = []; i = 0; n = len(text); cs = 0
    def flush(j):
        nonlocal cs
        if j > cs: out.append(("code", cs, j))
    while i < n:
        c = text[i]
        lc = (line_comment and text.startswith(line_comment, i)) or (hash_comments and c == "#" and (i == 0 or text[i - 1] != "$"))
        if lc and not (line_comment == "#" and i > 0 and text[i - 1] == "$"):
            flush(i); j = text.find("\n", i); j = n if j < 0 else j; out.append(("lcomment", i, j)); i = cs = j; continue
        if block and text.startswith(block[0], i):
            flush(i); j = text.find(block[1], i + 2); j = n if j < 0 else j + len(block[1]); out.append(("bcomment", i, j)); i = cs = j; continue
        if c in quotes:
            flush(i); j = i + 1
            while j < n:
                if sql_quotes and c == "'":
                    if text[j] == "'" and j + 1 < n and text[j + 1] == "'": j += 2; continue
                    if text[j] == "'": break
                    j += 1; continue
                if text[j] == "\\": j += 2; continue
                if text[j] == c: break
                j += 1
            j = min(j + 1, n); out.append(("string", i, j)); i = cs = j; continue
        i += 1
    flush(n)
    return out


_TPL_EXPR = re.compile(r"\$\{((?:[^{}]|\{[^{}]*\})*)\}")


def _rewrite_js(text, segs, fn, comments=False):
    """Apply fn to code segments and to the ${...} expressions of template literals (and to comments if asked); everything else identical."""
    parts = []
    for kind, a, b in segs:
        s = text[a:b]
        if kind == "code" or (comments and kind in ("lcomment", "bcomment")):
            parts.append(fn(s))
        elif kind == "template":
            parts.append(_TPL_EXPR.sub(lambda m: "${" + fn(m.group(1)) + "}", s))
        else:
            parts.append(s)
    return "".join(parts)


def _code_view(text, segs, kinds=("code",)):
    return "".join(text[a:b] if k in kinds else " " * (b - a) for k, a, b in segs)


def _check(lang, nat, alt, parser=True):
    if alt is None or alt == nat:
        return None
    if parser and not (bash_ok(alt) if lang == "Bash" else valid_source(lang, alt)):
        return None
    return alt


# ------------------------------------------------------------------------------------------------------------------------ rules
def js_quotes(nat, spec=""):
    segs = js_segments(nat); parts = []; n = 0
    for kind, a, b in segs:
        s = nat[a:b]
        if kind == "dq" and s.endswith('"') and len(s) >= 2:
            inner = s[1:-1]
            if "\n" in inner:
                return None
            inner = inner.replace('\\"', '"')
            inner = re.sub(r"(?<!\\)'", r"\\'", inner)
            parts.append("'" + inner + "'"); n += 1
        else:
            parts.append(s)
    return _check("JavaScript", nat, "".join(parts)) if n else None


_ASI_HAZARD = re.compile(r"^\s*(\(|\[|`|\+\+|--|\+|-|/)")


def js_semicolons(nat, spec=""):
    segs = js_segments(nat); kind_at = {}
    for k, a, b in segs:
        for j in range(a, b): kind_at[j] = k
    lines = nat.split("\n"); offs = [0]
    for ln in lines: offs.append(offs[-1] + len(ln) + 1)
    removals = set(); stack = []
    for li, ln in enumerate(lines):
        base = offs[li]; j = 0
        while j < len(ln):
            ch = ln[j]
            if kind_at.get(base + j) != "code": j += 1; continue
            if ch in "([{": stack.append(ch)
            elif ch in ")]}":
                if stack: stack.pop()
            elif ch == ";":
                if stack and stack[-1] != "{": j += 1; continue                # inside ( ) or [ ]: for(;;) etc.
                k = j
                while k + 1 < len(ln) and ln[k + 1] == ";" and kind_at.get(base + k + 1) == "code": k += 1
                rest = ln[k + 1:]; rest_code = "".join(c if kind_at.get(base + k + 1 + i) == "code" else " " for i, c in enumerate(rest))
                if rest_code.strip() in ("", "}"):
                    removals.update(range(base + j, base + k + 1))
                else:
                    return None                                                  # two statements on one line
                j = k
            j += 1
    if not removals:
        return None
    for li, ln in enumerate(lines):                                            # ASI hazard after a removed terminator
        base = offs[li]
        if any(base <= p < base + len(ln) for p in removals):
            for nl in lines[li + 1:]:
                s = nl.strip()
                if not s: continue
                if s.startswith("//") or s.startswith("/*"): break
                if _ASI_HAZARD.match(nl): return None
                break
    alt = "".join(c for i, c in enumerate(nat) if i not in removals)
    return _check("JavaScript", nat, alt)


def js_var(nat, spec=""):
    segs = js_segments(nat); cnt = 0
    def fn(s):
        nonlocal cnt
        cnt += len(re.findall(r"(?<![\w$.])(?:const|let)(?=\s)", s))
        return re.sub(r"(?<![\w$.])(?:const|let)(?=\s)", "var", s)
    alt = _rewrite_js(nat, segs, fn)
    return _check("JavaScript", nat, alt) if cnt else None


def js_strict_eq(nat, spec=""):
    segs = js_segments(nat); cnt = 0
    def fn(s):
        nonlocal cnt
        cnt += s.count("===") + s.count("!==")
        return s.replace("===", "==").replace("!==", "!=")
    alt = _rewrite_js(nat, segs, fn)
    return _check("JavaScript", nat, alt) if cnt else None


_CAMEL = re.compile(r"(?<![\w$])[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+(?![\w$])")
_JS_GLOBALS = {"parseInt", "parseFloat", "setTimeout", "setInterval", "clearTimeout", "clearInterval", "encodeURIComponent", "decodeURIComponent",
               "encodeURI", "decodeURI", "isNaN", "isFinite", "requestAnimationFrame", "structuredClone", "queueMicrotask", "globalThis"}
_JS_MEMBERS = {"toLowerCase", "toUpperCase", "isArray", "charCodeAt", "isInteger", "indexOf", "isNaN", "padStart", "padEnd", "forEach", "hasOwnProperty",
               "isFinite", "toString", "toFixed", "getDate", "getMonth", "getTime", "getFullYear", "fromCharCode", "charAt", "startsWith", "lastIndexOf",
               "getDay", "codePointAt", "endsWith", "setHours", "localeCompare", "isSafeInteger", "toISOString", "getUTCDate", "findIndex", "getUTCMonth",
               "getUTCDay", "getUTCFullYear", "trimEnd", "trimStart", "fromCodePoint", "byteLength", "toPrecision", "toLocaleString", "toLocaleDateString",
               "toLocaleTimeString", "getHours", "getMinutes", "getSeconds", "getMilliseconds", "setDate", "setMonth", "setFullYear", "setMinutes",
               "setSeconds", "getTimezoneOffset", "toDateString", "toJSON", "valueOf", "reduceRight", "flatMap", "findLast", "findLastIndex", "copyWithin",
               "toSorted", "toReversed", "getOwnPropertyNames", "defineProperty", "isFrozen", "isExtensible", "fromEntries", "groupBy", "isPrototypeOf",
               "propertyIsEnumerable", "toLocaleUpperCase", "toLocaleLowerCase", "matchAll", "replaceAll", "substr", "normalize", "isWellFormed",
               "toWellFormed", "getPrototypeOf", "setPrototypeOf", "isSealed", "preventExtensions", "getOwnPropertySymbols", "getOwnPropertyDescriptor",
               "getOwnPropertyDescriptors", "MAX_SAFE_INTEGER", "isView", "parseInt", "parseFloat", "isSubsetOf", "isSupersetOf", "isDisjointFrom",
               "symmetricDifference", "allSettled", "withResolvers", "readFileSync", "writeFileSync", "existsSync", "readdirSync", "mkdirSync", "appendFileSync",
               "unlinkSync", "statSync", "toUTCString", "toTimeString", "getUTCHours", "getUTCMinutes", "getUTCSeconds", "setUTCDate", "setUTCHours",
               "setUTCMinutes", "setUTCMonth", "setUTCFullYear", "setTime", "toExponential", "localeCompare", "getElementById", "querySelector",
               "querySelectorAll", "addEventListener", "textContent", "innerHTML", "createElement", "appendChild", "setAttribute", "getAttribute",
               "removeChild", "classList", "getContext", "fillRect", "fillText", "clearRect", "lineTo", "moveTo", "beginPath", "requestAnimationFrame",
               "readLine", "stdIn", "stdOut", "toArray", "asyncIterator", "byteOffset", "isConcatSpreadable", "hasInstance", "getInt32", "setInt32",
               "getUint8", "setUint8", "getFloat64", "setFloat64", "getUint16", "setUint16", "getUint32", "setUint32", "getInt16", "setInt16",
               "getInt8", "setInt8", "getFloat32", "setFloat32", "getBigInt64", "setBigInt64", "getBigUint64", "setBigUint64", "randomUUID",
               "getRandomValues", "performance", "hrtime", "nextTick", "argv", "exitCode", "stdout", "stderr", "stdin", "cwd", "isTTY", "columns"}


def snake(name):
    s = re.sub(r"([A-Z]+)([A-Z][a-z0-9])", r"\1_\2", name); s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s); return s.lower()


def js_camel_snake(nat, spec=""):
    from src.sandbox.style_translation.code_free import _task_code_idents
    segs = js_segments(nat); code = _code_view(nat, segs)
    for k, a, b in segs:
        if k == "template":
            t = nat[a:b]; masked = list(" " * len(t))
            for m in _TPL_EXPR.finditer(t):
                masked[m.start(1):m.end(1)] = m.group(1)
            code = code[:a] + "".join(masked) + code[b:]
    pinned = _task_code_idents(spec or "")
    bound = set()
    for m in re.finditer(r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)", code):
        bound.add(m.group(1)); bound.update(re.findall(r"[A-Za-z_$][\w$]*", re.sub(r"=[^,]*", "", m.group(2))))
    for m in re.finditer(r"\bfunction\s*\(([^)]*)\)", code):
        bound.update(re.findall(r"[A-Za-z_$][\w$]*", re.sub(r"=[^,]*", "", m.group(1))))
    for m in re.finditer(r"\(([^()]*)\)\s*=>", code):
        bound.update(re.findall(r"[A-Za-z_$][\w$]*", re.sub(r"=[^,]*", "", m.group(1))))
    for m in re.finditer(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*=>", code):
        bound.add(m.group(1))
    for m in re.finditer(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*(?:\s*,\s*[A-Za-z_$][\w$]*)*)", code):
        bound.update(re.findall(r"[A-Za-z_$][\w$]*", m.group(1)))
    for m in re.finditer(r"\b(?:const|let|var)\s*[\[{]([^=]*?)[\]}]\s*=", code):                      # destructuring: the names are variables
        bound.update(re.findall(r"[A-Za-z_$][\w$]*", re.sub(r":\s*[A-Za-z_$][\w$]*", lambda q: q.group(0), m.group(1))))
    for m in re.finditer(r"\bcatch\s*\(\s*([A-Za-z_$][\w$]*)", code):
        bound.add(m.group(1))
    keys = set(re.findall(r"[{,]\s*([A-Za-z_$][\w$]*)\s*:(?!:)", code))                              # keys of object literals: data, not variables
    for m in re.finditer(r"\bclass\s+([A-Za-z_$][\w$]*)", code):
        bound.discard(m.group(1))
    targets = {x for x in bound if _CAMEL.fullmatch(x) and x not in pinned and x not in keys}
    if not targets:
        return None
    mapping = {t: snake(t) for t in targets}
    if len(set(mapping.values())) < len(mapping) or any(re.search(r"(?<![\w$])" + re.escape(v) + r"(?![\w$])", code) for v in mapping.values()):
        return None                                                            # collision with an existing name
    rx = re.compile(r"(?<![\w$])(" + "|".join(map(re.escape, sorted(targets, key=len, reverse=True))) + r")(?![\w$])")
    fn = lambda s: rx.sub(lambda m: mapping[m.group(1)], s)
    alt = _rewrite_js(nat, segs, fn, comments=True)
    asegs = js_segments(alt); acode = _code_view(alt, asegs)
    for k, a, b in asegs:
        if k == "template":
            t = alt[a:b]; masked = list(" " * len(t))
            for m in _TPL_EXPR.finditer(t): masked[m.start(1):m.end(1)] = m.group(1)
            acode = acode[:a] + "".join(masked) + acode[b:]
    for m in _CAMEL.finditer(acode):                                           # leftovers the rule cannot classify -> decline
        w = m.group(0); pre = acode[:m.start()].rstrip()
        if w in pinned or w in _JS_GLOBALS or w in _JS_MEMBERS or w in keys:
            continue
        if pre.endswith(".") and w in keys:
            continue
        return None
    return _check("JavaScript", nat, alt)


_PHP_LITERAL_WORDS = {"return", "echo", "print", "yield", "case", "throw", "as", "in", "and", "or", "xor", "new", "else", "do", "fn", "use", "default", "match"}


def php_array(nat, spec=""):
    if "<<<" in nat:
        return None
    segs = _generic_segments(nat, line_comment="//", block=("/*", "*/"), quotes=("'", '"'), hash_comments=True)
    kind_at = {}
    for k, a, b in segs:
        for j in range(a, b): kind_at[j] = k
    stack = []; pairs = {}
    for j, ch in enumerate(nat):
        if kind_at.get(j) != "code": continue
        if ch == "[":
            k = j - 1
            while k >= 0 and nat[k] in " \t\n": k -= 1
            prev = nat[k] if k >= 0 else ""
            w = re.search(r"([A-Za-z_]\w*)$", nat[max(0, k - 20):k + 1]) if k >= 0 else None
            literal = not (prev.isalnum() or prev in "_$)]\"'") or (w is not None and w.group(1) in _PHP_LITERAL_WORDS and not nat[k - len(w.group(1)):k - len(w.group(1)) + 1] == "$")
            stack.append((j, literal))
        elif ch == "]":
            if not stack: return None
            o, lit = stack.pop()
            if lit: pairs[o] = j
    if stack or not pairs:
        return None
    edits = {}
    for o, c in pairs.items():
        after = nat[c + 1:c + 4].lstrip()
        before = nat[max(0, o - 6):o].rstrip()
        target = (after.startswith("=") and not after.startswith("==") and not after.startswith("=>")) or before.endswith(" as")   # [$a, $b] = f() / as [$k, $v]
        if target:                                                             # list destructuring is not an array literal: unchanged (as the reviewed pairs)
            continue
        edits[o] = "array("; edits[c] = ")"
    alt = "".join(edits.get(i, ch) for i, ch in enumerate(nat))
    return _check("PHP", nat, alt)


def r_assignment(nat, spec=""):
    segs = _generic_segments(nat, line_comment="#", block=None, quotes=("'", '"'))
    code = _code_view(nat, segs)
    if "<<-" in code or "->" in code:
        return None
    kind_at = {}
    for k, a, b in segs:
        for j in range(a, b): kind_at[j] = k
    stack = []; edits = set(); i = 0; n = len(nat)
    while i < n:
        if kind_at.get(i) != "code": i += 1; continue
        ch = nat[i]
        if ch in "({[": stack.append(ch)
        elif ch in ")}]":
            if stack: stack.pop()
        elif ch == "<" and nat[i + 1:i + 2] == "-":
            if stack and stack[-1] == "(":
                return None                                                    # assignment inside call arguments / condition
            edits.add(i); i += 2; continue
        i += 1
    if not edits:
        return None
    alt = "".join(("=" if i in edits else ("" if i - 1 in edits else ch)) for i, ch in enumerate(nat))
    return _check("R", nat, alt, parser=False)


def c_comment_style(nat, spec=""):
    segs = js_segments(nat); parts = []; cnt = 0
    for kind, a, b in segs:
        s = nat[a:b]
        if kind == "lcomment":
            body = s[2:]
            if "*/" in body: return None
            chunks = re.split(r"(?<=\S)( +)// ", body)                          # a second `// ` on the same line -> separate block
            if len(chunks) > 1:
                out = ["/*" + chunks[0].rstrip() + " */"]
                for i in range(1, len(chunks), 2):
                    out.append(chunks[i] + "/* " + chunks[i + 1].strip() + " */")
                parts.append("".join(out))
            else:
                parts.append("/*" + body.rstrip() + " */")
            cnt += 1
        else:
            parts.append(s)
    return _check("JavaScript", nat, "".join(parts)) if cnt else None


_BASH_UNSAFE = [r"=~", r"&&", r"\|\|", r"(?<![\w-])[<>](?!=)", r"[()]", r"(?:^|\s)-(?:nt|ot|ef)\s"]
_BASH_ARITH = {"-eq", "-ne", "-lt", "-le", "-gt", "-ge"}


def _bash_strip(inner):
    t = re.sub(r"\$\(\((?:[^()]|\([^()]*\))*\)\)", "A", inner); t = re.sub(r"\$\((?:[^()]|\([^()]*\))*\)", "C", t); t = re.sub(r"\$\{(?:[^{}]|\{[^{}]*\})*\}", "$V", t)
    return re.sub(r"\"(?:[^\"\\]|\\.)*\"|'[^']*'", "Q", t)


def _bash_test_convertible(inner):
    t = _bash_strip(inner)
    if any(re.search(p, t) for p in _BASH_UNSAFE):
        return False
    if re.search(r"(?:==|!=|=)\s*\S*[*?\[]", t):                            # glob pattern on the right-hand side
        return False
    return True


def _bash_quote_vars(inner):
    """Bare $var / ${...} operands get double quotes (word splitting differs in [ ]); everything else unchanged."""
    out = []; i = 0
    for tok in re.split(r"(\s+)", inner):
        if re.fullmatch(r"\$(?:\{[^}]*\}|[A-Za-z_][\w]*|[#?$!@*0-9])", tok) and tok not in ("$#", "$?", "$$", "$!"):
            out.append('"' + tok + '"')
        else:
            out.append(tok)
    return "".join(out)


def bash_test(nat, spec=""):
    segs = _generic_segments(nat, line_comment="#", block=None, quotes=("'", '"'))
    kind_at = {}
    for k, a, b in segs:
        for j in range(a, b): kind_at[j] = k
    edits = {}; cnt = 0
    for m in re.finditer(r"\[\[ (.*?) \]\]", nat):
        if kind_at.get(m.start()) != "code": continue
        if "\n" in m.group(1): return None
        if _bash_test_convertible(m.group(1)):
            inner = m.group(1); arith = any(t in _BASH_ARITH for t in _bash_strip(inner).split())
            edits[m.start()] = "[ " + (inner if arith else _bash_quote_vars(inner)) + " ]"; cnt += 1
            for j in range(m.start() + 1, m.end()): edits[j] = ""
    if not cnt:
        return None
    alt = "".join(edits.get(i, ch) for i, ch in enumerate(nat))
    return _check("Bash", nat, alt)


def sql_bool_case(nat, spec=""):
    segs = _generic_segments(nat, line_comment="--", block=("/*", "*/"), quotes=("'", '"'), sql_quotes=True); cnt = 0
    def fn(s):
        nonlocal cnt
        cnt += len(re.findall(r"\b(?:true|false)\b", s))
        return re.sub(r"\b(true|false)\b", lambda m: m.group(1).upper(), s)
    alt = "".join(fn(nat[a:b]) if k == "code" else nat[a:b] for k, a, b in segs)
    return _check("SQL", nat, alt, parser=False) if cnt else None


HEX_SET = {255, 256, 128, 64, 4096, 65535, 1024, 16, 32}
_JS_INT = re.compile(r"(?<![\w.$])(\d+)(?![\w.$])")


def _hex_rule(nat, pred):
    segs = js_segments(nat); cnt = 0
    def fn(s):
        nonlocal cnt
        def f(m):
            nonlocal cnt
            v = int(m.group(1))
            if pred(v) and not m.group(1).startswith("0"):
                cnt += 1; return "0x" + format(v, "X")
            return m.group(0)
        return _JS_INT.sub(f, s)
    alt = _rewrite_js(nat, segs, fn)
    return _check("JavaScript", nat, alt) if cnt else None


def hex_constants(nat, spec=""):
    return _hex_rule(nat, lambda v: v in HEX_SET)


def hex_constants_pow2(nat, spec=""):
    return _hex_rule(nat, lambda v: (v >= 16 and v & (v - 1) == 0) or (v >= 15 and (v + 1) & v == 0))


RULES = {"js_quotes": js_quotes, "js_semicolons": js_semicolons, "js_var": js_var, "js_strict_eq": js_strict_eq, "js_camel_snake": js_camel_snake,
         "php_array": php_array, "r_assignment": r_assignment, "c_comment_style": c_comment_style, "bash_test": bash_test,
         "sql_bool_case": sql_bool_case, "hex_constants": hex_constants}
