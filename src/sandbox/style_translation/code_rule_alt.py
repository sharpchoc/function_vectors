"""Exact rule-based alternative twins for families whose rewrite is a mechanical rename / literal change (2026-09-21, padding retry).
LLM rewrites of these families kept failing review on the REWRITE side (one identifier left unconverted, string keys renamed, a literal
missed). Each rule applies the family's stated rewrite to EVERY covered occurrence and to nothing else, and verifies itself (AST check);
it returns None when it cannot guarantee that, and the caller rejects the natural twin. Only used when finish_pair(rule_alt=True).
    py_snake_camel : multi-word snake_case names BOUND in the file (functions, parameters, variables, self-attributes, methods) -> camelCase
    num_separators : every decimal integer literal of >= 4 digits -> underscore thousands separators
    py_private     : underscore-prefixed private members of classes -> bare names at definition and every use (redefined 2026-09-22)
    hex_constants  : JavaScript integer literals from the family's constant set -> hexadecimal (0xFF style)"""
import ast, io, re, tokenize

SNAKE = re.compile(r"^(_*)([a-z][a-z0-9]*)((?:_[a-z0-9]+)+)(_*)$")


def camel(name):
    m = SNAKE.match(name)
    if not m:
        return name
    lead, first, rest, trail = m.groups()
    return lead + first + "".join(p[:1].upper() + p[1:] for p in rest.split("_") if p) + trail


def _bound_names(tree):
    """(plain names bound in the file, attribute/method names bound in the file); import-bound names are excluded."""
    names, attrs, imported = set(), set(), set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            imported |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(n.name)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names |= set(n.names)
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name) and n.value.id in ("self", "cls"):
            attrs.add(n.attr)
    for c in ast.walk(tree):
        if isinstance(c, ast.ClassDef):
            for b in c.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    attrs.add(b.name)
                elif isinstance(b, (ast.Assign, ast.AnnAssign)):
                    for t in (b.targets if isinstance(b, ast.Assign) else [b.target]):
                        if isinstance(t, ast.Name):
                            attrs.add(t.id)
    keep = lambda s: {x for x in s if SNAKE.match(x) and not (x.startswith("__") and x.endswith("__"))}
    return keep(names) - imported, keep(attrs)


class _Rename(ast.NodeTransformer):
    def __init__(self, names, attrs, own_callables):
        self.n, self.a, self.own = names, attrs, own_callables
    def visit_Name(self, node):
        if node.id in self.n: node.id = camel(node.id)
        return node
    def visit_arg(self, node):
        self.generic_visit(node)
        if node.arg in self.n: node.arg = camel(node.arg)
        return node
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name in self.n or node.name in self.a: node.name = camel(node.name)
        return node
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr in self.a: node.attr = camel(node.attr)
        return node
    def visit_keyword(self, node):
        self.generic_visit(node)
        if node.arg and node.arg in self.n and getattr(node, "_own", False): node.arg = camel(node.arg)
        return node
    def visit_ExceptHandler(self, node):
        self.generic_visit(node)
        if node.name and node.name in self.n: node.name = camel(node.name)
        return node
    def visit_Global(self, node):
        node.names = [camel(x) if x in self.n else x for x in node.names]; return node
    visit_Nonlocal = visit_Global


def snake_to_camel(nat, spec=""):
    """Function names that the task statement dictates (`name(` in the spec) keep their spelling, as in the reviewed pairs; everything else bound in the file is renamed."""
    try:
        tree = ast.parse(nat)
    except SyntaxError:
        return None
    names, attrs = _bound_names(tree)
    from src.sandbox.style_translation.code_free import _task_code_idents
    fixed = _task_code_idents(spec or ""); names -= fixed; attrs -= fixed           # identifiers the task text pins (same definition as code_free.harmonise_pinned)
    if not names and not attrs:
        return None
    own = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    skip_kw = set()                                                # keyword arguments of calls to functions NOT defined in this file keep their name
    for c in ast.walk(tree):
        if isinstance(c, ast.Call):
            f = c.func; fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            for k in c.keywords:
                k._own = fname in own
                if k.arg and k.arg in names and not k._own:
                    skip_kw.add((k.lineno, k.col_offset))
    if any(camel(x) in names | attrs or camel(x) in own for x in names | attrs):
        return None                                                # rename collision
    out = []; prev = None
    toks = list(tokenize.generate_tokens(io.StringIO(nat).readline))
    for t in toks:
        s = t.string
        if t.type == tokenize.NAME:
            after_dot = prev is not None and prev.type == tokenize.OP and prev.string == "."
            if after_dot:
                if s in attrs: s = camel(s)
            elif s in names and (t.start[0], t.start[1]) not in skip_kw:
                s = camel(s)
        elif t.type == tokenize.STRING and re.match(r"^[rRbB]*[fF][rRbB]*['\"]", s):
            s = _fstring(s, names, attrs)
        out.append((t, s))
        if t.type not in (tokenize.NL, tokenize.COMMENT, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT): prev = t
    lines = nat.splitlines(keepends=True); res = []; row, col = 1, 0          # rebuild the text byte-for-byte around the changed tokens
    for t, s in out:
        if t.type in (tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT) or (t.type in (tokenize.NEWLINE, tokenize.NL) and t.string == ""): continue
        (r0, c0), (r1, c1) = t.start, t.end
        while row < r0:
            res.append(lines[row - 1][col:]); row += 1; col = 0
        res.append(lines[row - 1][col:c0]); res.append(s); row, col = r1, c1
    while row <= len(lines):
        res.append(lines[row - 1][col:]); row += 1; col = 0
    alt = "".join(res)
    try:
        want = ast.dump(_Rename(names, attrs, own).visit(ast.parse(nat))); got = ast.dump(ast.parse(alt))
    except SyntaxError:
        return None
    return alt if want == got and alt != nat else None


def _fstring(tok, names, attrs):
    def fix(m):
        expr = m.group(1)
        def ren(mm):
            w = mm.group(2)
            if mm.group(1) == ".": return mm.group(0) if w not in attrs else "." + camel(w)
            return mm.group(1) + (camel(w) if w in names else w)
        return "{" + re.sub(r"(^|[^\w.]|\.)([A-Za-z_]\w*)", ren, expr) + "}"
    return re.sub(r"(?<!\{)\{([^{}]+)\}(?!\})", fix, tok)


def num_separators(nat, spec=""):
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(nat).readline)); ast.parse(nat)
    except (SyntaxError, tokenize.TokenError):
        return None
    lines = nat.splitlines(keepends=True); edits = []
    for t in toks:
        if t.type == tokenize.NUMBER and re.fullmatch(r"[1-9][0-9]{3,}", t.string):
            edits.append((t.start[0], t.start[1], t.end[1], f"{int(t.string):,}".replace(",", "_")))
        elif t.type == tokenize.STRING and re.match(r"^[rRbB]*[fF][rRbB]*['\"]", t.string) and t.start[0] == t.end[0]:   # literals inside f-string expressions
            new = re.sub(r"(?<!\{)\{([^{}]+)\}(?!\})", lambda m: "{" + re.sub(r"(?<![\w.:])([1-9][0-9]{3,})(?![\w.])", lambda q: f"{int(q.group(1)):,}".replace(",", "_"), m.group(1).split(":")[0] if False else m.group(1)) + "}", t.string)
            if new != t.string:
                edits.append((t.start[0], t.start[1], t.end[1], new))
    if not edits:
        return None
    for r, c0, c1, s in sorted(edits, reverse=True):
        lines[r - 1] = lines[r - 1][:c0] + s + lines[r - 1][c1:]
    alt = "".join(lines)
    return alt if ast.dump(ast.parse(alt)) == ast.dump(ast.parse(nat)) else None



def py_private(nat, spec=""):
    """py_private (redefined 2026-09-22): every single-underscore private member bound in a class (self._x stores, def _m, class-level _c)
    loses its leading underscore at the definition and at EVERY use (dotted, f-strings included). Declines on a rename collision
    (the bare name already exists as an attribute/name in the file), dynamic attribute access, or when the AST check fails."""
    try:
        tree = ast.parse(nat)
    except SyntaxError:
        return None
    priv = set()
    for c in ast.walk(tree):
        if isinstance(c, ast.ClassDef):
            for n in ast.walk(c):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n in c.body: priv.add(n.name)
                elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name) and n.value.id in ("self", "cls"): priv.add(n.attr)
                elif isinstance(n, (ast.Assign, ast.AnnAssign)) and n in c.body:
                    priv |= {t.id for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}
    priv = {x for x in priv if re.fullmatch(r"_[A-Za-z0-9][A-Za-z0-9_]*", x) and not x.endswith("__")}
    fixed = set(re.findall(r"[A-Za-z_]\w*", spec or "")); priv -= fixed
    if not priv:
        return None
    bare = {x[1:] for x in priv}
    taken = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}          # an attribute of that bare name already exists
    for c in ast.walk(tree):                                                            # or a method / class-level name of that bare name
        if isinstance(c, ast.ClassDef):
            taken |= {b.name for b in c.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))}
            taken |= {t.id for b in c.body if isinstance(b, (ast.Assign, ast.AnnAssign)) for t in (b.targets if isinstance(b, ast.Assign) else [b.target]) if isinstance(t, ast.Name)}
    if bare & taken or re.search(r"getattr|setattr|hasattr|__dict__|vars\(", nat):
        return None
    lines = nat.splitlines(keepends=True); off = [0]
    for ln in lines: off.append(off[-1] + len(ln))
    pos = lambda rc: off[rc[0] - 1] + rc[1]
    edits = []                                                     # absolute-offset edits (multi-line f-strings included)
    rx = r"(?<![\w])(" + "|".join(map(re.escape, sorted(priv, key=len, reverse=True))) + r")(?![\w])"
    for t in tokenize.generate_tokens(io.StringIO(nat).readline):
        if t.type == tokenize.NAME and t.string in priv:
            edits.append((pos(t.start), pos(t.end), t.string[1:]))
        elif t.type == tokenize.STRING and re.match(r"^[rRbB]*[fF][rRbB]*['\"]", t.string):
            new = re.sub(r"(?<!\{)\{([^{}]+)\}(?!\})", lambda m: "{" + re.sub(rx, lambda q: q.group(1)[1:], m.group(1)) + "}", t.string, flags=re.S)
            if new != t.string: edits.append((pos(t.start), pos(t.end), new))
    alt = nat
    for s0, s1, x in sorted(edits, reverse=True):
        alt = alt[:s0] + x + alt[s1:]
    class R(ast.NodeTransformer):
        def visit_Attribute(self, n):
            self.generic_visit(n)
            if n.attr in priv: n.attr = n.attr[1:]
            return n
        def visit_Name(self, n):
            if n.id in priv: n.id = n.id[1:]
            return n
        def visit_FunctionDef(self, n):
            self.generic_visit(n)
            if n.name in priv: n.name = n.name[1:]
            return n
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_arg(self, n):                                    # a parameter that shares a private name is renamed with it
            self.generic_visit(n)
            if n.arg in priv: n.arg = n.arg[1:]
            return n
        def visit_keyword(self, n):
            self.generic_visit(n)
            if n.arg in priv: n.arg = n.arg[1:]
            return n
    try:
        ok = ast.dump(R().visit(ast.parse(nat))) == ast.dump(ast.parse(alt))
    except SyntaxError:
        return None
    return alt if ok and alt != nat else None


HEX_SET = {255, 256, 128, 64, 4096, 65535, 1024, 16, 32}
_JS_MASK = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`", re.S)
_JS_INT = re.compile(r"(?<![\w.$])(\d+)(?![\w.$])")


def hex_constants(nat, spec=""):
    out = []; pos = 0; n = 0
    def conv(seg):
        nonlocal n
        def f(m):
            nonlocal n
            v = int(m.group(1))
            if v in HEX_SET and not m.group(1).startswith("0"):
                n += 1; return "0x" + format(v, "X")
            return m.group(0)
        return _JS_INT.sub(f, seg)
    for m in _JS_MASK.finditer(nat):
        out.append(conv(nat[pos:m.start()])); out.append(m.group(0)); pos = m.end()
    out.append(conv(nat[pos:]))
    return "".join(out) if n else None


# hex_constants is NOT enabled: 17 of 44 reviewer-approved pairs also convert constants outside the family set, so the set-only rule does not reproduce the accepted data
RULE_ALT = {"py_snake_camel": snake_to_camel, "num_separators": num_separators, "py_private": py_private}
ALWAYS_RULE = {"py_private"}                     # families whose alternative twin is ALWAYS derived by rule (never an LLM rewrite)
# regen8 (2026-09-22): validated rule modules for further families (tmp/rules8/report_*.md). A rule that returns None falls back to the
# LLM rewrite in finish_pair (recorded as alt_rule False) unless the family is in ALWAYS_RULE.
for _mod in ("code_rules_a", "code_rules_b", "code_rules_c"):
    try:
        _m = __import__(f"src.sandbox.style_translation.{_mod}", fromlist=["RULES"])
        for _f, _fn in getattr(_m, "RULES", {}).items():
            if _f in getattr(_m, "DISABLED", ()):
                continue
            RULE_ALT.setdefault(_f, _fn)
    except ImportError:
        pass                     # families whose alternative twin is ALWAYS derived by rule (never an LLM rewrite)
