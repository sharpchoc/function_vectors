"""Exact rule-based alternative twins, GROUP B (Python naming / literal / comment / whitespace families; 2026-09-22, regen8 Part 0).
Each rule derives the alternative twin from the natural twin by applying the family's rewrite to EVERY covered occurrence and to nothing
else; it returns None when it cannot guarantee that (unparsable, rename collision, ambiguous literal). Naming rules verify themselves by
transforming the AST the same way and comparing `ast.dump`; literal / comment / whitespace rules verify AST equality (or the intended
value change only). Text outside the changed tokens is byte-identical. Pinned identifiers (`code_free._task_code_idents(spec)`) are kept.

    py_const_naming   UPPER_SNAKE names bound in the file (>= 2 segments) -> camelCase at definition and every use
    py_class_naming   PascalCase classes DEFINED in the file -> snake_case at definition and every use (strings untouched)
    py_loop_vars      single-letter for / comprehension targets -> descriptive word (LOOP_WORDS), every use in the same scope
    py_abbrev         abbreviated identifier segments (ABBREV) -> full words, at every bound use (names, parameters, self attributes)
    py_quotes         single-quoted string literals ('..', '''..''') -> double quotes, unless the content contains a double quote
    docstring_quotes  docstring delimiters \"\"\" -> ''' (docstrings only; declines if a docstring contains ''' or ends with ')
    float_literals    0.5 -> .5, 1.0 -> 1., 2.50 -> 2.5, 0.0 -> 0. (every decimal float literal with digits on both sides)
    trailing_commas   the trailing comma before a closing bracket is removed (1-tuples and subscript tuples keep it)
    comment_case      the first word of every line comment: ALLCAPS word -> lowercase, else first letter -> lowercase
    comma_space / operator_spaces / blank_lines: delegated to code_whitespace_alt (already rule-based, 200/200 identical)
"""
import ast
import io
import re
import tokenize

from src.sandbox.style_translation import code_whitespace_alt as WS

UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)*$")          # MAX_RETRIES and single-word PROMPT (>= 2 chars)
PASCAL = re.compile(r"^[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*$")
LOOP_WORDS = {"i": "index", "j": "column", "k": "key", "n": "number", "c": "char", "x": "value", "y": "other", "m": "member", "p": "position",
              "q": "item", "r": "row", "s": "string", "t": "token", "v": "val", "w": "word", "z": "depth", "a": "first", "b": "second",
              "d": "digit", "e": "entry", "f": "field", "g": "group", "h": "height", "l": "line", "o": "obj", "u": "unit"}
ABBREV = {"cfg": "configuration", "idx": "index", "buf": "buffer", "num": "number", "val": "value", "tmp": "temporary", "msg": "message",
          "ctx": "context", "res": "result", "err": "error", "cnt": "count", "pos": "position", "prev": "previous", "curr": "current",
          "elem": "element", "param": "parameter"}


# ----------------------------------------------------------------------------------------------------------------- shared helpers
def _pinned(spec):
    from src.sandbox.style_translation.code_free import _task_code_idents
    return _task_code_idents(spec or "")


def _offsets(src):
    """absolute character offset of (row, col) where col is a BYTE offset (ast) or a character offset (tokenize)."""
    lines = src.splitlines(keepends=True); starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln))
    def by_char(row, col):
        return starts[row - 1] + col
    def by_byte(row, col):
        ln = lines[row - 1] if row - 1 < len(lines) else ""
        return starts[row - 1] + len(ln.encode("utf-8")[:col].decode("utf-8", "ignore"))
    return by_char, by_byte


def _apply(src, edits):
    out = src
    for s, e, rep in sorted(set(edits), reverse=True):
        out = out[:s] + rep + out[e:]
    return out


def _tokens(src):
    try:
        return list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None


def _fstring_rename(tok, mapping):
    """rename whole-word identifiers inside the {...} expressions of an f-string token (attribute names after '.' excluded)."""
    rx = r"(?<![\w.])(" + "|".join(map(re.escape, sorted(mapping, key=len, reverse=True))) + r")(?![\w])"
    return re.sub(r"(?<!\{)\{([^{}]+)\}(?!\})", lambda m: "{" + re.sub(rx, lambda q: mapping[q.group(1)], m.group(1)) + "}", tok, flags=re.S)


def _rename_by_tokens(src, mapping, attr_mapping=None, in_scope=None):
    """Rename NAME tokens: plain names by `mapping` (attribute names after '.' by `attr_mapping`); f-strings handled; `in_scope(row)`
    optionally restricts plain-name renames to certain rows. Returns the new text or None if the source does not tokenize."""
    toks = _tokens(src)
    if toks is None:
        return None
    by_char, _ = _offsets(src); edits = []; prev = None
    for t in toks:
        if t.type == tokenize.NAME:
            after_dot = prev is not None and prev.type == tokenize.OP and prev.string == "."
            if after_dot:
                if attr_mapping and t.string in attr_mapping:
                    edits.append((by_char(*t.start), by_char(*t.end), attr_mapping[t.string]))
            elif t.string in mapping and (in_scope is None or in_scope(t.start[0])):
                edits.append((by_char(*t.start), by_char(*t.end), mapping[t.string]))
        elif t.type == tokenize.STRING and re.match(r"^[rRbB]*[fF][rRbB]*['\"]", t.string) and mapping:
            new = _fstring_rename(t.string, mapping)
            if new != t.string:
                edits.append((by_char(*t.start), by_char(*t.end), new))
        if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            prev = t
    return _apply(src, edits)


class _AstRename(ast.NodeTransformer):
    def __init__(self, names, attrs=None, classes=None):
        self.n, self.a, self.c = names, attrs or {}, classes or {}
    def visit_Name(self, node):
        if node.id in self.n: node.id = self.n[node.id]
        elif node.id in self.c: node.id = self.c[node.id]
        return node
    def visit_arg(self, node):
        self.generic_visit(node)
        if node.arg in self.n: node.arg = self.n[node.arg]
        return node
    def visit_keyword(self, node):
        self.generic_visit(node)
        if node.arg in self.n: node.arg = self.n[node.arg]
        return node
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr in self.a: node.attr = self.a[node.attr]
        return node
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name in self.n: node.name = self.n[node.name]
        elif node.name in self.a: node.name = self.a[node.name]
        return node
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if node.name in self.c: node.name = self.c[node.name]
        return node
    def visit_ExceptHandler(self, node):
        self.generic_visit(node)
        if node.name in self.n: node.name = self.n[node.name]
        return node
    def visit_Global(self, node):
        node.names = [self.n.get(x, x) for x in node.names]; return node
    visit_Nonlocal = visit_Global


def _ast_equal(src, alt, transformer):
    try:
        return ast.dump(transformer.visit(ast.parse(src))) == ast.dump(ast.parse(alt))
    except SyntaxError:
        return False


def _all_idents(tree):
    s = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name): s.add(n.id)
        elif isinstance(n, ast.arg): s.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): s.add(n.name)
        elif isinstance(n, ast.Attribute): s.add(n.attr)
        elif isinstance(n, ast.alias): s.add((n.asname or n.name).split(".")[0])
    return s


def _imported(tree):
    return {(a.asname or a.name).split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}


# ----------------------------------------------------------------------------------------------------------------- naming families
def _camel_const(name):
    parts = name.lower().split("_")
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def py_const_naming(nat, spec=""):
    try:
        tree = ast.parse(nat)
    except SyntaxError:
        return None
    bound = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} - _imported(tree)
    names = {x for x in bound if UPPER_SNAKE.match(x)} - _pinned(spec)
    if not names:
        return None
    mapping = {x: _camel_const(x) for x in names}
    if set(mapping.values()) & _all_idents(tree):
        return None
    alt = _rename_by_tokens(nat, mapping)
    return alt if alt is not None and alt != nat and _ast_equal(nat, alt, _AstRename(mapping)) else None


def _snake_class(name):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def py_class_naming(nat, spec=""):
    try:
        tree = ast.parse(nat)
    except SyntaxError:
        return None
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and PASCAL.match(n.name)} - _pinned(spec)
    if not classes:
        return None
    mapping = {c: _snake_class(c) for c in classes}
    if set(mapping.values()) & _all_idents(tree):
        return None
    alt = _rename_by_tokens(nat, mapping, mapping)
    return alt if alt is not None and alt != nat and _ast_equal(nat, alt, _AstRename({}, attrs=mapping, classes=mapping)) else None


def py_loop_vars(nat, spec=""):
    """Per function scope (module = top scope; comprehensions belong to their enclosing function): every single-letter name that is a
    for-loop / comprehension target somewhere in that scope is renamed at EVERY use in that scope (parameters of the same name included).
    Declines when the target word already exists anywhere in the file."""
    try:
        tree = ast.parse(nat)
    except SyntaxError:
        return None
    pinned = _pinned(spec)
    scopes = []                                                    # (node, set of letters looped in this scope)
    def targets(t):
        return [x.id for x in ast.walk(t) if isinstance(x, ast.Name)]
    def visit_scope(node):
        letters = set()
        def walk(n):
            for ch in ast.iter_child_nodes(n):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    visit_scope(ch); continue
                if isinstance(ch, (ast.For, ast.AsyncFor, ast.comprehension)):
                    letters.update(x for x in targets(ch.target) if len(x) == 1 and x.isalpha())
                walk(ch)
        walk(node); scopes.append((node, letters))
    visit_scope(tree)
    letters_all = set().union(*(l for _, l in scopes)) - pinned
    if not letters_all:
        return None
    used = _all_idents(tree)
    mapping = {l: LOOP_WORDS.get(l, l + "_value") for l in letters_all}
    if set(mapping.values()) & used:
        return None
    # rows -> renaming scope: each scope's own rows minus nested scopes' rows
    row_scope = {}
    for node, letters in sorted(scopes, key=lambda s: (getattr(s[0], "lineno", 0) or 0)):
        r0 = getattr(node, "lineno", 1) if not isinstance(node, ast.Module) else 1
        r1 = getattr(node, "end_lineno", None) or nat.count("\n") + 1
        for r in range(r0, r1 + 1):
            row_scope[r] = letters                                  # inner scopes overwrite outer rows
    class R(ast.NodeTransformer):
        def __init__(self): self.stack = []
        def _scope_letters(self, node):
            key = None if isinstance(node, ast.Module) else (node.lineno, node.col_offset, type(node).__name__)
            return next((l for n, l in scopes if (None if isinstance(n, ast.Module) else (n.lineno, n.col_offset, type(n).__name__)) == key), set())
        def generic_visit(self, node):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                self.stack.append(self._scope_letters(node)); super().generic_visit(node); self.stack.pop(); return node
            return super().generic_visit(node)
        def visit_Name(self, node):
            if self.stack and node.id in self.stack[-1] and node.id in mapping: node.id = mapping[node.id]
            return node
        def visit_arg(self, node):
            if self.stack and node.arg in self.stack[-1] and node.arg in mapping: node.arg = mapping[node.arg]
            return node
    in_scope = lambda row: True
    toks = _tokens(nat)
    if toks is None:
        return None
    by_char, _ = _offsets(nat); edits = []; prev = None
    for t in toks:
        if t.type == tokenize.NAME and t.string in mapping and t.string in row_scope.get(t.start[0], set()):
            after_dot = prev is not None and prev.type == tokenize.OP and prev.string == "."
            if not after_dot:
                edits.append((by_char(*t.start), by_char(*t.end), mapping[t.string]))
        elif t.type == tokenize.STRING and re.match(r"^[rRbB]*[fF][rRbB]*['\"]", t.string):
            sub = {l: w for l, w in mapping.items() if l in row_scope.get(t.start[0], set())}
            if sub:
                new = _fstring_rename(t.string, sub)
                if new != t.string: edits.append((by_char(*t.start), by_char(*t.end), new))
        if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            prev = t
    alt = _apply(nat, edits)
    try:
        want = ast.dump(R().visit(ast.parse(nat))); got = ast.dump(ast.parse(alt))
    except SyntaxError:
        return None
    return alt if alt != nat and want == got else None


def _expand_abbrev(name):
    def one(p):
        w = ABBREV.get(p.lower())
        if w is None:
            return p
        return w.upper() if p.isupper() else (w[:1].upper() + w[1:] if p[:1].isupper() else w)
    return "_".join(one(p) for p in name.split("_"))


def py_abbrev(nat, spec=""):
    try:
        tree = ast.parse(nat)
    except SyntaxError:
        return None
    pinned = _pinned(spec); imported = _imported(tree)
    names, attrs = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)): names.add(n.id)
        elif isinstance(n, ast.arg): names.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)): names.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name: names.add(n.name)
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name) and n.value.id in ("self", "cls"): attrs.add(n.attr)
    for c in ast.walk(tree):
        if isinstance(c, ast.ClassDef):
            attrs |= {b.name for b in c.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))}
    cov = lambda s: {x for x in s if _expand_abbrev(x) != x and x not in pinned and x not in imported}
    names, attrs = cov(names), cov(attrs)
    if not names and not attrs:
        return None
    nm = {x: _expand_abbrev(x) for x in names}; am = {x: _expand_abbrev(x) for x in attrs}
    if (set(nm.values()) | set(am.values())) & _all_idents(tree):
        return None
    alt = _rename_by_tokens(nat, nm, am)
    return alt if alt is not None and alt != nat and _ast_equal(nat, alt, _AstRename(nm, am)) else None


# ----------------------------------------------------------------------------------------------------------------- literal families
_STR_PREFIX = re.compile(r"^([rRbBuUfF]*)(.*)$", re.S)


def py_quotes(nat, spec=""):
    """every single-quoted literal -> double quotes; a literal whose content contains a double quote (or, for ''' strings, contains \"\"\"
    or ends with a double quote) is left as it is (the stored pairs do the same)."""
    toks = _tokens(nat)
    if toks is None:
        return None
    by_char, _ = _offsets(nat); edits = []
    for t in toks:
        if t.type != tokenize.STRING:
            continue
        pre, body = _STR_PREFIX.match(t.string).groups()
        if body.startswith("'''") and body.endswith("'''") and len(body) >= 6:
            inner = body[3:-3]
            if '"""' in inner or inner.endswith('"') or inner.endswith("\\"):
                continue
            edits.append((by_char(*t.start), by_char(*t.end), pre + '"""' + inner + '"""'))
        elif body.startswith("'") and body.endswith("'") and len(body) >= 2 and not body.startswith("'''"):
            inner = body[1:-1]
            if '"' in inner:
                if "r" in pre.lower() or "f" in pre.lower() or "\\" in inner:
                    continue                                       # raw / f-string (no backslash allowed in {..}) / already-escaped: left as it is
                inner = inner.replace('"', '\\"')
            if "r" not in pre.lower():
                inner = inner.replace("\\'", "'")                    # an escaped single quote is no longer needed inside double quotes
            edits.append((by_char(*t.start), by_char(*t.end), pre + '"' + inner + '"'))
    alt = _apply(nat, edits)
    return alt if edits and WS.same_ast(nat, alt) else None


def docstring_quotes(nat, spec=""):
    try:
        tree = ast.parse(nat)
    except SyntaxError:
        return None
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
            docs.add((n.body[0].value.lineno, n.body[0].value.col_offset))
    toks = _tokens(nat)
    if toks is None:
        return None
    by_char, by_byte = _offsets(nat); doc_offsets = {by_byte(r, c) for r, c in docs}; edits = []
    for t in toks:
        if t.type == tokenize.STRING:                              # every triple-double string (docstrings and standalone block strings)
            pre, body = _STR_PREFIX.match(t.string).groups()
            if not (body.startswith('"""') and body.endswith('"""') and len(body) >= 6):
                continue
            inner = body[3:-3]
            if "'''" in inner or inner.endswith("'") or inner.endswith("\\"):
                return None
            edits.append((by_char(*t.start), by_char(*t.end), pre + "'''" + inner + "'''"))
    alt = _apply(nat, edits)
    return alt if edits and WS.same_ast(nat, alt) else None


_FLOAT = re.compile(r"^(\d+)\.(\d+)((?:[eE][+-]?\d+)?)$")


def float_literals(nat, spec=""):
    """0.5 -> .5 (leading zero dropped), 1.0 -> 1. and 2.50 -> 2.5 (trailing zero dropped), 0.0 -> 0. ; exponent kept."""
    toks = _tokens(nat)
    if toks is None:
        return None
    by_char, _ = _offsets(nat); edits = []
    for t in toks:
        if t.type != tokenize.NUMBER:
            continue
        m = _FLOAT.match(t.string)
        if not m:
            continue
        a, b, e = m.groups(); nb = b.rstrip("0")
        if a == "0" and nb:                                        # 0.5 -> .5 ; 0.50 -> .5
            new = "." + nb
        elif nb == "":                                             # 1.0 -> 1. ; 0.0 -> 0.
            new = a + "."
        else:                                                      # 2.25 unchanged ; 2.50 -> 2.5
            new = a + "." + nb
        if new != a + "." + b:
            edits.append((by_char(*t.start), by_char(*t.end), new + e))
    alt = _apply(nat, edits)
    return alt if edits and WS.same_ast(nat, alt) else None


def trailing_commas(nat, spec=""):
    """remove the comma that is the last token before a closing bracket (only NL / comments in between); 1-tuples and subscript
    tuples keep theirs (WS._drop_trailing_comma)."""
    toks = _tokens(nat)
    if toks is None:
        return None
    by_char, _ = _offsets(nat)
    sig = [t for t in toks if t.type not in WS.SKIP]
    edits = []
    for i, t in enumerate(sig):
        if not (t.type == tokenize.OP and t.string == ","):
            continue
        j = i + 1
        while j < len(sig) and sig[j].type in (tokenize.NL, tokenize.COMMENT):
            j += 1
        if j < len(sig) and sig[j].type == tokenize.OP and sig[j].string in WS.CLOSE:
            if WS._drop_trailing_comma(sig, WS._opener_of(sig, j), j):
                edits.append((by_char(*t.start), by_char(*t.end), ""))
    alt = _apply(nat, edits)
    return alt if edits and WS.same_ast(nat, alt) else None


# ----------------------------------------------------------------------------------------------------------------- comment family
def comment_case(nat, spec=""):
    toks = _tokens(nat)
    if toks is None:
        return None
    by_char, _ = _offsets(nat); edits = []
    for t in toks:
        if t.type != tokenize.COMMENT or t.string.startswith("#!") or re.match(r"#\s*-\*-", t.string):
            continue
        m = re.match(r"(#+\s*)([A-Za-z][A-Za-z0-9']*)", t.string)
        if not m or not m.group(2)[0].isupper():
            continue
        word = m.group(2); low = word.lower() if (len(word) > 1 and word.isupper()) else word[0].lower() + word[1:]
        s = by_char(*t.start) + m.start(2)
        edits.append((s, s + len(word), low))
    alt = _apply(nat, edits)
    return alt if edits and WS.same_ast(nat, alt) else None


# ----------------------------------------------------------------------------------------------------------------- whitespace families
def comma_space(nat, spec=""):
    alt = WS.comma_space(nat); return alt if alt and alt != nat and WS.same_ast(nat, alt) else None


def operator_spaces(nat, spec=""):
    alt = WS.operator_spaces(nat); return alt if alt and alt != nat and WS.same_ast(nat, alt) else None


def blank_lines(nat, spec=""):
    alt = WS.blank_lines(nat); return alt if alt and alt != nat and WS.same_ast(nat, alt) else None


RULES = {"py_const_naming": py_const_naming, "py_class_naming": py_class_naming, "py_loop_vars": py_loop_vars, "py_abbrev": py_abbrev,
         "py_quotes": py_quotes, "docstring_quotes": docstring_quotes, "float_literals": float_literals, "trailing_commas": trailing_commas,
         "comment_case": comment_case, "comma_space": comma_space, "operator_spaces": operator_spaces, "blank_lines": blank_lines}
