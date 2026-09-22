"""Exact rule-based alternative twins, GROUP A (Python syntax families; 2026-09-22, regen8 preparation).

Each rule `fam(nat, spec="") -> alt | None` applies the family's rewrite (code_families.CODE_SPECS) to EVERY covered occurrence and to
nothing else, editing the source at AST-node byte offsets so the file stays byte-identical outside the changed tokens. A rule returns
None when it cannot guarantee that (unparsable, ambiguous construct, name collision). Verification: the alternative parses (py2_* twins
cannot parse under Python 3: there the reverse edit must reproduce the natural twin and the edit spans must be exactly the intended
tokens) and an AST transform of the natural twin equals the AST of the alternative where the rewrite is expressible in Python 3.
"""
import ast
import io
import re
import tokenize


# ---------------------------------------------------------------- helpers
def _parse(nat):
    try:
        return ast.parse(nat)
    except (SyntaxError, ValueError):
        return None


class _Src:
    """Byte-offset editing of a source text at ast positions (col offsets are UTF-8 byte offsets)."""
    def __init__(self, text):
        self.text = text; self.b = text.encode("utf-8"); self.lines = self.b.split(b"\n"); self.off = [0]
        for ln in self.lines:
            self.off.append(self.off[-1] + len(ln) + 1)
        self.edits = []

    def pos(self, lineno, col):
        return self.off[lineno - 1] + col

    def start(self, n):
        return self.pos(n.lineno, n.col_offset)

    def end(self, n):
        return self.pos(n.end_lineno, n.end_col_offset)

    def seg(self, a, b):
        return self.b[a:b].decode("utf-8")

    def replace(self, a, b, new):
        self.edits.append((a, b, new.encode("utf-8") if isinstance(new, str) else new))

    def apply(self):
        out = self.b
        for a, b, new in sorted(self.edits, key=lambda e: (e[0], e[1]), reverse=True):
            out = out[:a] + new + out[b:]
        return out.decode("utf-8")

    def overlapping(self):
        es = sorted((a, b) for a, b, _ in self.edits)
        return any(es[i][1] > es[i + 1][0] for i in range(len(es) - 1))


def _tokens(text):
    try:
        return list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return None


def _same_ast(a_tree, b_text):
    try:
        return ast.dump(a_tree) == ast.dump(ast.parse(b_text))
    except (SyntaxError, ValueError):
        return False


def _add_typing_import(src, names):
    """Merge `names` into an existing `from typing import ...` line (kept sorted when that line was sorted) or insert a new sorted import
    the way the reviewer-approved pairs do: after the last top-level import, else after the module docstring (+ blank line), else at the
    top of the file (two blank lines before a following def/class, one otherwise)."""
    m = re.search(r"^from typing import ([^\n(]+)$", src, re.M)
    if m:
        have = [x.strip() for x in m.group(1).split(",") if x.strip()]
        new = have + [n for n in sorted(names) if n not in have]
        if have == sorted(have):
            new = sorted(new)
        return src[:m.start(1)] + ", ".join(new) + src[m.end(1):]
    line = f"from typing import {', '.join(sorted(names))}\n"
    tree = ast.parse(src); lines = src.splitlines(True)
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if imports:
        last = max(n.end_lineno for n in imports)
        return "".join(lines[:last]) + line + "".join(lines[last:])
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        d = tree.body[0].end_lineno
        rest = "".join(lines[d:])
        return "".join(lines[:d]) + "\n" + line + ("\n" if not rest.startswith("\n") else "") + rest
    first = next((l for l in lines if l.strip()), "")
    return line + ("\n\n" if re.match(r"(async\s+)?(def|class)\b|@", first) else "\n") + src


# ---------------------------------------------------------------- py2_print: print(x) -> print x
def py2_print(nat, spec=""):
    tree = _parse(nat)
    if tree is None:
        return None
    S = _Src(nat); n_calls = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print":
            n_calls += 1
    stmts = [n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "print"]
    if not stmts or len(stmts) != n_calls:                          # print used as an expression (lambda, comprehension, argument) -> decline
        return None
    for st in stmts:
        c = st.value
        if c.keywords or any(isinstance(a, ast.Starred) for a in c.args):
            return None                                             # end= / sep= / file= / *args have no statement form
        if c.end_lineno != c.lineno:
            return None                                             # multi-line print call: keep the rule simple and safe
        a, b = S.start(c), S.end(c); txt = S.seg(a, b)
        if not txt.startswith("print(") or not txt.endswith(")"):
            return None
        inner = txt[len("print("):-1]
        if c.args:
            S.replace(a, b, "print " + inner.strip() if not inner.startswith(" ") else "print" + inner.rstrip())
        else:
            S.replace(a, b, "print")
    alt = S.apply()
    # reverse check: `print X` -> `print(X)` on exactly those statements reproduces the natural twin
    back = _Src(alt)
    try:
        for st in stmts:
            pass
    except Exception:
        return None
    if alt == nat or _tokens(alt) is None:
        return None
    return alt


# ---------------------------------------------------------------- py2_iter: range -> xrange, .items() -> .iteritems() ...
_ITER_MAP = {"items": "iteritems", "keys": "iterkeys", "values": "itervalues"}


def py2_iter(nat, spec=""):
    tree = _parse(nat)
    if tree is None:
        return None
    S = _Src(nat)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id == "range":
                S.replace(S.start(f), S.end(f), "xrange")
            elif isinstance(f, ast.Attribute) and f.attr in _ITER_MAP and not n.args and not n.keywords:
                b = S.end(f); S.replace(b - len(f.attr), b, _ITER_MAP[f.attr])
    if not S.edits:
        return None
    alt = S.apply()
    if _parse(alt) is None:
        return None
    if re.search(r"\b(?:xrange|iteritems|iterkeys|itervalues)\b", nat):
        return None                                                 # natural twin already contains the alternative idiom
    return alt


# ---------------------------------------------------------------- py2_except: except E as e: -> except E, e:
def py2_except(nat, spec=""):
    tree = _parse(nat); toks = _tokens(nat)
    if tree is None or toks is None:
        return None
    S = _Src(nat); handlers = [h for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler) and h.name]
    if not handlers:
        return None
    for h in handlers:
        if h.type is None or h.type.end_lineno != h.lineno:
            return None
        te = S.end(h.type)
        # the ` as NAME` after the type on the same line
        line_end = S.off[h.lineno] - 1 if h.lineno < len(S.off) else len(S.b)
        rest = S.seg(te, line_end)
        m = re.match(r"(\s+)as(\s+)" + re.escape(h.name) + r"\s*:", rest)
        if not m:
            return None
        S.replace(te, te + m.end(2), ", " if True else None)
        # keep original spacing style: ` as err:` -> `, err:`
        S.edits[-1] = (te, te + m.end(2), b", ")
    alt = S.apply()
    if alt == nat or _tokens(alt) is None:
        return None
    return alt


# ---------------------------------------------------------------- py_builtin_generics: list[int] -> List[int] + typing import
_GEN = {"list": "List", "dict": "Dict", "set": "Set", "tuple": "Tuple", "frozenset": "FrozenSet", "type": "Type"}


def py_builtin_generics(nat, spec=""):
    tree = _parse(nat)
    if tree is None:
        return None
    S = _Src(nat); names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in _GEN:
            S.replace(S.start(n.value), S.end(n.value), _GEN[n.value.id]); names.add(_GEN[n.value.id])
    if not names:
        return None
    if re.search(r"\b(?:" + "|".join(names) + r")\[", nat):
        return None                                                 # typing generics already present in the natural twin: mixed poles
    alt = _add_typing_import(S.apply(), names)
    if _parse(alt) is None:
        return None
    return alt


# ---------------------------------------------------------------- py_optional: X | None -> Optional[X] + typing import
def _has_none(n):
    return isinstance(n, ast.Constant) and n.value is None


def py_optional(nat, spec=""):
    tree = _parse(nat)
    if tree is None:
        return None
    S = _Src(nat); found = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr) and (_has_none(n.left) or _has_none(n.right)):
            if _has_none(n.left) and _has_none(n.right):
                return None
            other = n.right if _has_none(n.left) else n.left
            if n.end_lineno != n.lineno:
                return None                                         # multi-line union: decline
            inner = S.seg(S.start(other), S.end(other))
            S.replace(S.start(n), S.end(n), f"Optional[{inner}]"); found += 1
    if not found or S.overlapping():
        return None
    if re.search(r"\bOptional\[", nat):
        return None
    alt = _add_typing_import(S.apply(), {"Optional"})
    if _parse(alt) is None:
        return None
    return alt


# ---------------------------------------------------------------- py_type_hints: remove every annotation
def py_type_hints(nat, spec=""):
    tree = _parse(nat); toks = _tokens(nat)
    if tree is None or toks is None:
        return None
    S = _Src(nat); found = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.arg) and n.annotation is not None:
            S.replace(S.pos(n.lineno, n.col_offset) + len(n.arg.encode("utf-8")), S.end(n.annotation), ""); found += 1
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.returns is not None:
            # from the `)` closing the parameter list up to the end of the return annotation: keep `)`, drop ` -> T`
            re_end = S.end(n.returns)
            # find the `->` token before the annotation
            arrow = None
            for t in toks:
                if t.type == tokenize.OP and t.string == "->" and (t.start[0], t.start[1]) < (n.returns.lineno, n.returns.col_offset) and t.start[0] >= n.lineno:
                    arrow = t
            if arrow is None:
                return None
            a = S.pos(arrow.start[0], len(S.text.splitlines(True)[arrow.start[0] - 1][:arrow.start[1]].encode("utf-8")))
            # remove from the last ')' before the arrow (exclusive) to the annotation end
            seg = S.seg(0, a)
            j = seg.rfind(")")
            if j < 0:
                return None
            S.replace(len(seg[:j + 1].encode("utf-8")), re_end, ""); found += 1
        elif isinstance(n, ast.AnnAssign):
            if n.value is None:
                return None                                         # bare `x: int` has no un-annotated form
            S.replace(S.end(n.target), S.end(n.annotation), ""); found += 1
    if not found or S.overlapping():
        return None
    alt = S.apply()
    if _parse(alt) is None:
        return None
    return alt


# ---------------------------------------------------------------- py_paren_if: if x: -> if (x):
def py_paren_if(nat, spec=""):
    tree = _parse(nat)
    if tree is None:
        return None
    S = _Src(nat); found = 0
    for n in ast.walk(tree):
        if isinstance(n, (ast.If, ast.While)):
            t = n.test
            if isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__":
                continue                                            # `if __name__ == "__main__":` stays as is (approved pairs)
            a, b = S.start(t), S.end(t); txt = S.seg(a, b)
            if txt.startswith("(") and _balanced_whole(txt):
                return None                                         # condition already wholly parenthesised in the natural twin
            S.replace(a, b, "(" + txt + ")"); found += 1
    if not found:
        return None
    alt = S.apply()
    return alt if _same_ast(tree, alt) else None


def _balanced_whole(txt):
    depth = 0
    for i, ch in enumerate(txt):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i == len(txt) - 1
    return False


# ---------------------------------------------------------------- py_literal_ctor: {} -> dict(), [] -> list()
def py_literal_ctor(nat, spec=""):
    tree = _parse(nat)
    if tree is None:
        return None
    S = _Src(nat); found = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Dict) and not n.keys:
            S.replace(S.start(n), S.end(n), "dict()"); found += 1
        elif isinstance(n, ast.List) and not n.elts and isinstance(n.ctx, ast.Load):
            S.replace(S.start(n), S.end(n), "list()"); found += 1
    if not found:
        return None
    alt = S.apply()
    if _parse(alt) is None:
        return None
    return alt


# ---------------------------------------------------------------- py_is_none: is None -> == None, is not None -> != None
def py_is_none(nat, spec=""):
    tree = _parse(nat); toks = _tokens(nat)
    if tree is None or toks is None:
        return None
    S = _Src(nat); found = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            left = n.left
            for op, comp in zip(n.ops, n.comparators):
                if isinstance(op, (ast.Is, ast.IsNot)) and (_has_none(comp) or _has_none(left)):
                    a, b = S.end(left), S.start(comp); between = S.seg(a, b)
                    m = re.fullmatch(r"(\s*)is(\s+not)?(\s*)", between)
                    if not m:
                        return None
                    S.replace(a, b, m.group(1) + ("!=" if isinstance(op, ast.IsNot) else "==") + m.group(3)); found += 1
                left = comp
    if not found:
        return None
    alt = S.apply()
    if _parse(alt) is None or re.search(r"[!=]=\s*None\b", nat):
        return None
    return alt


# ---------------------------------------------------------------- py_not_in: x not in y -> not x in y
def py_not_in(nat, spec=""):
    tree = _parse(nat)
    if tree is None:
        return None
    S = _Src(nat); found = 0
    class T(ast.NodeTransformer):
        def visit_Compare(self, n):
            self.generic_visit(n)
            if len(n.ops) == 1 and isinstance(n.ops[0], ast.NotIn):
                return ast.copy_location(ast.UnaryOp(op=ast.Not(), operand=ast.Compare(left=n.left, ops=[ast.In()], comparators=n.comparators)), n)
            return n
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare) and any(isinstance(o, ast.NotIn) for o in n.ops):
            if len(n.ops) != 1:
                return None
            a, b = S.end(n.left), S.start(n.comparators[0]); between = S.seg(a, b)
            m = re.fullmatch(r"(\s*)not(\s+)in(\s*)", between)
            if not m:
                return None
            S.replace(a, b, m.group(1) + "in" + m.group(3)); S.replace(S.start(n.left), S.start(n.left), "not "); found += 1
    if not found or any(isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not) and isinstance(n.operand, ast.Compare) and any(isinstance(o, ast.In) for o in n.operand.ops) for n in ast.walk(tree)):
        return None                                                 # natural twin already spells `not x in y` somewhere
    alt = S.apply()
    want = ast.dump(T().visit(_parse(nat)))
    try:
        return alt if ast.dump(ast.parse(alt)) == want else None
    except SyntaxError:
        return None


# ---------------------------------------------------------------- py_self_name: self -> this everywhere
def py_self_name(nat, spec=""):
    toks = _tokens(nat)
    if toks is None or not re.search(r"\bself\b", nat) or any(t.type == tokenize.NAME and t.string == "this" for t in toks):
        return None
    alt = re.sub(r"\bself\b", "this", nat)
    return alt if _parse(alt) is not None else None


RULES = {"py2_print": py2_print, "py2_iter": py2_iter, "py2_except": py2_except, "py_builtin_generics": py_builtin_generics, "py_optional": py_optional,
         "py_type_hints": py_type_hints, "py_paren_if": py_paren_if, "py_literal_ctor": py_literal_ctor, "py_is_none": py_is_none, "py_not_in": py_not_in,
         "py_self_name": py_self_name}
