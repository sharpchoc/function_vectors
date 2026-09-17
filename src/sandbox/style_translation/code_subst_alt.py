#!/usr/bin/env python
"""Deterministic alternative twin for bash_subst (bug 7, 2026-09-17): every `$(cmd)` command substitution of the natural twin becomes a
backtick substitution, nested ones with Bash's own escaping rule (inside `...` a backslash is removed only before $ ` \\, so an inner
backtick pair is written \\`...\\` and inner backslashes before those three characters are doubled; deeper levels escape again).

The scanner knows single quotes, double quotes, comments, here-documents and $((...)) arithmetic, so a `$(` inside any of those is left
alone. `convert(text)` returns the alternative text; `subst_tree(text)` returns the nested command-substitution structure used to check
that the converted text parses back to the same commands (`same_tree`).
"""
import re

_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def _scan(text):
    """Yield the top-level `$(...)` substitutions of `text` as (start, end) with start at `$` and end after `)`, honouring quotes,
    comments, heredocs and arithmetic. `text` is parsed as shell code (a fresh context)."""
    i = 0; n = len(text); out = []
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2; continue
        if c == "'":
            j = text.find("'", i + 1); i = n if j < 0 else j + 1; continue
        if c == "#" and (i == 0 or text[i - 1] in " \t\n;&|("):
            j = text.find("\n", i); i = n if j < 0 else j; continue
        if c == '"':
            i = _skip_dquote(text, i, out); continue
        if c == "<" and text.startswith("<<", i) and not text.startswith("<<<", i):
            m = _HEREDOC.match(text, i)
            if m:
                term = m.group(2); eol = text.find("\n", m.end()); eol = n if eol < 0 else eol
                # substitutions on the rest of this line (before the body) are still code
                out += [(s + m.end(), e + m.end()) for s, e in _scan(text[m.end():eol])]
                body_end = _heredoc_end(text, eol + 1, term); i = body_end; continue
        if c == "$" and text.startswith("$((", i):
            i = _match_arith(text, i); continue
        if c == "$" and text.startswith("$(", i):
            j = _match_paren(text, i + 1); out.append((i, j)); i = j; continue
        if c == "`":
            j = text.find("`", i + 1); i = n if j < 0 else j + 1; continue
        i += 1
    return out


def _heredoc_end(text, i, term):
    while i < len(text):
        eol = text.find("\n", i); eol = len(text) if eol < 0 else eol
        if text[i:eol].lstrip("\t") == term:
            return eol
        i = eol + 1
    return len(text)


def _skip_dquote(text, i, out):
    """i at the opening double quote; substitutions inside are collected; returns index after the closing quote."""
    i += 1; n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2; continue
        if c == '"':
            return i + 1
        if c == "$" and text.startswith("$((", i):
            i = _match_arith(text, i); continue
        if c == "$" and text.startswith("$(", i):
            j = _match_paren(text, i + 1); out.append((i, j)); i = j; continue
        if c == "`":
            j = text.find("`", i + 1); i = n if j < 0 else j + 1; continue
        i += 1
    return n


def _match_paren(text, i):
    """i at '('; returns index after the matching ')' — the content is shell code, so nested quotes / substitutions are honoured."""
    depth = 0; n = len(text); j = i
    while j < n:
        c = text[j]
        if c == "\\":
            j += 2; continue
        if c == "'":
            k = text.find("'", j + 1); j = n if k < 0 else k + 1; continue
        if c == '"':
            j = _skip_dquote(text, j, []); continue
        if c == "#" and text[j - 1] in " \t\n;&|(":
            k = text.find("\n", j); j = n if k < 0 else k; continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    raise ValueError("unbalanced $(")


def _match_arith(text, i):
    depth = 0; j = i + 1; n = len(text)
    while j < n:
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    raise ValueError("unbalanced $((")


def _escape(cmd):
    """Text to write inside a backtick pair so that Bash's backtick unescaping yields `cmd` exactly."""
    out = []; i = 0
    while i < len(cmd):
        c = cmd[i]
        if c == "`":
            out.append("\\`")
        elif c == "\\" and i + 1 < len(cmd) and cmd[i + 1] in "$`\\":
            out.append("\\\\")
        else:
            out.append(c)
        i += 1
    return "".join(out)


def convert(text):
    subs = _scan(text)
    if not subs:
        return text
    parts = []; pos = 0
    for s, e in subs:
        inner = convert(text[s + 2:e - 1])                       # inner substitutions first
        parts.append(text[pos:s]); parts.append("`" + _escape(inner) + "`"); pos = e
    parts.append(text[pos:])
    return "".join(parts)


# ---- independent check: parse the converted text with Bash's backtick rules and compare substitution trees
def _unescape(s):
    out = []; i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] in "$`\\":
            out.append(s[i + 1]); i += 2
        else:
            out.append(s[i]); i += 1
    return "".join(out)


def _scan_backticks(text):
    """Top-level backtick substitutions as (start, end); a backslash-escaped backtick does not close (Bash rule)."""
    i = 0; n = len(text); out = []
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2; continue
        if c == "'":
            j = text.find("'", i + 1); i = n if j < 0 else j + 1; continue
        if c == "#" and (i == 0 or text[i - 1] in " \t\n;&|("):
            j = text.find("\n", i); i = n if j < 0 else j; continue
        if c == "<" and text.startswith("<<", i) and not text.startswith("<<<", i):
            m = _HEREDOC.match(text, i)
            if m:
                eol = text.find("\n", m.end()); eol = n if eol < 0 else eol
                out += [(s + m.end(), e + m.end()) for s, e in _scan_backticks(text[m.end():eol])]
                i = _heredoc_end(text, eol + 1, m.group(2)); continue
        if c == "`":
            i = _backtick_group(text, i, out); continue
        if c == '"':                                             # inside double quotes: backticks stay active, an apostrophe is literal
            i += 1
            while i < n and text[i] != '"':
                if text[i] == "\\":
                    i += 2
                elif text[i] == "`":
                    i = _backtick_group(text, i, out)
                elif text[i] == "$" and text.startswith("$((", i):
                    i = _match_arith(text, i)
                else:
                    i += 1
            i += 1; continue
        if c == "$" and text.startswith("$((", i):
            i = _match_arith(text, i); continue
        i += 1
    return out


def _backtick_group(text, i, out):
    j = i + 1; n = len(text)
    while j < n and text[j] != "`":
        j += 2 if text[j] == "\\" else 1
    out.append((i, j + 1)); return j + 1


def subst_tree(text, kind="dollar"):
    """Nested structure [(command text with inner substitutions replaced by ⟨k⟩, [children...]), ...] — same for both notations if the
    conversion is faithful."""
    spans = _scan(text) if kind == "dollar" else _scan_backticks(text)
    tree = []
    for s, e in spans:
        inner = text[s + 2:e - 1] if kind == "dollar" else _unescape(text[s + 1:e - 1])
        kids = subst_tree(inner, kind)
        skel = inner
        for k, (ks, ke) in enumerate(reversed(_scan(inner) if kind == "dollar" else _scan_backticks(inner))):
            skel = skel[:ks] + "⟨⟩" + skel[ke:]
        tree.append((skel, kids))
    return tree


def outside_text(text, kind="dollar"):
    """The text with every top-level substitution removed — must be identical in both notations."""
    spans = _scan(text) if kind == "dollar" else _scan_backticks(text); out = []; pos = 0
    for s, e in spans:
        out.append(text[pos:s]); pos = e
    out.append(text[pos:]); return "".join(out)


def same_tree(nat, alt):
    return subst_tree(nat, "dollar") == subst_tree(alt, "backtick") and outside_text(nat, "dollar") == outside_text(alt, "backtick")


if __name__ == "__main__":
    import sys
    src = open(sys.argv[1]).read(); out = convert(src); print(out, end="")
    print("same_tree:", same_tree(src, out), file=sys.stderr)
