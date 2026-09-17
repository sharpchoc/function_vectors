"""Code-convention families (user request 2026-09-14): 60 style conventions in code, tested with the style-translation machinery.
Prompt = "Task:\\n{spec}\\n\\n{Language}:\\n{solution twin cut at cue k}". Natural pole = dominant modern form; alternative = the other.
Twins are produced by Gemini (natural solution, then a rewrite changing ONLY the convention) and aligned by a token diff (code_build.py);
each family's Property here only CLASSIFIES a continuation (regex pair on the tail; decide() falls back to the stored next tokens).
Registered into ml_families.ML_FAMILIES with domain="code" so cue_tokens / build_prompts / rollout / judge / analysis work unchanged."""
import re
import sys
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.ext_styleprops.properties import PROPS, Property
from src.sandbox.style_translation.ml_families import MLFamily, ML_FAMILIES, ML_FAMILY


def _code_property(name, nat_rx, alt_rx, nat_lab, alt_lab):
    nr, ar = re.compile(nat_rx, re.M), re.compile(alt_rx, re.M)

    class C(Property):
        family = "code"

        def find_opps(self, text):   # opportunities come from the twin diff, not from a detector
            return []

        def classify(self, tail):
            head = tail[:160]
            mn, ma = nr.search(head), ar.search(head)
            if mn and (ma is None or mn.start() <= ma.start()):
                return "nat"
            if ma:
                return "alt"
            return None

    C.name = name; C.nat_label, C.alt_label, C.confound = nat_lab, alt_lab, "low"
    return C


# (name, language, natural label, alternative label, generation hint for the NATURAL solution, rewrite instruction for the ALTERNATIVE, nat_rx, alt_rx, ignore-for-judge)
ID_SNAKE = r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"
ID_CAMEL = r"\b[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*\b"
CODE_SPECS = [
 # ---- naming
 ("py_snake_camel", "Python", "snake_case identifiers", "camelCase identifiers",
  "use snake_case for ALL variable, parameter and function names, with at least 8 multi-word names (e.g. total_count, max_value, input_list)",
  "rename every multi-word variable, parameter and function name from snake_case to camelCase (total_count -> totalCount); do not touch class names, builtins, strings, comments or anything else",
  ID_SNAKE, ID_CAMEL, "snake_case vs camelCase identifiers"),
 ("js_camel_snake", "JavaScript", "camelCase identifiers", "snake_case identifiers",
  "use camelCase for ALL variable, parameter and function names, with at least 8 multi-word names",
  "rename every multi-word variable, parameter and function name from camelCase to snake_case (totalCount -> total_count); change nothing else",
  ID_CAMEL, ID_SNAKE, "camelCase vs snake_case identifiers"),
 ("js_func_pascal", "JavaScript", "camelCase function names", "PascalCase function names",
  "define at least 6 helper functions (function declarations or const arrow functions) with camelCase multi-word names, called several times",
  "rename every function (declaration and const arrow) from camelCase to PascalCase (parseInput -> ParseInput) at definition and every call site; change nothing else",
  r"\b(?:function\s+|const\s+)[a-z]\w*[A-Z]\w*\b", r"\b(?:function\s+|const\s+)[A-Z][a-z]\w*\b", "camelCase vs PascalCase function names"),
 ("py_const_naming", "Python", "UPPER_SNAKE constants", "camelCase constants",
  "define at least 8 module-level constants in UPPER_SNAKE_CASE (MAX_RETRIES, DEFAULT_TIMEOUT) and use them",
  "rename every UPPER_SNAKE constant to camelCase (MAX_RETRIES -> maxRetries) at definition and every use; change nothing else",
  r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b", r"^[a-z]+[A-Z]\w*\s*=", "UPPER_SNAKE vs camelCase constants"),
 ("py_class_naming", "Python", "PascalCase class names", "snake_case class names",
  "define at least 5 small classes with multi-word PascalCase names (OrderItem, PriceRule) and instantiate each",
  "rename every class from PascalCase to snake_case (OrderItem -> order_item) at definition and every use; change nothing else",
  r"\bclass\s+[A-Z]\w*\b|\b[A-Z][a-z]+[A-Z]\w*\(", r"\bclass\s+[a-z]+_\w*\b|\b[a-z]+_[a-z]\w*\(", "PascalCase vs snake_case class names"),
 ("py_private", "Python", "single-underscore private attributes", "double-underscore private attributes",
  "write a class with at least 8 private attributes/methods prefixed with a single underscore (self._cache, self._load())",
  "change every single-underscore private attribute/method prefix to a double underscore (self._cache -> self.__cache); change nothing else",
  r"self\._[a-z]", r"self\.__[a-z]", "single vs double underscore private members"),
 ("py_bool_prefix", "Python", "is_/has_ boolean names", "bare boolean names",
  "use at least 8 boolean variables/parameters named with an is_/has_/can_/should_ prefix (is_valid, has_items)",
  "remove the is_/has_/can_/should_ prefix from every boolean name (is_valid -> valid, has_items -> items) at every occurrence; change nothing else",
  r"\b(?:is|has|can|should)_[a-z]\w*\b", r"\b(?:valid|ready|done|empty|active|enabled|found|complete|items|errors|children|permission|finished|visible|open)\b", "boolean name prefixes"),
 ("py_loop_vars", "Python", "single-letter loop variables", "descriptive loop variables",
  "write at least 6 for-loops that use single-letter loop variables (i, j, k, n)",
  "rename every single-letter loop variable to a descriptive word (i -> index, j -> column, k -> key) everywhere it is used; change nothing else",
  r"\bfor\s+[ijkn]\b", r"\bfor\s+[a-z]{3,}\b", "single-letter vs descriptive loop variables"),
 ("js_hungarian", "JavaScript", "plain identifiers", "Hungarian notation",
  "use at least 8 plain camelCase variable names for strings, numbers, arrays and booleans",
  "rename every variable with Hungarian-notation prefixes by type (count -> nCount, name -> strName, items -> arrItems, ready -> bReady) at every use; change nothing else",
  r"\b(?:const|let|var)\s+(?!n[A-Z]|str[A-Z]|arr[A-Z]|b[A-Z]|obj[A-Z]|fn[A-Z])[a-z]\w*", r"\b(?:n|str|arr|b|obj|fn)[A-Z]\w*\b", "Hungarian notation"),
 ("py_abbrev", "Python", "abbreviated identifiers", "spelled-out identifiers",
  "use at least 8 common abbreviations as identifiers (cfg, idx, buf, num, val, tmp, msg, ctx, res, err, cnt, pos, prev, curr, elem, param)",
  "expand every abbreviated identifier to its full word (cfg -> configuration, idx -> index, buf -> buffer, num -> number, val -> value, tmp -> temporary, msg -> message, ctx -> context, res -> result, err -> error, cnt -> count, pos -> position, prev -> previous, curr -> current, elem -> element, param -> parameter) everywhere; change nothing else",
  r"\b(?:cfg|idx|buf|num|val|tmp|msg|ctx|res|err|cnt|pos|prev|curr|elem|param)\b", r"\b(?:configuration|index|buffer|number|value|temporary|message|context|result|error|count|position|previous|current|element|parameter)\b", "abbreviated vs full identifiers"),
 # ---- literals
 ("py_quotes", "Python", "single-quoted strings", "double-quoted strings",
  "use single quotes for every string literal and include at least 8 string literals",
  "change every string literal from single quotes to double quotes; change nothing else",
  r"'", r'"', "single vs double quotes"),
 ("js_quotes", "JavaScript", "double-quoted strings", "single-quoted strings",
  "use double quotes for every string literal and include at least 8 string literals",
  "change every string literal from double quotes to single quotes; change nothing else",
  r'"', r"'", "double vs single quotes"),
 ("py_fstring", "Python", "f-strings", "str.format",
  "build at least 6 strings with f-strings (f\"...{x}...\")",
  "rewrite every f-string as an equivalent \"...{}...\".format(x) call; change nothing else",
  r"\bf['\"]", r"\.format\(", "f-strings vs str.format"),
 ("js_template", "JavaScript", "template literals", "string concatenation",
  "build at least 6 strings with template literals (`...${x}...`)",
  "rewrite every template literal as an equivalent concatenation with + and quoted strings; change nothing else",
  r"`", r"['\"]\s*\+|\+\s*['\"]", "template literals vs concatenation"),
 ("num_separators", "Python", "plain numeric literals", "underscore digit separators",
  "use at least 6 integer literals of 4 or more digits (10000, 1000000, 86400, 65536)",
  "write every integer literal of 4 or more digits with underscore thousands separators (1000000 -> 1_000_000); change nothing else",
  r"\b\d{4,}\b", r"\b\d{1,3}(?:_\d{3})+\b", "digit separators"),
 ("hex_constants", "JavaScript", "decimal constants", "hexadecimal constants",
  "use at least 6 numeric constants from this set written in decimal: 255, 256, 128, 64, 4096, 65535, 1024, 16, 32",
  "write each of those constants in hexadecimal (255 -> 0xFF, 256 -> 0x100, 1024 -> 0x400); change nothing else",
  r"\b(?:255|256|128|64|4096|65535|1024|16|32)\b", r"\b0[xX][0-9a-fA-F]+\b", "decimal vs hexadecimal literals"),
 ("float_literals", "Python", "full float literals", "abbreviated float literals",
  "use at least 6 float literals with digits on both sides of the point (0.5, 1.0, 2.25, 0.75)",
  "abbreviate every float literal: drop the leading zero (0.5 -> .5) and the trailing zero (1.0 -> 1.); change nothing else",
  r"\b\d+\.\d+\b", r"(?<![\d.\w])\.\d+\b|\b\d+\.(?![\d\w])", "float literal spelling"),
 ("sql_bool_case", "SQL", "lowercase true/false", "uppercase TRUE/FALSE",
  "use the boolean literals true and false (lowercase) at least 6 times in WHERE clauses and inserted values",
  "write every boolean literal in uppercase (true -> TRUE, false -> FALSE); change nothing else",
  r"\b(?:true|false)\b", r"\b(?:TRUE|FALSE)\b", "boolean literal case"),
 # ---- version dialects
 ("py2_print", "Python", "print() function", "print statement",
  "call print(...) at least 8 times",
  "convert every print(...) call to a Python 2 print statement (print(x) -> print x, print(a, b) -> print a, b); change nothing else",
  r"\bprint\(", r"\bprint\s+[^(\s]", "print function vs statement"),
 ("py2_iter", "Python", "range/items/keys", "xrange/iteritems/iterkeys",
  "use range(...) and dict .items()/.keys()/.values() at least 8 times in loops",
  "convert to Python 2 idioms: range -> xrange, .items() -> .iteritems(), .keys() -> .iterkeys(), .values() -> .itervalues(); change nothing else",
  r"\b(?:range|items|keys|values)\(", r"\b(?:xrange|iteritems|iterkeys|itervalues)\(", "Python 2 vs 3 iteration idioms"),
 ("py2_except", "Python", "except E as e", "except E, e",
  "write at least 6 try/except blocks that bind the exception (except ValueError as err:)",
  "convert every except clause to the Python 2 comma form (except ValueError as err: -> except ValueError, err:); change nothing else",
  r"\bexcept\s+\w+\s+as\s+\w+", r"\bexcept\s+\w+\s*,\s*\w+\s*:", "except clause syntax"),
 ("js_var", "JavaScript", "const/let", "var",
  "declare at least 8 variables with const or let",
  "change every const and let declaration to var; change nothing else",
  r"\b(?:const|let)\s", r"\bvar\s", "const/let vs var"),
 ("js_arrow", "JavaScript", "arrow functions", "function expressions",
  "define at least 6 functions as arrow functions (const f = (a, b) => {...} and inline callbacks x => x * 2)",
  "rewrite every arrow function as a function expression (x => x * 2 -> function(x) { return x * 2; }); change nothing else",
  r"=>", r"\bfunction\s*\(", "arrow vs function expressions"),
 ("js_semicolons", "JavaScript", "semicolons", "no semicolons",
  "end every statement with a semicolon (at least 10 statements)",
  "remove every statement-terminating semicolon; change nothing else",
  r";\s*\n", r"[^;{}\s]\s*\n", "semicolons"),
 ("js_strict_eq", "JavaScript", "=== / !==", "== / !=",
  "use strict equality (=== and !==) in at least 6 comparisons",
  "replace every === with == and every !== with !=; change nothing else",
  r"[=!]==", r"(?<![=!<>])[=!]=(?!=)", "strict vs loose equality"),
 ("trailing_commas", "Python", "trailing commas", "no trailing commas",
  "write at least 6 multi-line list/dict/call literals whose last element ends with a trailing comma",
  "remove every trailing comma before a closing bracket; change nothing else",
  r",\s*\n\s*[\]\)\}]", r"[^,\s]\s*\n\s*[\]\)\}]", "trailing commas"),
 ("py_paren_if", "Python", "if x:", "if (x):",
  "write at least 8 if/elif/while conditions without parentheses",
  "wrap every if/elif/while condition in parentheses (if x > 0: -> if (x > 0):); change nothing else",
  r"\b(?:if|while|elif)\s+[^(\s]", r"\b(?:if|while|elif)\s*\(", "parenthesised conditions"),
 ("py_type_hints", "Python", "type hints", "no type hints",
  "annotate every parameter and return type (at least 8 annotations)",
  "remove every type annotation (parameters and return arrows); change nothing else",
  r"\w+\s*:\s*(?:int|str|float|bool|list|dict|List|Dict|Optional)\b|\)\s*->", r"def\s+\w+\([^:)]*\)\s*:", "type hints"),
 ("py_optional", "Python", "X | None", "Optional[X]",
  "annotate at least 6 parameters/returns as optional using the union syntax (int | None)",
  "rewrite every X | None annotation as Optional[X] (and add from typing import Optional at the top); change nothing else",
  r"\|\s*None\b", r"\bOptional\[", "Optional syntax"),
 ("py_builtin_generics", "Python", "list[int] builtins", "typing.List",
  "annotate at least 6 types with builtin generics (list[int], dict[str, int], tuple[int, int])",
  "rewrite every builtin generic as its typing equivalent (list[int] -> List[int], dict -> Dict, tuple -> Tuple) and add the typing import; change nothing else",
  r"\b(?:list|dict|set|tuple)\[", r"\b(?:List|Dict|Set|Tuple)\[", "builtin vs typing generics"),
 ("py_literal_ctor", "Python", "{} / [] literals", "dict() / list() constructors",
  "create at least 6 empty containers with literals (x = {} or x = [])",
  "replace every empty literal with its constructor ({} -> dict(), [] -> list()); change nothing else",
  r"=\s*(?:\{\}|\[\])", r"\b(?:dict|list|set)\(\)", "literal vs constructor"),
 ("py_comprehension", "Python", "list comprehensions", "explicit loops",
  "build at least 5 lists with list comprehensions",
  "rewrite every list comprehension as an explicit for-loop that appends to a list initialised just before; change nothing else",
  r"\[[^\]\n]*\bfor\b[^\]\n]*\]", r"\.append\(", "comprehension vs loop"),
 ("py_not_in", "Python", "x not in y", "not x in y",
  "use the operator 'not in' at least 6 times",
  "rewrite every 'x not in y' as 'not x in y'; change nothing else",
  r"\bnot\s+in\b", r"\bnot\s+\w[\w.\[\]']*\s+in\b", "not-in spelling"),
 ("py_is_none", "Python", "is None", "== None",
  "compare with None using 'is None' / 'is not None' at least 6 times",
  "rewrite every 'is None' as '== None' and 'is not None' as '!= None'; change nothing else",
  r"\bis\s+(?:not\s+)?None\b", r"[!=]=\s*None\b", "None comparison"),
 ("py_ternary", "Python", "conditional expressions", "if/else blocks",
  "assign at least 5 variables with conditional expressions (x = a if cond else b)",
  "rewrite every conditional-expression assignment `x = a if cond else b` as the block `if cond:` / `    x = a` / `else:` / `    x = b` (four lines, NO pre-declaration of x before the if); change nothing else",
  r"=\s*[^\n]+\s+if\s+[^\n]+\s+else\s+", r"^\s*if\b[^\n]*:\s*\n\s+\w+\s*=", "ternary vs if/else"),
 ("early_return", "Python", "guard clauses", "nested if/else",
  "write at least 5 guard clauses (if bad: return ...) at the start of functions",
  "rewrite every guard clause as an if/else that nests the rest of the function in the else branch; change nothing else",
  r"\bif\b[^\n]*:\s*\n\s+(?:return|raise|continue)\b", r"\belse\s*:\s*\n", "guard clauses vs nesting"),
 ("py_join_concat", "Python", "str.join", "+ concatenation",
  "build at least 5 strings from parts with ''.join([...]) or ', '.join(...)",
  "rewrite every join call as explicit + concatenation of the parts; change nothing else",
  r"\.join\(", r"\+\s*['\"]|['\"]\s*\+", "join vs concatenation"),
 ("py_with_open", "Python", "with open(...)", "open() / close()",
  "read or write at least 4 files using with open(...) as f:",
  "rewrite every with-open block as f = open(...) ... f.close(); change nothing else",
  r"\bwith\s+open\(", r"=\s*open\(|\.close\(\)", "context manager vs explicit close"),
 ("py_enumerate", "Python", "enumerate", "range(len())",
  "iterate with index using enumerate(...) at least 6 times",
  "rewrite every enumerate loop as for i in range(len(x)) with x[i] inside; change nothing else",
  r"\benumerate\(", r"range\(len\(", "enumerate vs range(len)"),
 ("py_self_name", "Python", "self", "this",
  "write at least 3 classes with methods that use self at least 10 times",
  "rename self to this everywhere (parameters and uses); change nothing else",
  r"\bself\b", r"\bthis\b", "self vs this"),
 # ---- formatting
 ("py_indent", "Python", "4-space indentation", "2-space indentation",
  "indent with 4 spaces (at least 10 indented lines)",
  "re-indent the whole file with 2 spaces per level; change nothing else",
  r"\n {4}\S", r"\n {2}\S", "indentation width"),
 ("py_tabs", "Python", "spaces", "tabs",
  "indent with 4 spaces (at least 10 indented lines)",
  "replace every 4-space indentation level with a tab character; change nothing else",
  r"\n {4}\S", r"\n\t", "tabs vs spaces"),
 ("c_braces", "C", "K&R braces", "Allman braces",
  "write at least 6 blocks (functions, if, for, while) with the opening brace on the same line as the statement",
  "move every opening brace to its own line (Allman style); change nothing else",
  r"\)\s*\{\s*\n", r"\)\s*\n\s*\{", "brace placement"),
 ("operator_spaces", "Python", "spaces around operators", "no spaces around operators",
  "put single spaces around = + - * / and comparison operators (at least 10 operators)",
  "remove the spaces around every binary operator (a = b + c -> a=b+c); change nothing else",
  r"\w\s(?:==|!=|<=|>=|=|\+|-|\*|/|<|>)\s\w", r"\w(?:==|!=|<=|>=|=|\+|-|\*|/|<|>)\w", "operator spacing"),
 ("comma_space", "Python", "space after comma", "no space after comma",
  "write at least 10 argument lists or literals with several elements separated by ', '",
  "remove the space after every comma (f(a, b) -> f(a,b)); change nothing else",
  r",\s\S", r",[^\s,]", "comma spacing"),
 ("line_wrap", "Python", "wrapped long calls", "single long lines",
  "write at least 5 calls or definitions with many arguments, wrapped Black-style (one argument per line, closing bracket on its own line)",
  "join every wrapped call/definition into a single long line; change nothing else",
  r"\(\s*\n", r"[^\n]{100,}", "line wrapping"),
 ("blank_lines", "Python", "two blank lines between defs", "one blank line",
  "define at least 6 top-level functions separated by two blank lines",
  "separate the top-level functions by a single blank line instead of two; change nothing else",
  r"\n\n\n(?:def|class)\b", r"[^\n]\n\n(?:def|class)\b", "blank lines between definitions"),
 # ---- comments / docs
 ("docstring_style", "Python", "Google docstrings", "NumPy docstrings",
  "give at least 5 functions Google-style docstrings with Args: and Returns: sections",
  "convert every docstring to NumPy style (Parameters / Returns headers underlined with dashes, 'name : type' lines); change nothing else",
  r"\bArgs:\s*\n|\bReturns:\s*\n", r"\bParameters\s*\n\s*-{3,}|\bReturns\s*\n\s*-{3,}", "docstring style"),
 ("docstring_quotes", "Python", "triple double quotes", "triple single quotes",
  "give at least 6 functions docstrings delimited by triple double quotes",
  "change every docstring delimiter to triple single quotes; change nothing else",
  r'"""', r"'''", "docstring quote style"),
 ("comment_case", "Python", "capitalised comments", "lowercase comments",
  "write at least 8 line comments that start with a capital letter",
  "start every comment with a lowercase letter instead; change nothing else",
  r"#\s+[A-Z]", r"#\s+[a-z]", "comment capitalisation"),
 ("comment_language", "Python", "English comments", "Spanish comments",
  "write at least 8 line comments in English",
  "translate every comment into Spanish; change nothing else in the code",
  r"#\s+[^\n]*\b(?:the|and|for|of|to|is|if|return|check|compute|get|set|with|each|this|that|when|from|into|value|list)\b", r"#\s+[^\n]*\b(?:el|la|los|las|de|para|que|si|con|y|es|del|una|un|cada|este|esta|valor|lista|devuelve|calcula|comprueba)\b", "comment language"),
 ("c_comment_style", "JavaScript", "// comments", "/* */ comments",
  "write at least 8 line comments with //",
  "convert every // comment into a /* ... */ block comment; change nothing else",
  r"//", r"/\*", "comment delimiters"),
 # ---- other languages
 ("sql_keyword_case", "SQL", "uppercase keywords", "lowercase keywords",
  "write the SQL keywords in uppercase (SELECT, FROM, WHERE, JOIN, GROUP BY, ORDER BY, AS, ON, AND) in at least 4 queries",
  "write every SQL keyword in lowercase; change nothing else",
  r"\b(?:SELECT|FROM|WHERE|JOIN|GROUP|ORDER|INSERT|UPDATE|DELETE|LIMIT|INNER|LEFT|HAVING|VALUES|SET|AND|OR|AS|ON|BY)\b", r"\b(?:select|from|where|join|group|order|insert|update|delete|limit|inner|left|having|values|set)\b", "keyword case"),
 ("sql_join_style", "SQL", "explicit JOIN ... ON", "implicit comma join",
  "write at least 4 queries that combine tables with explicit JOIN ... ON clauses",
  "rewrite every explicit JOIN as an implicit join (FROM a, b WHERE a.id = b.a_id); change nothing else",
  r"\bJOIN\b", r"\bFROM\s+\w+(?:\s+\w+)?\s*,\s*\w+", "join style"),
 ("r_assignment", "R", "<- assignment", "= assignment",
  "assign with <- at least 10 times",
  "replace every <- assignment with =; change nothing else",
  r"<-", r"^\s*[\w.]+\s*=\s*[^=]", "assignment operator"),
 ("rust_question", "Rust", "? operator", "explicit match",
  "propagate errors with the ? operator at least 5 times",
  "rewrite every use of ? as an explicit match with Ok(v) => v and Err(e) => return Err(e.into()); change nothing else",
  r"\)\?", r"\bmatch\b[^\n]*\{|Err\(\w+\)\s*=>", "? vs match"),
 ("bash_test", "Bash", "[[ ]] tests", "[ ] tests",
  "write at least 6 conditions with [[ ... ]]",
  "rewrite every [[ ... ]] test as a POSIX [ ... ] test (with the same content); change nothing else",
  r"\[\[", r"(?<!\[)\[\s(?!\[)", "test brackets"),
 ("bash_subst", "Bash", "$(...) substitution", "backtick substitution",
  "use command substitution $(...) at least 6 times",
  "rewrite every $(...) as backtick substitution `...`; change nothing else",
  r"\$\(", r"`", "command substitution"),
 ("css_shorthand", "CSS", "long hex / 0px", "short hex / 0",
  "write at least 8 declarations with 6-digit hex colors (#ffffff, #336699) and zero lengths with units (0px, 0em)",
  "shorten every color that can be shortened (#ffffff -> #fff) and write every zero length without unit (0px -> 0); change nothing else",
  r"#[0-9a-fA-F]{6}\b|\b0(?:px|em|rem)\b", r"#[0-9a-fA-F]{3}\b(?![0-9a-fA-F])|:\s*0\s*;", "CSS shorthand"),
 ("php_array", "PHP", "[] arrays", "array() arrays",
  "create at least 8 arrays with the short syntax [ ... ]",
  "rewrite every array literal with the long syntax array( ... ); change nothing else",
  r"=\s*\[|\(\s*\[|,\s*\[", r"\barray\(", "array syntax"),
]
assert len(CODE_SPECS) == 60, len(CODE_SPECS)

CODE_FAMILIES = []
for name, lang, nat, alt, gen_hint, rewrite, nat_rx, alt_rx, ignore in CODE_SPECS:
    fam = MLFamily(name, lang.lower(), lang, f"{lang} in the natural style: {nat}", nat, alt, [], gen_hint, "", ignore, src_lang="Task")
    fam.domain = "code"; fam.rewrite = rewrite; fam.gen_hint = gen_hint
    fam.prop = _code_property(name, nat_rx, alt_rx, nat, alt)(); PROPS[name] = fam.prop
    CODE_FAMILIES.append(fam); ML_FAMILIES.append(fam); ML_FAMILY[name] = fam
CODE_FAMILY = {f.name: f for f in CODE_FAMILIES}
CODE_LANGS = sorted({f.tgt_lang for f in CODE_FAMILIES})

if __name__ == "__main__":
    for f in CODE_FAMILIES:
        print(f"{f.name:20s} {f.tgt_lang:11s} nat={f.nat!r:40s} alt={f.alt!r}")
    print(len(CODE_FAMILIES), "families;", CODE_LANGS)
