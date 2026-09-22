# Group A rule validation — status (2026-09-22)

Module: `src/sandbox/style_translation/code_rules_a.py` (RULES dict, 11 families). Validation: `rules8/validate_a.py`; approved-pair diffs: `approved_diffs_a.txt`.
"identical" = rule output byte-equal to the stored alternative twin; "approved" = pairs with `reviewed_by` (Opus 5 + GPT-5 accepted).

| family | status | approved identical | all-200 identical / differs / declined | counted ≥5 (20 docs) | notes |
|---|---|---|---|---|---|
| py2_print | ENABLE | 10/11 (1 declined: multi-line print call) | 14 / 185 / 1 | 20/20 | the 185 old differs = Gemini twins also replaced f-strings by `%` (scope creep); rule keeps everything else byte-identical. Declines: print used as an expression, keyword args (end=/sep=/file=), starred args, multi-line calls. Alt twin cannot be parsed under Py3 (checked: tokenises). Generation hint must forbid f-strings so the alt is valid Python 2. |
| py2_iter | ENABLE | 148/148 | 188 / 12 / 0 | 19/20 | 12 old differs = Gemini scope creep (f-string→format, added hints). Converts every `range(` call and `.items()/.keys()/.values()` no-arg calls; declines if nat already uses xrange/iter*. |
| py2_except | ENABLE | 9/9 | 200 / 0 / 0 | 20/20 | `except E as e:` → `except E, e:`; declines multi-line except types. Same f-string caveat as py2_print. |
| py_builtin_generics | ENABLE | 29/31 | 31 / 169 / 0 | 20/20 | differs = import-line placement/order only (rule follows the approved convention: after the last top-level import, else after the docstring, else at top). The 2 approved differs: a bare `-> dict:` converted to `Dict` by the stored twin (not a generic; rule leaves it), one blank line. Only Subscript nodes are converted, never `dict(...)` calls. Declines when typing generics already appear in nat. |
| py_optional | ENABLE | 49/54 | 56 / 96 / 48 | 20/20 | 5 approved differs = separate vs merged typing import line; 48 declines = nat already contains `Optional[` (mixed pole; those docs are regenerated anyway). Declines multi-line unions. |
| py_type_hints | ENABLE | 45/45 | 195 / 2 / 3 | 20/20 | removes arg annotations, return arrows, annotations of AnnAssign with a value (attribute targets too); declines bare `x: int` (3 docs). Keeps typing imports (approved convention). |
| py_paren_if | ENABLE | 51/52 | 189 / 11 / 0 | 20/20 | `if __name__ == "__main__":` left unwrapped (6 of 7 approved pairs do so). Old differs = stored twins with `) :` spacing, partial wrapping, or backslash continuations; rule wraps the whole condition. Declines when a nat condition is already wholly parenthesised. AST-verified. |
| py_literal_ctor | ENABLE | 39/39 | 191 / 9 / 0 | 18/20 | converts every empty `{}` / `[]` (Load context) incl. `return []`; 9 old differs = stored twins converting non-empty literals / `()` (scope creep). 2 counted failures = docs with < 5 empties (regenerate). |
| py_is_none | ENABLE | 200/200 | 200 / 0 / 0 | 20/20 | token-exact; declines if nat already has `== None`/`!= None`. |
| py_not_in | ENABLE | 200/200 | 200 / 0 / 0 | 20/20 | `x not in y` → `not x in y`, AST-verified (UnaryOp(Not, Compare(In))); declines chained comparisons or a nat that already spells `not x in y`. |
| py_self_name | ENABLE | 3/3 | 196 / 4 / 0 | 20/20 | `\bself\b` → `this` everywhere incl. comments/strings (stored convention); `self_x` identifiers untouched (4 old differs where Gemini renamed them). Declines if `this` is already a code name. |

Not in scope / noted: none of the rules touches strings, comments or docstrings except py_self_name (by the family's stored convention).


# Group A rule validation

## py2_print
- approved pairs: {'identical': 10, 'declined': 1}
- all 200 pairs: {'differs': 185, 'identical': 14, 'declined': 1}
- counted check (20 docs): 20/20 pass, mean counted 9.2

- differs py2_print__t001 (reviewed=False): stored(-) vs rule(+)
    -    print "Original string received: %s" % s
    +    print f"Original string received: {s}"
    -    print "Starting index for iteration: %s" % index
    +    print f"Starting index for iteration: {index}"
    -        print "Current character: %s" % s[index]
    +        print f"Current character: {s[index]}"
- differs py2_print__t002 (reviewed=False): stored(-) vs rule(+)
    -    print "Starting vowel count for text: '%s'" % text
    +    print f"Starting vowel count for text: '{text}'"
    -    print "Converted text to lowercase: '%s'" % lower_text
    +    print f"Converted text to lowercase: '{lower_text}'"
    -        print "Checking character: '%s'" % char
    +        print f"Checking character: '{char}'"
- differs py2_print__t003 (reviewed=False): stored(-) vs rule(+)
    -    print "Input received: n = %s" % n
    +    print f"Input received: n = {n}"
    -        print "Recursive case: n is %s" % n
    +        print f"Recursive case: n is {n}."
    -        print "Intermediate calculation: %s * factorial(%s) = %s" % (n, n-1, result)
    -        print "Returning result for n=%s: %s" % (n, result)
- differs py2_print__t004 (reviewed=False): stored(-) vs rule(+)
    -        print "Processing character: '%s'" % char
    +        print f"Processing character: '{char}'"
    -    print "Cleaned text: '%s'" % processed_text
    +    print f"Cleaned text: '{processed_text}'"
    -    print "Reversed text: '%s'" % reversed_text
    +    print f"Reversed text: '{reversed_text}'"
- declined py2_print__t071: declined

## py2_iter
- approved pairs: {'identical': 148}
- all 200 pairs: {'identical': 188, 'differs': 12}
- counted check (20 docs): 19/20 pass, mean counted 8.7

- differs py2_iter__t009 (reviewed=False): stored(-) vs rule(+)
    -    my_other_dict = {"val_{}".format(i): val for i, val in enumerate(numbers)}
    +    my_other_dict = {f"val_{i}": val for i, val in enumerate(numbers)}
    -        some_items["item_{}".format(i)] = num
    +        some_items[f"item_{i}"] = num
- differs py2_iter__t031 (reviewed=False): stored(-) vs rule(+)
    -        deltas['dx%s' % key_suffix] = coords['x%s' % key_suffix]
    -        deltas['dy%s' % key_suffix] = coords['y%s' % key_suffix]
    +        deltas[f'dx{key_suffix}'] = coords[f'x{key_suffix}']
    +        deltas[f'dy{key_suffix}'] = coords[f'y{key_suffix}']
    -        term_name = 'term%s' % (j+1)
    +        term_name = f'term{j+1}'
- differs py2_iter__t065 (reviewed=False): stored(-) vs rule(+)
    -    check_dict = {"item_%s" % i: num for i, num in enumerate(final_list)}
    +    check_dict = {f"item_{i}": num for i, num in enumerate(final_list)}
- differs py2_iter__t069 (reviewed=False): stored(-) vs rule(+)
    -def is_armstrong(number):
    +def is_armstrong(number: int) -> bool:

## py2_except
- approved pairs: {'identical': 9}
- all 200 pairs: {'identical': 200}
- counted check (20 docs): 20/20 pass, mean counted 7.2


## py_builtin_generics
- approved pairs: {'identical': 29, 'differs': 2}
- all 200 pairs: {'differs': 169, 'identical': 31}
- counted check (20 docs): 20/20 pass, mean counted 8.4

- differs py_builtin_generics__t001 (reviewed=False): stored(-) vs rule(+)
    -from typing import List, Dict, Tuple
    +from typing import Dict, List, Tuple
    +
- differs py_builtin_generics__t002 (reviewed=False): stored(-) vs rule(+)
    +
- differs py_builtin_generics__t003 (reviewed=False): stored(-) vs rule(+)
    -from typing import List, Dict, Tuple
    +from typing import Dict, List, Tuple
    +
- differs py_builtin_generics__t004 (reviewed=False): stored(-) vs rule(+)
    -from typing import List, Dict, Tuple
    +from typing import Dict, List

## py_optional
- approved pairs: {'differs': 5, 'identical': 49}
- all 200 pairs: {'differs': 96, 'declined': 48, 'identical': 56}
- counted check (20 docs): 20/20 pass, mean counted 6.8

- differs py_optional__t001 (reviewed=False): stored(-) vs rule(+)
    +
- differs py_optional__t003 (reviewed=False): stored(-) vs rule(+)
    +
- differs py_optional__t004 (reviewed=True): stored(-) vs rule(+)
    -from typing import Iterable
    -from typing import Optional
    +from typing import Iterable, Optional
- differs py_optional__t006 (reviewed=False): stored(-) vs rule(+)
    +
- declined py_optional__t002: declined
- declined py_optional__t005: declined
- declined py_optional__t016: declined
- declined py_optional__t018: declined

## py_type_hints
- approved pairs: {'identical': 45}
- all 200 pairs: {'identical': 195, 'declined': 3, 'differs': 2}
- counted check (20 docs): 20/20 pass, mean counted 11.6

- differs py_type_hints__t137 (reviewed=False): stored(-) vs rule(+)
    -        self.items: list = []
    +        self.items = []
    -    q: Queue = Queue()
    +    q = Queue()
- differs py_type_hints__t154 (reviewed=False): stored(-) vs rule(+)
    +from typing import List, Union
- declined py_type_hints__t051: declined
- declined py_type_hints__t059: declined
- declined py_type_hints__t101: declined

## py_paren_if
- approved pairs: {'identical': 51, 'differs': 1}
- all 200 pairs: {'identical': 189, 'differs': 11}
- counted check (20 docs): 20/20 pass, mean counted 9.3

- differs py_paren_if__t010 (reviewed=False): stored(-) vs rule(+)
    -if (__name__ == "__main__") :
    +if __name__ == "__main__" :
- differs py_paren_if__t011 (reviewed=False): stored(-) vs rule(+)
    -    elif (operator == '/'):
    -        if (num2 == 0):
    +    elif (operator == '/') :
    +        if (num2 == 0) :
- differs py_paren_if__t028 (reviewed=False): stored(-) vs rule(+)
    -if (__name__ == '__main__'):
    +if __name__ == '__main__':
- differs py_paren_if__t039 (reviewed=False): stored(-) vs rule(+)
    -    if (year % 4 == 0) and (year % 100 != 0):
    +    if ((year % 4 == 0) and (year % 100 != 0)):

## py_literal_ctor
- approved pairs: {'identical': 39}
- all 200 pairs: {'identical': 191, 'differs': 9}
- counted check (20 docs): 18/20 pass, mean counted 6.2

- differs py_literal_ctor__t053 (reviewed=False): stored(-) vs rule(+)
    -    mapping = dict(
    -        )
    -    mapping[")"] = "("
    -    mapping["}"] = "{"
    -    mapping["]"] = "["
    +    mapping = {
- differs py_literal_ctor__t055 (reviewed=False): stored(-) vs rule(+)
    -    metadata = tuple()
    +    metadata = ()
- differs py_literal_ctor__t077 (reviewed=False): stored(-) vs rule(+)
    -    valid_schemes = list(['http://', 'https://'])
    +    valid_schemes = ['http://', 'https://']
- differs py_literal_ctor__t090 (reviewed=False): stored(-) vs rule(+)
    -        metadata = dict({'source': 'log_parser'})
    +        metadata = {'source': 'log_parser'}
    -        temp_dict = dict({'status': 'success'})
    +        temp_dict = {'status': 'success'}
    -        metadata = dict({'error': 'invalid_format'})
    +        metadata = {'error': 'invalid_format'}

## py_is_none
- approved pairs: {'identical': 200}
- all 200 pairs: {'identical': 200}
- counted check (20 docs): 20/20 pass, mean counted 6.6


## py_not_in
- approved pairs: {'identical': 200}
- all 200 pairs: {'identical': 200}
- counted check (20 docs): 20/20 pass, mean counted 5.5


## py_self_name
- approved pairs: {'identical': 3}
- all 200 pairs: {'identical': 196, 'differs': 4}
- counted check (20 docs): 20/20 pass, mean counted 7.0

- differs py_self_name__t017 (reviewed=False): stored(-) vs rule(+)
    -    this_processor = FileProcessor(filepath)
    -    return this_processor.get_line_count()
    +    self_processor = FileProcessor(filepath)
    +    return self_processor.get_line_count()
- differs py_self_name__t033 (reviewed=False): stored(-) vs rule(+)
    -            this.calculate_and_and_store_grade()
    +            this.calculate_and_store_grade()
- differs py_self_name__t059 (reviewed=False): stored(-) vs rule(+)
    -        this_overlaps_other_start = this.start < other_event.end
    -        this_overlaps_other_end = this.end > other_event.start
    -        return this_overlaps_other_start and this_overlaps_other_end
    +        self_overlaps_other_start = this.start < other_event.end
    +        self_overlaps_other_end = this.end > other_event.start
    +        return self_overlaps_other_start and self_overlaps_other_end
- differs py_self_name__t191 (reviewed=False): stored(-) vs rule(+)
    -            this_c_value = value
    -            return this_c_value
    +            self_c_value = value
    +            return self_c_value
    -            this_f_to_c = (value - this.f_to_c_offset) * this.f_to_c_factor
    -            return this_f_to_c
