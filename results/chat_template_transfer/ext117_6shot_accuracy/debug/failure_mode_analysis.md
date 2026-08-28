# Why the chat template underperforms Q:/A: — failure-mode analysis notes

**Audience:** future humans and AIs picking up the chat-template-transfer branch. Read this
before interpreting the accuracy numbers or designing follow-ups. Companion file:
`sampled_responses.md` (readable example generations, incl. one full rendered prompt per format).

**Setting** (see `../../README.md` for the full protocol): Qwen2.5-7B-Instruct, 6-shot, 117-task
extended pool, 50 prompts/task, T=1.0 sampled, first-line/strip/exact-match. Arms: `plain`
(Q:/A:), `chat_no_system`, `chat_blank_system`. Same demos/queries in every arm.

## Headline

| arm | mean acc | tasks pruned (<30%) |
|---|---|---|
| plain Q:/A: | **0.708** | 13 |
| chat, no system block | 0.629 | 23 |
| chat, blank system prompt | 0.615 | 23 |
| (GPT-J plain, reference) | 0.416 | 48 |

The chat template genuinely hurts (same model, same demos). `chat_no_system` beats
`chat_blank_system` slightly but consistently (paired +.014, t=2.65): the system block cues the
assistant persona a bit harder.

## The two failure modes

Classified by a heuristic on the chat errors (verbose/conversational vs terse), spot-verified by
reading generations (`sampled_responses.md` has a dedicated examples section):

- **Mode A — persona override.** The model answers the final user turn as a fresh chat message
  and never applies the demo mapping: q=`april` (gold `no`) → *"Sure! April is the fourth month
  of the year"*. NOT style-wrapping of a correct answer: crediting verbose responses only when
  the gold string actually appears in the generation recovers almost nothing (mean .629 → .649;
  23 → 22 pruned). The verbose answers mostly do not contain the answer — task abandonment,
  not presentation.
- **Mode B — loose rule binding despite format compliance.** Terse, in-register, wrong: the
  label convention/granularity/rule binds imprecisely. `puffin` → `bird` (label set is
  animal/plant/object), `permettre` → `permit` (demos use `allow`-style), first LETTER instead
  of first VOWEL, counts off by one, `armchair` → `armchair` (echo instead of `arm`).

## Occurrence rates (chat_no_system unless noted)

- All 2,168 chat errors: 69% B / 30% A / 1% case-only. But plain's 1,711 errors are 91% B —
  terse-wrong is what ordinary ICL failure looks like in any format.
- Net vs plain: chat ADDS +551 mode-A errors (plain has 142 total) while mode B roughly nets
  out (+206 on some tasks, −268 on others). **The aggregate accuracy gap is carried by mode A;
  mode B redistributes across tasks.**
- Prompt-level disagreements (plain ✓, chat ✗; n=813): 38% A / 60% B / 2% case — per prompt the
  two modes contribute comparably; B-flips are offset by ~356 reverse flips (chat ✓, plain ✗),
  A-flips are pure loss.
- Tasks pruned under chat but kept under plain (11/arm): 37% A / 63% B overall, and **per task
  nearly pure one mode or the other**: ~100% A on the yes/no judgment tasks
  (`ends_with_ing`, `contains_letter_e`), ~90–100% B on the counting/char tasks (`count_zeros`,
  `count_consonants`, `word_length`, `first_vowel`); mixed on `person-instrument`, `synonym`,
  `next_in_group`.

## "Adjusting for the persona" does NOT rescue the chat template

- Scoring **all** mode-A errors as correct brings pruned counts to 12/11 (≈ plain's 13) and mean
  acc to ~.74–.76 — but this is an upper bound that assumes every conversational response
  reflects competence, which the gold-in-text check refutes.
- The defensible adjustment (credit verbose answers containing the gold) leaves the picture
  essentially unchanged: mean .649 vs plain .708, 22 tasks pruned vs 13.
- Even under the generous bound, the pure-mode-B counting family (`count_consonants`,
  `count_zeros`, `word_length`) stays dead — mode B is not a persona artifact at all.

**Bottom line: the chat persona is a big, visible failure channel, but adjusting for it still
leaves a real performance deficit — the chat format also degrades how precisely the demo-defined
task binds (mode B).**

## Which tasks are hurt (characteristic, not universal)

Delta (chat_no_system − plain): mean −.078; 46/117 tasks ≈ equal (±.05), 41 worse, 16 much worse
(< −.20), 14 better (> +.05).

- **Crushed lanes** (−.11 to −.25): orthographic/char-level, word-property judgments
  (`article_choice`, `verb_tense_label`), lexical/morphology mappings, translation. Shared
  trait: the gold label is nothing an assistant would naturally reply to the final message
  alone — producing it REQUIRES the demo-bound rule (nobody answers "eddying" with "yes").
- **Unhurt/helped lanes** (0 to +.07): arithmetic, comparison, dates, digit queries, numeric
  sequences. The query alone nearly determines the task, so instruction-following substitutes
  for demo binding (`last_digit` .76 → .94).
- Pruned-set relation: 12 of plain's 13 pruned tasks are also chat-pruned; the exception is
  `capitalize_last_letter` (plain .26 → chat .40/.50), a char-level task the instruct pathway
  happens to handle directly.

## Interpretation (labelled: reading of the data, not a proven mechanism)

Under the chat template, the demonstration turns exert weaker influence on the final answer
than the same text as one Q:/A: stream. Graded, not binary: the surface channel (register,
terseness, domain) still transfers; the precise task-identification channel (which exact
mapping/convention) weakens; on persona-prone tasks it loses entirely to
answer-the-user-directly behavior. Candidate mechanisms (indistinguishable from generations
alone): instruct training treating prior turns as "history" rather than pattern evidence, vs
turn-boundary special tokens disrupting induction-style copying.

## Open follow-ups

1. **Constrained scoring** (cheap): logprob of gold vs alternatives at the answer slot, chat vs
   plain — separates "capability suppressed at sampling" from "task identification impaired".
2. Longer `max_new_tokens` + judge on mode-A tasks (current 12-token cap truncates verbose
   answers; makes the gold-in-text bound pessimistic).
3. The mechanistic version: do the read feature (label-token mean) and write feature (FV) still
   form under chat formatting? Requires porting head selection etc. to Qwen2.5 — the natural big
   next step for this branch.

## Provenance

Numbers computed from the stored per-prompt records in
`artifacts/chat_template_transfer/ext117_6shot/<arm>/<task>.json` (every generation kept).
Mode classifier heuristic: error is "A/verbose" if pred exceeds the gold's word count by >2 or
the generation opens conversationally/asks a question; exact figures are approximate to that
heuristic. Examples dump: `src/eval_scripts/dump_chat_template_debug_examples.py`. Accuracy
tables/plots: `src/eval_scripts/plot_chat_template_ext117.py`. Analysis date: 2026-08-28.
