# WORKLOG

Coordination log for in-flight experimental work on the Function Vectors repo.
Newest entries at top. One stream per active line of work.

---

## 2026-06-15 — Stream E follow-up: stable-rank + pairwise-cos per layer, all vs GPT-4-correct

**Owner:** Coordinator (tmux "oneshot-geometry"). **Status:** DONE.

**What:** `src/eval_scripts/plot_stable_rank_by_layer_byjudge.py`. Per layer, per token position, per task
pair: stable rank (Σσ²/σ₁², unit-normalized rows) and mean pairwise cosine of the stacked
difference-vector matrix D=act(f1)−act(f2). 2 figures (label, final), each [stable rank | mean cos],
with per-pair lines × {all (solid), GPT-4-correct-both-functions (dashed)}. Computed on the CORRECTED
graded capture (shared-input query). Overwrote `fig_compare_{label,final}.png` in BOTH
`results/oneshot_paired_analysis/` and `results/oneshot_paired_diff_geometry/` (the latter's old 4-pair
shared-OUTPUT versions superseded; its CSVs left as provenance). Series json in oneshot_paired_analysis.

**Stable rank, label token (all / correct):** antonym_synonym L9 4.85/3.58, L11 5.38/3.83;
next_number_prev_number L9 1.55/1.42, L11 1.61/1.47. Dips mid-layer (~L9) both pairs; numbers ≈ rank-1
(one dominant ±1 axis), words higher (~5–9). Mean-cos panel is the mirror (peaks mid-layer); correct >
all everywhere.

**CAVEAT (flagged):** stable rank depends on #rows W; correct subset has far fewer words (22 vs 544;
105 vs 198) → stable rank is biased DOWN for the correct line, so "correct < all" in the left panel is
partly a sample-size artifact. The mean-pairwise-cosine panel is W-robust and IS the trustworthy
all-vs-correct signal (correct more co-directional). Fix if needed: size-matched subsampling of all-words.
**Blockers:** None.

---

## 2026-06-15 — Stream E follow-up: pairwise diff-vector cosine histograms by task pair & judge

**Owner:** Coordinator (tmux "oneshot-geometry"). **Status:** DONE.

**What:** `src/eval_scripts/plot_pairwise_cos_hist_byjudge.py`. Histograms of pairwise cosine among the
per-word function-difference vectors D = act(f1) − act(f2) (unit-normalized), per TASK PAIR and per
token position. 4 panels: rows {antonym_synonym, next_number_prev_number} × cols {label, final query}.
Each overlays ALL words (grey) vs the GPT-4-correct subset (green) = words judged top-1 correct under
BOTH functions of the pair. Reads graded+tagged captures. Output
`results/oneshot_paired_analysis/fig_pairwise_diffcos_hist_L11_byjudge.png` (+ stats json).

**Mean pairwise cos (L11): all / correct (n_correct_both):**
- antonym_synonym label 0.152/0.203, final 0.116/0.174 (n=22)
- next_number_prev_number label 0.593/0.671, final 0.452/0.523 (n=105)

**FINDINGS:** (1) numbers ≫ words — the ±1 difference vectors are strongly co-directional (~0.5–0.6)
vs the high-dim word-meaning contrast (~0.12–0.15); echoes the stable-rank result (numbers ≈ rank-1).
(2) label > final in both pairs (function-difference more coherent at the demo label). (3) correct >
all in EVERY panel — correctly-handled words have more consistent function-difference directions
(caveat: antonym_synonym both-correct n=22, noisier). "correct" = both functions judge_top1 (can switch
to either/per-function). **Next:** geometry on judge-correct subsets. **Blockers:** None.

---

## 2026-06-15 — Stream E follow-up: cosine scatter (label vs final) colored by GPT-4 judge

**Owner:** Coordinator (tmux "oneshot-geometry"). **Status:** DONE.

**What:** Remade the cosine scatter on the CORRECTED graded capture (shared-input query, judge-tagged).
New `src/eval_scripts/plot_cos_label_vs_final_byjudge.py`. One point per shared word (n=544): x =
cos(antonym act, synonym act) at the demo LABEL token, y = same cos at the FINAL query token (both
positions already captured). Colored 4 ways by GPT-4 judge_top1 per function: both correct (22),
antonym-only (128), synonym-only (56), neither (338) — reconciles with judge totals (ant 150, syn 78).
Output overwrites `results/oneshot_paired_analysis/antonym_synonym/fig_cos_label_vs_final_L11_byjudge.png`
(+ points `.json`).

**FINDINGS (L11):** (1) ALL points above y=x — f1/f2 activations are MORE aligned at the final query
token (mean cos ~0.95) than at the label token (~0.89); the two functions converge by the query
position. (2) Judge correctness does NOT separate in this cosine space — correct points are
interspersed, per-category means near-identical (both .88/.95, neither .89/.96). So f1–f2 representational
similarity does not predict answer correctness. **Next:** same plot for numbers if wanted; geometry on
judge-correct subset. **Blockers:** None.

---

## 2026-06-15 — Stream E follow-up: SAME pipeline for next_number|prev_number

**Owner:** Coordinator (tmux "oneshot-geometry"). **Status:** DONE.

**What:** Replicated the corrected-capture + grade + GPT-4 judge + activation-tag pipeline for the
`next_number_prev_number` pair (number WORDS). Generalized the scripts:
- `capture_and_grade_oneshot_paired.py`: added the pair to TASK_PAIRS + `--allow_multitoken_label`
  (number words are mostly multi-token: only 26/198 shared outputs are single-token). With it, the
  demo-label pool = all 198 shared outputs and the source is the LAST label token (identical across
  f1/f2); `expected_src_id` now compares the LAST token of " "+w. Query from shared INPUT (200, gold
  under both). → 198 paired units, `results/oneshot_paired_graded/next_number_prev_number/` (2 shards,
  792 rows).
- `judge_oneshot_paired.py`: added `next_number`/`prev_number` judge prompts (answer denotes input±1 in
  ANY notation; SAME number doesn't count) + multi-word answer extraction (`one hundred one`, not just
  the first word). Greedy-generates the FULL answer (max_new_tokens=8).

**Results (n=198/task):**
| task | GPT-4 judge top-1 (full answer) | first-token top-1 | gold exact | copied |
|---|---|---|---|---|
| next_number | **0.818** (162/198) | 0.894 | 0.813 | 24 |
| prev_number | **0.621** (123/198) | 0.859 | 0.621 | 27 |

**KEY DIFFERENCE vs antonym/synonym:** here the GPT-4 judge is LOWER than first-token top-1 (inverted)
because **first-token OVER-counts for numbers** — gold "one hundred one" shares first token "one" with a
wrong "one hundred"/"one hundred two", and the copy-the-query cases (q=fifty, gold=fifty-one, model=fifty)
score first-token-correct but are wrong. So for numbers you MUST grade the full multi-token answer (judge
≈ gold-exact-match, 0.818/0.621); first-token is unreliable here. next > prev (GPT-J better at +1 than −1).

**Tagged** 792/792 activation rows + 396 grading rows with `judge_top1` (via
`tag_oneshot_activations_judge.py`). Activations filter by `top1/2/3` (first-token) and `judge_top1`
(GPT-4 full-answer). Both pairs now share the same on-disk schema. **Next:** geometry on judge-correct
subsets. **Blockers:** None.

---

## 2026-06-15 — Stream E follow-up: GPT-4-judged top-1 (antonym+synonym) + activation tagging

**Owner:** Coordinator (tmux "oneshot-geometry"). **Status:** DONE.

**What:** First-token exact-match undercounts open-ended tasks (many valid antonyms/synonyms). Judged
GPT-J's actual greedy top-1 answer with GPT-4.1 for BOTH tasks. `src/eval_scripts/judge_oneshot_paired.py`
(supersedes the earlier synonym-only script; one model load, `--function_tasks antonym synonym`):
rebuilds the EXACT prompts from the corrected capture (`results/oneshot_paired_graded/antonym_synonym/
grading.json` — shared-input query), greedy-generates the top-1 word, asks GPT-4.1 if it's a valid
antonym/synonym. **Both judge prompts explicitly state the SAME WORD (or cap/plural/inflection) does
NOT count** (per user); also reject the opposite relation, non-words, topical associates. Verdict key
`"correct"`. Key from `/proc/1/environ`, batched 50/request.

**Results (n=544/task):**
| task | GPT-4 judge top-1 | first-token top-1 | gold exact | copied query |
|---|---|---|---|---|
| antonym | **0.276** (150/544) | 0.232 | 0.232 | 263 (0.48) |
| synonym | **0.143** (78/544) | 0.066 | 0.063 | 339 (0.62) |

**FINDING:** judging the real answer lifts both (antonym +.044, synonym >2×: .143 vs .066) — first-token
scoring undercounts, much more so for synonym (many valid alternatives). The dominant failure is
**copying the query word** (antonym 48%, synonym 62%) — in 1-shot with a single `q→w` demo GPT-J largely
echoes the query; all copies judged false (per same-word rule). Judge accepts non-gold answers
(synonym fearless→brave; gold "daring"). Outputs `results/oneshot_{antonym,synonym}_judge/judged_results.json`.

**ACTIVATION TAGGING:** `src/eval_scripts/tag_oneshot_activations_judge.py` stamped a `judge_top1` bool
into EVERY activation row's metadata (match key (function_task, output_word, query_word); both source &
target roles) in `results/oneshot_paired_graded/antonym_synonym/shard_*.pt` AND grading.json — 2176/2176
rows tagged. So activations now filter three ways: `top1/2/3` (first-token rank) and `judge_top1` (GPT-4
semantic). judge_top1 True rows: antonym 300 (=150 prompts×2 roles), synonym 156 (=78×2).
**Next:** geometry on the new capture; compare stable-rank/cosine on judge_top1-correct subsets. **Blockers:** None.

---

## 2026-06-15 — Stream D: GPT-4-judged rhyme eval scaled to ALL 200 pairs

**Owner:** Coordinator. **Status:** DONE.

**What:** Re-ran the GPT-4.1-judged rhyme eval (Stream C, was n=42 test split) over the FULL
200-pair dataset. Added `--all_queries` mode to `src/eval_scripts/judge_rhyme_generations.py`:
each of the 200 examples is the query in turn, its 10-shot ICL demos drawn randomly leave-one-out
from the other 199 (disjoint, no leakage). Also chunked the judge into batches (`--judge_batch_size`,
default 50) so large n doesn't overflow one request. Default Stream-C path unchanged.

**Command:**
`python src/eval_scripts/judge_rhyme_generations.py --all_queries --output_dir results/rhyme_judge_eval_all200 --judge_batch_size 50`

**Result (n=200):** judged-rhyme accuracy **5/200 = 0.025**; gold exact-match **2/200 = 0.010**;
copied the input verbatim **94/200 = 0.47**. The 5 judge-accepted rhymes: array→delay, ban→man,
catch→attach, command→band, cry→eye (only the two exact-match ones also match gold). Confirms the
n=42 finding at full scale: GPT-J essentially cannot rhyme via 10-shot ICL — ~half the time it just
echoes the query word; near-zero true rhymes. Judge ≈ gold (both ~0–2.5%), so the gold-token metric
was NOT understating competence.

**Files:** added flags to `judge_rhyme_generations.py`; artifacts `results/rhyme_judge_eval_all200/`
(generations.json, judged_results.json, run.log). Deleted the superseded n=42 `results/rhyme_judge_eval/`.

---

## 2026-06-15 — Stream E follow-up: CORRECTED paired 1-shot capture (shared-INPUT query) + grading

**Owner:** Coordinator (tmux "oneshot-geometry"). **Status:** DONE (activations + grading).

**Why the redo:** the original Stream-E capture (`capture_oneshot_paired.py`) sampled the QUERY from
the shared OUTPUT-word pool (just a token to read activations from). That makes the query ill-posed —
only 464/389 of 544 queries even had a gold antonym/synonym (50 had neither; an earlier scoring pass,
`score_oneshot_paired_prompts.py`, got ant .280/.431, syn .113/.270 over that biased 853-prompt
subset). CORRECTED: query is now drawn from the shared INPUT space (words that are a valid input under
BOTH tasks → gold defined under both). Demo label `w` still from shared OUTPUT (paired design unchanged).

**Shared-space sizes (antonym/synonym):** shared INPUT = **1224** (all single-token), shared OUTPUT =
555 (544 single-token, used for the demo label). Input vocab > output vocab for both tasks.

**New script** `src/eval_scripts/capture_and_grade_oneshot_paired.py` (supersedes
`capture_oneshot_paired.py` for graded runs): captures source(demo-label)+target(final-query)
activations AND grades each prompt in-line (one extra forward; **first-token rank**, top-k = rank<k —
per user, first-token only, no multi-token gen). The grade (`gold_first_tok_rank`, `top1/2/3`) is
written into EVERY activation row's metadata → activations filter directly to top-1/2/3-correct.

**DELETED old output-space-query run:** `results/oneshot_paired/antonym_synonym/`,
`results/oneshot_paired_analysis/antonym_synonym/` (+ its stale top-level `fig_*.png`),
`results/oneshot_paired_scored/antonym_synonym/`. `landmark_park` + other pairs untouched.

**Outputs** `results/oneshot_paired_graded/antonym_synonym/`: 6 shards, **2176 rows** (544×2 funcs×2
roles), `index.json`, `grading.json` (per-prompt + model top-1 token), `scores.json`.

**Results (1-shot, first-token, all 544/task now scorable):**
| task | n | top1 | top2 | top3 |
|---|---|---|---|---|
| antonym | 544 | 0.232 | 0.324 | 0.401 |
| synonym | 544 | 0.066 | 0.195 | 0.263 |

LOWER than the old output-space-query slice (ant .280/.431, syn .113/.270) — expected: that slice was
an easier/biased subset (queries that happen to be task outputs, gold-having only). These are the
honest well-posed-query numbers. antonym ≫ synonym persists; synonym ~triples top1→top2 (many valid
synonyms, single gold first-token misses). **Next:** re-run geometry (`analyze_oneshot_geometry.py`)
on the new capture; optionally compare stable-rank/cosine when filtered to top-1-correct prompts.
**Blockers:** None.

---

## 2026-06-15 — Stream G: task-specific + train-pooled FVs & steering for 4 ambiguous tasks

**Owner:** Coordinator (tmux "magnitude/identity FV"). **Status:** RUNNING (pipeline launched).

**What:** For the 4 chosen ambiguous tasks (`magnitude`, `identity`, `count_vowels`,
`count_consonants`): (1) task-specific FVs (own CIE top-10) + (2) train-pooled-head FVs at top-10/20/40
(reuse `results/multitask_aie_heads/multitask_top_aie_heads.pt` = original 20-train pool), then steering
eval (per-layer zero-shot + 10-shot-shuffled) for all.

**Setup:** symlinked the 4 JSONs into `dataset_files/abstractive/` (loader hardcodes abstractive/
extractive; same precedent as paired_tasks). NEW manifest `task_splits/ambiguous_4.json`. NEW driver
`src/eval_scripts/run_ambiguous_fv_pipeline.sh`. Installed `bitsandbytes` (intervention_utils imports
it at module top even though the 4-bit path is unused in fp16).

**GPU/parallelism:** single 24GB card fits ONE GPT-J fp16 (~12GB) → GPU stages serialize. Pipeline:
stage1 = compute_function_vectors (GPU, all 4 tasks, batch 16); stage2 = 3× top-N builds on **CPU in
parallel** (out_proj sums, overlap GPU); stage3 = evaluate_heldout ×{10,20,40} (GPU serial; also
re-evals the task-specific FV each run). Logs in `results/_ambiguous_logs/`.

**CAVEAT (expected):** count_vowels/count_consonants have low ICL accuracy (~0.3 first-token) so the
correct-ICL filter leaves few prompts → low-N/noisy FVs (cf. rhyme/geo low-N). magnitude/identity are
high-acc. Outputs: `results/gptj_fv/<task>/` (task-specific), `results/gptj_fv_multitask_top{10,20,40}_
ambiguous/` (train-pooled), `results/heldout_ambiguous_eval_top{10,20,40}/` (steering).

**DONE (with OOM hiccup).** Stage-1 task-specific FVs built for all 4 (count_* survived the filter:
n_filtered_test magnitude/identity=21, count_vowels/count_consonants=**6** → low-N as predicted).
OOM during the run: the 3 parallel CPU model-loads (~12GB RAM each) + GPU steering load tripped the
OOM-killer (killed steer n=10 + 2 builds). **FIX: re-ran the missing pieces SEQUENTIALLY** (no parallel
model loads) — `results/_ambiguous_logs/rerun_missing.sh`. All complete now: steering top-10/20/40 (4
tasks each), persisted train-pooled FVs top-10/20/40 (4 each), task-specific FVs (4).
**Lesson:** GPT-J CPU loads are ~12GB RAM each; don't run 3 in parallel alongside a GPU load on this box.

**STEERING RESULTS (best-layer zero-shot FV top-1; 10-shot-shuffled in parens) —
`summarize_ambiguous_steering.py`:**

| task | n_test | task-specific | train top10 | top20 | top40 |
|---|---|---|---|---|---|
| magnitude | 21 | 0.62@L12 (0.95) | 0.57@L0 (0.90) | 0.57@L0 (0.95) | 0.57@L0 (0.95) |
| identity | 21 | 1.00@L0 (1.00) | 1.00@L0 (1.00) | 1.00@L0 (1.00) | 1.00@L0 (1.00) |
| count_vowels | 6 | 0.00 (0.83) | 0.00 (0.67) | 0.00 (0.67) | 0.00 (0.83) |
| count_consonants | 6 | 0.17@L14 (0.83) | 0.00 (0.83) | 0.00 (0.83) | 0.00 (0.83) |

**CAVEATS / FINDINGS:**
- **By-layer curves are nearly FLAT** (magnitude ≈0.57=12/21 at every layer, identity 1.0 everywhere),
  argmax often at **L0** (large-norm embedding artifact). → the zero-shot numbers largely reflect BASE
  zero-shot behavior, NOT a clean FV effect. Need the **no-FV baseline** to isolate the causal lift
  (evaluate_heldout doesn't compute it).
- **Zero-shot FV steering is weak/absent for these tasks**: magnitude flat ~0.57, count ≈0. Only the
  10-shot-shuffled (context present) numbers are high (0.83–1.0).
- **task-specific ≈ train-pooled, and top-10≈20≈40** — adding heads barely changes these (contrast the
  29-task finding where n40 helped zero-shot). identity is trivially 1.0 (copy).
- **count_* are low-N (n_test=6) + ~0 zero-shot** — competence/low-N caveat confirmed.

**Next:** run `evaluate_function_vector.py --compute_baseline` for the 4 tasks to get the no-FV
zero-shot/10-shot baselines → report the FV's causal steering DELTA (the meaningful metric).
**Blockers:** None.

---

## 2026-06-14 — Stream F: `ambiguous` task-disambiguation datasets (4 pairs)

**Owner:** Coordinator (tmux "magnitude/identity FV"). **Status:** DONE — all 4 brainstormed pairs generated.

**What:** New dataset family for the task-DISAMBIGUATION study (3 ambiguous ICL demos + 1
differentiating demo + 1 test query). Each pair (f1,f2) AGREES on an overlap region and
DISAGREES on a differentiator region; the two task files share the SAME input set, overlap
entries are byte-identical (input AND output), differ entries share input / split output.

**Pairs (`dataset_files/ambiguous/`):**
- `magnitude.json | identity.json` — n→|n| vs n→n. Overlap = non-neg ints (1..50); differ =
  negatives (-1..-50). **50/50.** Digits ("-5"); switch to words for next/prev_number consistency.
- `past_tense.json | past_participle.json` — verb→past vs verb→participle. Overlap = 50 regular
  verbs (past==participle); differ = 50 irregulars (ate/eaten…). **50/50.** Strongest pair (rich O+D,
  known priors).
- `first_letter.json | last_letter.json` — word→word[0] vs word→word[-1]. Overlap = 50 words with
  first==last char; differ = 50 others. **50/50.** Vocab sourced from repo synonym+antonym inputs
  (4054 words; 184 first==last available), seed=0.
- `capital_city.json | largest_city.json` — country→capital vs →largest. Overlap = 50 (capital IS
  largest); differ = **35** (capital≠largest). **50/35, NOT 50/50** — only ~35 real capital≠largest
  countries exist worldwide and only ~20 are GPT-J-plausible (Naypyidaw/Gitega/Ngerulmud etc. are
  low-freq). RECALL CAVEAT, cf. the geo/element low-N pairs. Trim overlap→35 for a balanced 35/35,
  or restrict differ to the ~20 famous head of the list, in the eval.

**Files:** NEW `dataset_files/generate/create_ambiguous_datasets.py` (one generator, all 4 pairs);
NEW folder `dataset_files/ambiguous/` (8 JSONs).

**Verified:** all pairs share matched inputs; overlap/differ counts as above; differ outputs split
correctly (eat→ate/eaten; vulgar→v/r; United States→Washington/New York).

**3+1+1 EVAL DONE (n=100/task, cross-prompt batched greedy, token-level exact match):** new
`src/eval_scripts/eval_ambiguous_disambiguation.py` (batched generate; reuses
`word_pairs_to_prompt_data`/`create_prompt`; `matches_partner` diagnostic = model produced the OTHER
function's answer). Each prompt = 3 overlap demos + 1 differentiator demo (task's output) + 1
differentiator query (seed 42, paired draws). Results → `results/ambiguous_disambiguation/eval_summary.json`.

| task | acc | partner | neither |
|---|---|---|---|
| magnitude | 0.98 | 0.02 | 0.00 |
| identity | 1.00 | 0.00 | 0.00 |
| past_tense | 0.93 | 0.03 | 0.04 |
| past_participle | **0.36** | 0.60 | 0.04 |
| first_letter | 0.97 | 0.01 | 0.02 |
| last_letter | **0.04** | 0.89 | 0.07 |
| capital_city | 0.57 | 0.37 | 0.06 |
| largest_city | 0.47 | 0.45 | 0.08 |

**FINDING — strong within-pair PRIOR-BIAS asymmetry; one disambiguating demo only redirects the
model in the "easy" direction.** magnitude/identity ~perfect both ways. But the other pairs are
lopsided: the model nails the *prior-aligned* task (past_tense .93, first_letter .97) and largely
ignores the 4th demo for the *anti-prior* task, instead emitting the prior answer (past_participle
.36 with partner .60; last_letter .04 with partner .89). capital/largest both ~chance-ish and noisy
(recall-limited differ tail + capital↔largest confusion). Spot-checks confirm scoring is correct
(last_letter gold 'r' → pred 'f' = first letter; past_participle gold 'shown' → pred 'showed').

**3+2+1 EVAL (n_diff_demos=2; `--n_diff_demos 2`):** → `eval_summary_3plus2plus1.json`. A second
disambiguating demo helps the anti-prior side only partially, and not at all for the strongest prior:

| task | 3+1+1 acc | 3+2+1 acc | partner (3+2+1) |
|---|---|---|---|
| magnitude | 0.98 | 1.00 | 0.00 |
| identity | 1.00 | 1.00 | 0.00 |
| past_tense | 0.93 | 0.98 | 0.02 |
| past_participle | 0.36 | **0.51** | 0.49 |
| first_letter | 0.97 | 0.97 | 0.00 |
| last_letter | 0.04 | **0.02** | 0.88 |
| capital_city | 0.57 | 0.60 | 0.37 |
| largest_city | 0.47 | 0.46 | 0.47 |

**FINDING (k=2 vs k=1 differentiator demos):** the prior-aligned tasks saturate (past_tense→.98,
magnitude→1.0). past_participle improves +.15 (.36→.51, now ~coin-flip vs its prior). **last_letter
does NOT budge (.04→.02, partner .88)** — two demos still can't override the first-letter prior.
capital/largest essentially flat (recall-bound). So a 2nd disambiguating example helps moderate
priors but not the dominant first-letter one; harness now parametrized (`--n_shared_demos`,
`--n_diff_demos`) for a k-sweep.

**3 MORE PAIRS ADDED (2026-06-14, all 50/50):** `round|truncate` (1-dp decimals; overlap frac<.5,
differ frac≥.5), `first_digit|last_digit` (overlap first==last digit; numeric analog of the
first/last-letter prior diagnostic), `american|british` (input=US spelling, american=identity /
british=US→UK convert; overlap=invariant words, differ=50 US/UK variants). All wired into
`eval_ambiguous_disambiguation.py` PAIRS. Not yet eval'd.

**+ reverse|identity_word (50/50):** overlap = 50 palindromes (reverse==identity), differ = 50
ordinary 3–5 letter words (`prose→esorp` vs `prose`). Partner named `identity_word.json` to avoid
clashing with the numeric `identity.json`. Wired into eval PAIRS. Prediction: reversal is hard for
BPE → expect a *competence* failure (high `neither`) not a *prior* failure (high `partner`).

**+ count_vowels|count_consonants (50/50):** overlap = 50 words with #vowels==#consonants
(`able` 2/2), differ = 50 unequal (`silver` 2/4). Vowels=aeiou. Counting task → expect
competence-limited (high `neither`), noisy (small int output space); overlap output also = len/2 so
consistent with several counting rules. Wired into eval PAIRS. (8 pairs total now.)

**3+1+1 EVAL — ALL 8 PAIRS (2026-06-14).** Two scorings, n=100/task, cross-prompt batched:
exact-match (generation) → `eval_summary_3plus1plus1_all.json`; first-answer-token top-1/top-2
(single forward pass, `--scoring topk`) → `eval_topk_3plus1plus1.json`. Added `batched_topk`/
`score_topk` + `--scoring {topk,exact}` to the harness.

| task | top1 | top2 | partner@1 |
|---|---|---|---|
| magnitude | 0.98 | 0.99 | 0.02 |
| identity | 1.00 | 1.00 | 0.00 |
| past_tense | 0.93 | 0.95 | 0.05 |
| past_participle | 0.38 | **0.92** | 0.60 |
| first_letter | 0.97 | 0.99 | 0.01 |
| last_letter | 0.04 | **0.18** | 0.89 |
| capital_city | 0.60 | 0.85 | 0.38 |
| largest_city | 0.48 | 0.76 | 0.48 |
| round | 0.05 | **0.93** | 0.95 |
| truncate | 1.00 | 1.00 | 0.00 |
| first_digit | 0.77 | 0.83 | 0.06 |
| last_digit | 0.16 | **0.37** | 0.51 |
| american | 1.00 | 1.00 | 0.15 |
| british | 0.43 | **0.92** | 0.69 |
| reverse | 0.00 | 0.01 | 0.66 |
| identity_word | 1.00 | 1.00 | 0.00 |
| count_vowels | 0.31 | 0.57 | 0.11 |
| count_consonants | 0.29 | 0.52 | 0.22 |

**FINDINGS (top-2 separates the failure modes):**
- **magnitude/identity** still the only pair solid in BOTH directions (.98/1.00).
- **PRIOR-BIAS pairs (anti-prior side rank-2, not absent):** past_participle .38→**.92**, round .05→**.93**,
  british .43→**.92** all jump at top-2 → the disambiguating demo IS registered (gold is the #2 token)
  but the prior owns #1 (partner@1 .60/.95/.69). The model "knows" the right answer, prior overrides.
- **STRONG-PRIOR pairs (anti-prior side genuinely suppressed):** last_letter .04→.18, last_digit
  .16→.37 — gold not even rank 2. First-position prior is real for BOTH letters and digits, weaker
  for digits (last_digit top2 .37 vs last_letter .18).
- **COMPETENCE failure:** reverse .00/.01 (can't emit the reversed string at all — partner .66, neither
  high in exact-match); count_vowels/consonants ~.3/.55 (counting hard) — distinct from prior bias.
- **RECALL:** capital/largest ~.5–.6 top1, .76–.85 top2 (noisy, model-known-entity bound).
- **american top1=1.0 but partner@1=.15** (identity side occasionally emits the UK form — interesting).

**Next:** 3+2+1 top-k; k-sweep n_diff_demos 0..5 (k=0 control); non-numeric symmetric pairs
(AND|OR, earlier|later, alphabetical-order). **Blockers:** None.

---

## 2026-06-14 — Stream A: train_selected FVs at top-20 and top-40 heads

**Owner:** Coordinator (tmux "FV top20/40"). **Status:** DONE — 29 FVs each, both runs + organized + metadata.

**What:** Added `train_selected` FV runs at n_top_heads=20 and 40 for all 29 tasks (previously only
top-10). No CIE recompute — the train-pooled head artifact `results/multitask_aie_heads/
multitask_top_aie_heads.pt` already stores top-40, mean head activations cached in `results/gptj_fv/`.
Just re-summed `out_proj(mean_act[L,H,-1])` over the top-20 / top-40 heads.

**Commands run (HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1):**
- `compute_all_task_fvs_from_multitask_heads.py --n_top_heads 20 --output_root results/gptj_fv_multitask_top20` → 29 FVs.
- `compute_all_task_fvs_from_multitask_heads.py --n_top_heads 40 --output_root results/gptj_fv_multitask_top40` → 29 FVs.
- `write_fv_head_metadata.py --methods train_selected_top20 --n_top_heads 20` (and `_top40 --n_top_heads 40`).

**Files changed:**
- NEW `results/gptj_fv_multitask_top{20,40}/<task>/<task>_function_vector.pt` (29 each) + `fv_manifest.json`.
- NEW organized views `results/function_vectors/gpt-j/train_selected_top{20,40}/` (FV symlinks +
  `heads.pt`/`heads_metadata.json` → multitask_aie_heads + per-task `selected_heads.json`).
- `write_fv_head_metadata.py`: added `train_selected_top20/40` to `POOL_DESC`.

**Findings:** All 87 FVs (29×3 n) sane. Norms grow with head count: top-10 ~30–47, top-20 ~34–54,
top-40 ~41–66 (more heads add more head-out-proj terms to the sum). The first-10 / first-20 heads are
prefixes of the top-40 list (same ranking truncated), so top-10 ⊂ top-20 ⊂ top-40 head sets.

**Next:** optional — held-out steering eval (`evaluate_heldout_multitask_head_fvs.py`) to see whether
n=20/40 transfers better than n=10. **Blockers:** None.

**UPDATE 2026-06-15 (steering eval done):** ran `evaluate_heldout_multitask_head_fvs.py --n_top_heads
{20,40} --overwrite` (9 held-out test tasks, seed 42, full 28-layer sweep, zero-shot+FV and
10-shot-shuffled+FV; cached `fs_results_layer_sweep.json` filters → no baseline recompute). Logs
`results/_heldout_steering_logs/`.

**Consolidated into ONE folder** `results/heldout_multitask_head_eval_nheads/`:
- `top20/`, `top40/` — the full per-task result JSONs (moved from the old `_top20`/`_top40` dirs).
- `<task>_effectiveness_by_layer_nheads.png` (9) + `AGGREGATE_…png` — combined plots overlaying
  multitask n=10/20/40 + task-specific reference, both conditions. (n=10 data still lives in the
  pre-existing baseline `results/heldout_multitask_head_eval/`.)
- `nheads_comparison.json` — best-layer comparison table.
- NEW plotting script `src/eval_scripts/plot_nheads_steering_comparison.py` (pure; re-renders from
  the JSONs). Redundant per-folder `*_effectiveness_by_layer.png` were deleted (superseded).

**FINDING — more heads helps ZERO-SHOT steering, neutral/slightly-hurts with ICL context:**
- best-layer **zero-shot+FV** mean top-1: n10 **0.376** → n20 **0.381** → n40 **0.446** (task-specific 0.483).
  More heads recovers most of the gap to task-specific. Big n40 winners: capitalize 0.75→0.96,
  capitalize_first_letter 0.70→0.95; product-company recovers at n40 (0.22→0.24 after dipping to 0.12 @n20).
  word_length stays 0.00 at every n (FV can't drive it zero-shot).
- best-layer **10-shot-shuffled+FV** mean: n10 **0.796** → n20 0.785 → n40 0.780 (task-specific 0.812).
  Flat/slightly down — with real ICL context already present, extra heads add nothing.
- **Best zero-shot layer drifts EARLIER as n grows** for some tasks (capitalize n10 L13→n40 L1;
  country-currency L11→L1; product-company L11→L0). The larger-norm 40-head FV scores best injected
  very early — sanity-check before over-reading those cells; the mid-layer (L8–13) optimum is stable
  for antonym/synonym/capitalize_first_letter across all n.
- Takeaway: for zero-shot steering, n=40 train-pooled heads clearly beats n=10; for ICL-context
  steering, n=10 is already saturated.

---

## 2026-06-12 — Stream C: GPT-4-judged rhyme eval — GPT-J truly cannot rhyme (0/42)

**What:** `src/eval_scripts/judge_rhyme_generations.py` — rebuilds the same 10-shot test prompts
(seed 42), greedy-generates GPT-J's actual answer word, and has GPT-4.1 judge whether it truly
rhymes (strict: final stressed vowel + coda match; identical word/non-word/inflection = false).
Tests whether the 0.024 first-token score understates competence via valid alternative rhymes.

**Result: judged accuracy 0/42 = 0.000** (gold exact-match 1/42; the judge rejects even that one —
`apply→ally` isn't a true rhyme → the dataset's CMUdict gold pairs are themselves loose).
Failure modes: copies the input (17/42), semantic associates (black→white, birth→death),
orthographic neighbors (bite→bit, defense→defence). Consistent with BPE models lacking phonology.
**Implication:** rhyme is a genuine zero-competence task for GPT-J — keep only as a known-zero
control, or drop. The OpenAI key is available from the container env
(`/proc/1/environ` → `OPENAI_API_KEY`; pod env vars don't reach interactive shells).
Artifacts: `results/rhyme_judge_eval/`. The judge harness is reusable for lenient scoring of other
tasks (e.g. "is X actually east of Y" for the geo pair).

---

## 2026-06-12 — Stream C (round 2): FVs for 4 MORE paired tasks (east/west_neighbor, next_in_period/group) × 3 methods

**Owner:** Coordinator (tmux "FV derivation"). **Status:** DONE — 12 FVs built + verified; LOW-N CAVEAT below.

**Same recipe as round 1 (below)**: symlinked into `dataset_files/abstractive/`; NEW
`task_splits/paired_tasks_7.json` (supersedes _3); 4 parallel tmux windows (canonical → varicl
stage-1 shard, num_shards=4); stage-2 builders with `fv_manifest_paired2.json`; organized symlinks
+ `selected_heads.json` for all 3 methods. The auto no-filter retry **never triggered** — all 4
tasks pass the ICL filter with nonzero correct counts. Norms sane (24.4–43.9).

**IMPORTANT LOW-N CAVEAT — GPT-J is weak at all 4 tasks, so the correct-only averaging rests on
very few queries:**
- canonical 10-shot ICL-correct: east_neighbor **5/51**, west_neighbor **7/51**,
  next_in_period **2/14**, next_in_group **5/14** (vs next/prev_number 42/42).
- varicl (1–10 shots, valid split): east **2**, west **3**, next_in_period **1**, next_in_group **1**
  correct queries → those varicl FVs average over 1–3 prompts. Treat all four tasks' FVs (esp.
  varicl) as high-variance; consider `--no_filter_to_correct_icl` variants if robustness matters
  (rhyme precedent), or larger datasets / more valid examples.
- Filter kept ON for methodological consistency with the 29 originals (counts were nonzero).

Logs: `results/_paired_fv_logs/{canonical,varicl}_<task>.log`, `build_{varicl,train_selected}2.log`.

---

## 2026-06-12 — Stream B: periodic-table 2D-grid neighbour pair (next_in_period | next_in_group)

**Status:** COMPLETE (datasets generated).

**What:** Chemistry analog of the geography pair — the periodic table is a 2D grid the
model knows. `next_in_period` = element → element to its RIGHT (same row, atomic_number+1);
`next_in_group` = element → element BELOW it (same column, next period down). Both
element→element → matched input AND output marginals; a lone element name reveals nothing
about which grid direction. Generator
`dataset_files/generate/create_periodic_table_neighbor_datasets.py` (data: `mendeleev`, offline).

**Outputs:** `dataset_files/paired_tasks/{next_in_period,next_in_group}.json`, **66 pairs each**,
IDENTICAL input set (66 main-group + transition-block elements that have BOTH a right and a
below neighbour) → matched input marginal. f-block (lanthanides/actinides) excluded (no
group_id in source; also the elements the model knows worst). 0 self-pairs. Element names
mostly multi-token (only ~14 single-token) → allowed, first/last label-token scoring like
countries/numbers. Sane (Nitrogen R→Oxygen/B→Phosphorus; Selenium R→Bromine/B→Tellurium).

**Deps:** installed `mendeleev`.

**Intended pair:** next_in_period | next_in_group. (Skipped the US-presidents ordinal idea —
~46 too small.)

**Next:** capture paired activations (add `next_in_period_next_in_group` to
capture_oneshot_paired_tasks.py TASK_PAIRS) + geometry analysis.

**Blockers:** None.

---

## 2026-06-12 — Stream B: geography spatial-neighbour pair (east_neighbor | west_neighbor)

**Status:** COMPLETE (datasets generated).

**What:** New geographic task pair under the matched-marginal rule. Most geography leaks
(attribute pairs → disjoint output pools; landmark/park → input has "Park" in 747/749);
the only legal family is place→place SPATIAL relations over one shared pool — the geo
analog of next/prev_number. Built `east_neighbor`/`west_neighbor`: country → nearest
country in the E/W bearing quadrant, from most-populous-city coords (geonamescache,
deterministic, offline). Generator `dataset_files/generate/create_geography_neighbor_datasets.py`.

**Outputs:** `dataset_files/paired_tasks/{east_neighbor,west_neighbor}.json`, 244 pairs
each, IDENTICAL input set (244 countries with both an E and W neighbour) → matched input
marginal; both output countries → matched output marginal. 0 input==output collisions.
Multi-token country names allowed (first/last label-token scoring). Sane (Afghanistan
E→Pakistan/W→Turkmenistan; Netherlands E→Germany/W→UK); approximate for islands.

**Deps installed:** pycountry-convert, geonamescache, nltk+WordNet (WordNet unused so far —
semantic co-hyponym/hypernym/meronym pairs remain a validated option if wanted).

**Intended pair:** east_neighbor | west_neighbor (and could add north/south).

**Next:** capture paired activations for this pair (capture_oneshot_paired_tasks.py — add
`east_neighbor_west_neighbor` to TASK_PAIRS) and/or run the geometry analysis.

**Blockers:** None.

---

## 2026-06-12 — Stream C: derive FVs for 3 NEW paired tasks (rhyme, next_number, prev_number) × 3 methods

**Owner:** Coordinator (this session, tmux "FV derivation"). **Status:** DONE — all 9 FVs built + verified.

**RESULTS (all 9 sane: shape [4096]; norms 20–52; correct head-set provenance per method):**
- `task_specific` (own CIE top-10): rhyme 20.1, next_number 49.4, prev_number 51.9 → real files
  `results/gptj_fv/<task>/` + symlinks in `function_vectors/gpt-j/task_specific/<task>/`.
- `train_selected` (existing 20-train pooled heads): norms 37.0/43.5/44.0 → built into
  `results/gptj_fv_multitask_top10/` (`fv_manifest_paired.json`) + organized symlinks.
- `train_varicl` (existing varicl pooled heads): norms 32.1/34.8/37.6 → written directly into
  `function_vectors/gpt-j/train_varicl/<task>/` (`fv_manifest_paired.json`).
- `selected_heads.json` regenerated for all 3 methods (32 tasks each).

**IMPORTANT CAVEAT — rhyme is filter-less:** GPT-J scored **0/18 ICL-correct** on rhyme's valid
split (both pipelines crashed on the empty correctness filter), so rhyme was re-run with
`--no_filter_to_correct_icl` in BOTH the canonical and varicl pipelines. Its FVs average over all
18 (incorrect) queries — "activations while attempting rhyme", not successful execution. Not
methodologically comparable to the other 28+2 tasks (which filter to correct). By contrast
next_number = 18/18 correct, prev_number = 15/18 — number FVs are standard.

**Other notes:** task-specific top heads of next/prev_number overlap heavily with the pooled set
((15,5),(12,10)…) — the number tasks engage the canonical FV circuitry. Per-task GPU stages ran in
3 parallel tmux windows (~25 min total incl. the rhyme re-run); varicl stage-1 used the
`--num_shards 3` trick so writes_global=False protected the existing pooled artifact; isolated
`results/multitask_aie_heads_varicl_paired/` kept for provenance (mean activations copied into
`results/multitask_aie_heads_varicl/<task>/` for the builder). Logs: `results/_paired_fv_logs/`.

**Goal:** FVs for the new `dataset_files/paired_tasks/{rhyme,next_number,prev_number}.json` (200 pairs
each) via (1) **task_specific** (own CIE top-10), (2) **train_selected** (existing 20-train pooled
head set reused), (3) **train_varicl** (existing varicl train-pooled head set reused).

**Setup:**
- Symlinked the 3 JSONs into `dataset_files/abstractive/` (user-approved "treat as abstractive";
  loader hardcodes abstractive/extractive). Split = 140 train / **18 valid** / 42 test (seed 42) —
  NB small valid split → ≤18 query candidates for CIE/mean-activations (trials redraw demos).
- NEW `task_splits/paired_tasks_3.json` (the stage-2 builders validate `--tasks` against a manifest).

**Plan (GPU = 1×A100 80GB; GPT-J fp16 ≈ 12GB → 3 concurrent model instances, NOT 6):**
- 3 tmux windows (one per task), each sequentially: `src/compute_function_vectors.py
  --dataset_names <task> --batch_size 8` (mean acts + CIE + task-specific FV → `results/gptj_fv/<task>/`),
  then `compute_multitask_varicl_heads.py --tasks rhyme next_number prev_number --num_shards 3
  --shard_index <i> --save_path_root results/multitask_aie_heads_varicl_paired ...` (varicl acts+CIE;
  sharded ⇒ writes_global=False ⇒ existing pooled artifact safe).
- Then (cheap, single jobs): copy varicl mean activations into `results/multitask_aie_heads_varicl/<task>/`;
  `compute_all_task_fvs_varicl.py --tasks ... --task_manifest task_splits/paired_tasks_3.json
  --manifest_name fv_manifest_paired.json --overwrite`; `compute_all_task_fvs_from_multitask_heads.py
  --tasks ... --manifest_name fv_manifest_paired.json`; organized-folder symlinks; `write_fv_head_metadata.py`.

**Blockers:** None.

---

## 2026-06-12 — Stream B: PAIRED 1-shot activation capture for paired_tasks (step 1 of rerun)

**Status:** COMPLETE (capture done; downstream experiments next).

**What:** First step of rerunning the experiments on the new `paired_tasks` datasets,
using the PAIRED design (Stream-E style, generalized): for each shared output word `w`
producible under BOTH functions, build two 1-shot prompts that differ ONLY in the ICL
demo INPUT (same demo label `w`, same query) — so the activation difference at the
label token isolates the function, not token identity. New script
`src/eval_scripts/capture_oneshot_paired_tasks.py`, a sibling of (not an edit to)
Stream-E's `capture_oneshot_paired.py`; reuses the canonical capture primitives
(`get_residual_stack`, `selected_token_records`, `make_token_record`, `flush_shard`)
so locations + on-disk format match `results/residual_activations/*`. Reads
`dataset_files/paired_tasks/` directly.

(An earlier single-task capture — `capture_oneshot_singletask.py` + results — was the
wrong reading of the request and has been removed.)

**Design choice:** single-token outputs NOT required (the paired property holds for
multi-token labels too). This lets the number pair reach 100 (only 26 number-words are
single-token, but 198 shared). Captured the full standard location set per prompt:
pre/first/last label_token (+ label alias) and last_prompt_token (+ final alias), 28 layers.

**Pairs + counts (n_target=100, capped by shared-output availability):**
- antonym_synonym: 100 words → 200 prompts (555 shared available)
- synonym_rhyme:    82 words → 164 prompts (82 available — max)
- antonym_rhyme:    60 words → 120 prompts (60 available — max)
- next_number_prev_number: 100 words → 200 prompts (198 available)

**Command:** `for pair in antonym_synonym synonym_rhyme antonym_rhyme
next_number_prev_number; do HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1
python src/eval_scripts/capture_oneshot_paired_tasks.py --pair $pair; done`

**Outputs:** `results/oneshot_paired_tasks/<pair>/{shard_00000.pt, index.json}`. Each
shared word → 2 functions × 6 roles = 12 rows; activations `[12·n_words, 28, 4096]` fp32.

**Verified:** for each pair, the f1/f2 prompts are identical except the demo input — e.g.
`Q: seven\nA: eight…` vs `Q: nine\nA: eight…` (next vs prev of eight, same query);
`Q: unable\nA: able…` vs `Q: can\nA: able…`. Label span decodes to ` w` (assertion in
script, single- and multi-token).

**Next:** the actual experiments on these captures — FV/geometry analysis (function
axis at label vs query token), and/or generalize the `mixed_icl_*` top-k probe to the
four pairs.

**Blockers:** None.

## 2026-06-12 — Stream B: `paired_tasks` datasets for marginal-matched task pairs

**Status:** COMPLETE (datasets generated; experiments not yet run).

**What:** New dataset family for the mixed-task ICL / function-geometry probes, built
on the sharpened criterion that the function must live ONLY in the (input,output)
relation — neither the input token nor the output/label token alone may reveal the
task (matched input AND output marginals). This rules out the leaky pairs
(capitalize_first/last → output-marginal leak; translation → language ID leak;
country-capital/currency → output leak; hypernym/hyponym → input leak).

**Files:** `dataset_files/generate/create_paired_tasks_datasets.py` (new generator),
outputs in `dataset_files/paired_tasks/`:
- `next_number.json` / `prev_number.json` — k→k+1 / k→k−1, integers as WORDS (digits
  risk tokeniser clumping; words tokenise per-word). Same input set {1..200} → matched
  input marginal; outputs {2..201}/{0..199} → near-identical output marginal. 200 each.
- `rhyme.json` — word→rhyming word (CMUdict via `pronouncing`); inputs from syn/ant
  vocab, outputs constrained to that same real-word vocab → shares the general-word
  marginal with synonym & antonym. 200 pairs, single-token real-word outputs.
- `synonym.json`, `antonym.json` — copied verbatim from `abstractive/`.

**Intended pairs:** antonym|synonym, synonym|rhyme, antonym|rhyme, next_number|prev_number.

**Dropped next/prev_letter** (initially generated, then removed): only 26 letters → can't
reach 200, and a deeper review (see DECISIONS) showed the strict criterion +
in-context-learnability is a real double bind — letters added no new legal+learnable axis
the number/word pairs don't already cover. Decision: keep the 3 pairs, no 4th.

**Verification:** input sets identical across next/prev (matched input marginal);
0 input==output collisions; number-words never digit-clump (e.g. 200→201 = "two
hundred"→"two hundred one"); letters & rhyme outputs are single tokens. Compound
number-words >20 are multi-token but word-based (handled by first/last label-token
scoring). Installed `pronouncing` (PyPI, CMUdict).

**Next:** run `mixed_icl_*` probe on the new pairs (will need a task-generic variant
of the script, currently antonym/synonym-specific).

**Blockers:** None.

## 2026-06-11 — Stream E Phase 2: causal steering at the demo label token (COMPLETE, POSITIVE)

**Status:** Full run done on GPU. Plan: `/root/.claude/plans/immutable-finding-boole.md` (Phase-2 rewrite).

**What:** Inject `α·Δ_label(L_steer)` (mean antonym−synonym difference from the Phase-1 capture, 530
held-out-by-construction words) at the demo's label token of a **synonym-context 1-shot prompt**; read
(a) the query-final-token shift vs the natural direction `Δ_final(L_read)` and (b) rank/logit of gold
`ant(q)` vs `syn(q)`. Test queries = 1,003 shared-input words with single-token gold ant+syn (disjoint
from the Δ words). Steer L∈{6,9,11}, α∈{0,0.5,1,2,4,8}, matched-norm random control on every cell.
Geography (landmark↔park, 84 queries) geometry-only.

**Files:** NEW `src/eval_scripts/steer_label_to_query.py` (inlined add-hook/eval helpers; baukit only
inside main). **Two bugs found+fixed during smoke/plots:** (1) hook args must be the exact 2-arg
`(output, layer_name)` closure — extra default-kwargs get mis-bound positionally by baukit's
`invoke_with_optional_args` (a `Tensor += tuple` crash); (2) cos-vs-α plot initially read at
L_read==L_steer where the query-position shift is identically zero → now picks the best-aligned read
layer; added `fig_cos_shift_by_readlayer_L*.png`.

**Results (n=1003, antonym↔synonym):**
- **GEOMETRY — strongly positive:** steering at L6 moves the query token along the natural syn→ant
  direction: mean cos(shift, Δ_final) ≈ **0.71–0.76 at read L8** (α 0.5–2), >0.5 through L15, ~0.36 at
  L27; **random control ≈ 0.01–0.06 everywhere**. L9/L11 steer: peak cos 0.60/0.48 (read L15). Effect
  is α-stable in the linear regime (0.5–4) and degrades at α=8.
- **BEHAVIOR — positive flip:** baseline (α=0) flip rate 0.283, mean logit(ant)−logit(syn) = −1.44.
  Steering L6: flip **0.595 @ α=4, 0.637 @ α=8**, logit-diff crosses zero at α≈2 and reaches **+1.12**.
  L9 similar (peak 0.593 @ α=4, overshoot at 8); L11 weaker (0.48). Random control: flip ≤0.35,
  logit-diff still negative (−0.92) even at α=8 — the effect is direction-specific, not norm.
- **Geography geometry is much weaker** (cos ~0.13–0.23 vs random ~0; n=84). The landmark↔park
  "function" contrast at the label token propagates far less — consistent with its tiny Δ-final
  separation being carried near-entirely by position rather than content in 1-shot? (open question).
- **Reconciliation with Phase 1:** the *mean* function axis DOES propagate causally label→query
  (this run), even though the *per-word* label→query map is not linearly predictable (Phase-1 low R²).
  I.e. the dominant shared axis is causal; the high-dim per-word tail is what defeats the linear map.

**Outputs:** `results/oneshot_steering/<pair>/{per_query.csv, summary.json, fig_cos_shift_vs_alpha_L*.png,
fig_cos_shift_by_readlayer_L*.png, fig_flip_rate_vs_alpha.png, fig_logitdiff_vs_alpha.png}`.

**Steer-layer sweep extension (L1–5, L12/16/20/24; α∈{0,1,2,4}):** outputs in
`results/oneshot_steering_early/` and `_late/`; combined profile
`results/oneshot_steering/fig_steer_layer_profile_L1to24.png`. **A closed causal window [L4, ~L15]:**
L1–2 null (flip ≈ baseline 0.28, despite max downstream depth); sharp onset L3→L4 (flip 0.34→0.57 @α4,
cos 0.58→0.72); plateau L4–9 (flip 0.54–0.60, cos 0.61–0.76); decay L11–12 (0.48→0.44); **completely
dead by L16** (flip 0.282–0.284 = baseline at L16/20/24; cos 0.09→0.00). Natural ‖Δ_label‖ is also tiny
at L1–2 (0.66–0.94) vs L4 (3.3) — the axis doesn't exist yet. So: the function fingerprint is WRITTEN
into the label token ~L3–6 and CONSUMED (moved to the query position via attention) before ~L16 —
matching the CIE top-head band (L8–15) and best read layers (8–15). Caveat: α is scaled to the natural
per-layer ‖Δ_label‖, so the L1–2 null is "the natural-scale axis doesn't exist," not "no perturbation
could act there."

**Next (optional):** multi-shot variant (stronger baseline synonym behavior → cleaner flip test);
steer along top-k SVD directions instead of the mean; geography follow-up on why propagation is weak.

---

## 2026-06-11 — Stream E: 1-shot paired ICL function geometry (Phase 1, COMPLETE)

**Status:** Ran end-to-end on GPU (RTX PRO 4000 Blackwell). Approved plan:
`/root/.claude/plans/immutable-finding-boole.md`.

**What:** Same-output-word / different-function paired 1-shot prompts (demo label token = source;
query final token = target). Substrate: antonym↔synonym (544 shared single-token words) and
landmark↔park (84). Observational geometry only (steering = gated Phase 2).

**Files (new):** `src/eval_scripts/capture_oneshot_paired.py` (model+baukit; per-row source-token-id
assertion passed for all words), `src/eval_scripts/analyze_oneshot_geometry.py` (pure; subspace-projected
the source→target map so per-layer cost is O(m³), m≤2W, not O(4096³) — identical results, ~30× faster).
Installed `baukit` from git (`pip install git+https://github.com/davidbau/baukit.git`; not on PyPI).
**Env:** run with `HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1`.

**Outputs:** `results/oneshot_paired/<pair>/` (activations), `results/oneshot_paired_analysis/<pair>/`
(`{label,final}_geometry.json`, `fv_projection.json`, `source_target_map.json`, `summary.csv`,
`fig_fv_scatter_L*.png`).

**Findings:**
- **STRONG positive — function is linearly decodable at the query token in FV space.** Final-token acts
  separate antonym vs synonym along the FV difference axis `(fv_f1−fv_f2)`: peak **L11 AUC 0.941, d 2.22,
  88.8% acc** (antonym/syn); **L11 AUC 0.894** (geography). Clean mid-layer band L9–15. Confirmed in scatter.
- **NUANCED — label-token difference = one dominant axis + broad tail.** STABLE rank (Σσ²/σ₁²) of the
  L11 `D_label` is **5.2** (antonym/syn 544×4096) / **3.3** (geo 84×4096): σ₁ holds ~19%/~30% of energy →
  a single clear "function axis"; but 90% energy needs k≈315 dims (high-dim residual). Stable rank is
  lowest at mid layers (min ~4.7 @ L9), coinciding with peak FV separation. (Entropy eff-rank ~150 keyed
  on the fat tail; stable rank keys on the dominant axis.) Plots: `fig_Dlabel_svd_L11.png`,
  `fig_Dlabel_stable_rank_by_layer.png`; tensor `D_label.pt`.
- **NEGATIVE — the demo-label → query-token map is NOT low-rank, NOT rotation-like, weakly predictive.**
  Held-out `map_R2` ≤0.16 at mid layers (0.38 at L0, negative by L18); map effective rank ~190–280;
  in-sample rank-8 R² only ~0.25; Procrustes gap large at informative layers. So "label-space arithmetic
  → predictable rotation at the next position" is **not supported in 1-shot**.
- **Caveats:** 1-shot weakly identifies "function"; W(544/84) ≪ d(4096) so full-matrix M structural
  metrics are rank-limited — the trustworthy signals are held-out `map_R2` and the FV-separation AUC.

**Implication for Phase 2:** steering along the FV difference axis at/near the query token is well-motivated
(high separation); steering the demo label token expecting predictable low-rank propagation to the query is
NOT supported by Phase 1. Consider a multi-shot variant (stronger function identification) before concluding.

**Next:** decide whether to run Phase-2 steering (FV-axis at query) and/or a multi-shot variant.

---

## 2026-06-11 — Stream C: PCA-space (direct) activation→FV ridge sweep — DONE + full-dim comparison

**Status:** DONE. 899 cells (31 token positions × 29 layers), merged + heatmapped. Companion to the
full-dim ridge sweep, in a 16-PC bottleneck.

**Method:** per cell, fit activation PCA (k_act=16) on the pooled 20-train rows; FV PCA (k_fv=16)
fit once on the 20 train FVs; **ridge 16→16, λ via leave-one-train-task-out CV, single 20-train
standardizer** (same recipe as full-dim); predict 7 test tasks, **reconstruct to 4096-d**, report
MSE there. Direct projection (not joint). FV target = `train_selected`. cc/pc excluded.

**Commands:**
- `WORKER=src/eval_scripts/regress_activation_to_fv_pca_ridge.py OUTPUT_DIR=results/pca_ridge_activation_to_fv
   SESSION=pcaridge bash src/eval_scripts/run_fulldim_ridge_shards.sh` (3 tmux windows; ~4–5 min total).
- `python src/eval_scripts/merge_fulldim_ridge_results.py --input_dir results/pca_ridge_activation_to_fv`.

**Files changed:**
- NEW `src/eval_scripts/regress_activation_to_fv_pca_ridge.py` (worker).
- EDIT `src/eval_scripts/run_fulldim_ridge_shards.sh`: `WORKER` env override (backward-compatible).
- REUSED `merge_fulldim_ridge_results.py` unchanged (CSV is a superset of the full-dim schema).
- Output: `results/pca_ridge_activation_to_fv/{shard_icl1..10,combined_*}`.

**Findings (PCA 16→16 vs full-dim 4096→4096):**
- **Best cell ~identical:** PCA `icl10/finaltok @ L13 = 0.1147` vs full-dim `icl10/finaltok @ L11 =
  0.1161`. The 16-PC bottleneck is **free at the optimum** — even ~0.0014 better (PCA denoises →
  slightly better test transfer). Best layer flat L11–13.
- **Same structure:** query position (ICL 10) best; clean mid-layer bowl; embedding worst.
- **Where PCA wins vs loses:** in the sweet spot (L8–13) PCA ≈ or < full-dim (L11: 0.1321 vs
  0.1325; L13: 0.1320 vs 0.1331). In **later layers it's worse** (L20: 0.1475 vs 0.1407; L28:
  0.1583 vs 0.1491) and embedding worse (0.2026 vs 0.1960). Net per-cell mean Δ(pca−full)=+0.0032,
  PCA better in 243/899. → top-16 activation PCs capture all recoverable signal *where it's
  concentrated* (mid layers) but discard useful directions where it isn't (late/embedding).
- **Identity verified:** `test_mse = (16/4096)·pca_test_mse + floor` to 4e-8 (orthonormal FV-PCs).
  FV-PC reconstruction floor (test) ≈ 0.099 — the irreducible MSE of the 16-PC FV target.
- **α interior** (peak ~100; grid logspace(-2,6,17)); 11 pinned cells are all L0 embedding
  constant-feature positions (=0.217 predict-the-mean baseline), harmless.

**Next:** Headline comparison ready: a 16-PC activation→FV ridge loses essentially nothing vs full
4096-d at the best read points. Optional: sweep k_act to map the bottleneck cost by layer.

**Blockers:** None.

---

## 2026-06-10 — Stream B: mixed-task ICL probe (5 antonym + 5 synonym ICL → antonym vs synonym query)

**Status:** COMPLETE.

**Goal:** Quick experiment. 50 prompts, each with 10 ICL examples (5 antonym + 5
synonym demos). Two query conditions: antonym query vs synonym query. Measure
top-1/2/3 accuracy of the correct answer's first token (rank < k convention,
matching `eval_utils.compute_top_k_accuracy`). Cross-prompt batched (25/forward
pass). Clustered bar chart. Also ran the demo order swapped, and overlaid a
same-task 10-shot reference (10 antonym ICL→antonym, 10 synonym ICL→synonym).

**Files:**
- `src/eval_scripts/mixed_icl_antonym_synonym_topk.py` (new). Flags: `--icl_order
  {antonym_first,synonym_first}`, `--baseline_path` (cached same-task ref, computed
  once and reused), `--refresh_baseline`.
- `results/mixed_icl_antonym_synonym/` (antonym_first), `…_synfirst/` (synonym_first),
  `…_baseline/pure_baseline.json` (shared reference).

**Important fix (after first runs):** the query word was NOT being excluded from the
ICL demos (the `exclude_idx` arg existed but was never passed). Fixed: draw the query
first, then exclude its input word from both tasks' demos (`sample_pairs(...,
exclude_inputs=...)`). Collision rate was low (~0.2%/prompt), but the fix reordered
the RNG stream → effectively a fresh draw. Also bumped n_prompts 50→200 because n=50
SE≈0.06 made the order comparison unreliable. **Numbers below are the n=200,
collision-free run; the earlier n=50 numbers are superseded.**

**Commands:**
- `HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 python
  src/eval_scripts/mixed_icl_antonym_synonym_topk.py --n_prompts 200 --icl_order
  antonym_first --output_dir results/mixed_icl_antonym_synonym --refresh_baseline`
- same with `--icl_order synonym_first --output_dir …_synfirst` (reuses cached baseline)

**Results, n=200 (top-1 / top-2 / top-3):**
- antonym_first ICL (antonym demos far, synonym demos recent):
  antonym query 0.47/0.55/0.61; synonym query 0.20/0.30/0.42
- synonym_first ICL (synonym demos far, antonym demos recent):
  antonym query 0.55/0.66/0.73; synonym query 0.12/0.24/0.30
- same-task 10-shot ref: antonym 0.565/0.685/0.76; synonym 0.365/0.50/0.605

**Findings (n=200, SE≈0.035):**
- **Difficulty:** antonym query ≫ synonym query top-1 in every cut (0.47–0.55 vs
  0.12–0.20). Synonym has many valid answers; the single gold first-token often misses
  top-3, so first-token top-k understates synonym competence.
- **Recency IS real here** (corrects the noisy n=50 read): each task does ~+0.08 top-1
  better when *its* demo block is last (closest to the query). antonym query: 0.47
  (demos far) → 0.55 (demos recent); synonym query: 0.12 (demos far) → 0.20 (recent).
- **Dilution:** mixed-ICL synonym (0.12–0.20) sits far below its same-task 10-shot ref
  (0.365); mixed-ICL antonym (0.47–0.55) is close to its ref (0.565), nearly matching
  it when antonym demos are recent. → synonym pays most for halved demos.

**Env note:** transformers/accelerate not installed locally; `pip install
transformers==4.49.0 accelerate` (matches fv_environment.yml). baukit (transitive
dep of eval_utils) not on PyPI + git install blocked → inlined the 3 pure helpers
(get_answer_id, compute_individual_token_rank, compute_top_k_accuracy). GPT-J
weights cached at /workspace/.cache/huggingface; run with HF_HUB_OFFLINE=1.

**Next:** none. Optional: score synonym against any-acceptable-synonym for a fairer
read (current numbers understate synonym competence).

**Blockers:** None.

---

## 2026-06-11 — Stream D: held-out per-layer steering eval for train_varicl (3-series plots)

**Status:** RUNNING (8 tasks; word_length smoke done). Goal: overlay the `train_varicl` FV as a
3rd line on the existing 9 test-task `*_effectiveness_by_layer.png` steering plots.

**Approach (cheap reuse, not a 4h re-run):** new sibling script
`src/eval_scripts/evaluate_heldout_varicl_fv.py` reuses the existing June-8 per-layer results
(multitask train-only + task-specific) from each task's `comparison_summary.json`, and ONLY
evaluates the prebuilt `train_varicl` FV across all 28 layers (zero-shot + 10-shot-shuffled),
using the IDENTICAL filter set (clean_rank_list==0 from `results/gptj_fv/<task>/fs_results_layer_sweep.json`)
and seed=42 so the new line is directly overlayable. Imports evaluate_fv/get_filter_set/
summarize_results from `evaluate_heldout_multitask_head_fvs.py` (shared file NOT modified).

**Verified (word_length smoke):** filter set reproduced exactly (22 == recorded 22, no warning);
3-series plot renders. Each task ~1–2 min after model load (171 ex/layer × 28 × 2).

**Outputs:** per task → `varicl_heads_{zs,fs_shuffled}_results.json`, `varicl_comparison_summary.json`,
`<task>_effectiveness_by_layer_with_varicl.png` (originals kept). Aggregate →
`results/heldout_multitask_head_eval/heldout_varicl_vs_others_summary.json`. Series labels:
"Multitask heads (train, fixed-ICL)" / "Task-specific heads" / "Variable-ICL multitask (train_varicl)".

**Caveat:** the "Multitask heads" baseline = train-only 20-task FIXED-ICL set (the June-8 run),
so this compares fixed-ICL vs variable-ICL train pooling on the same 9 held-out tasks.

---

## 2026-06-10 — Stream D: variable-ICL train-pooled FV method (COMPLETE on pod)

**Status:** All 4 steps DONE. 29 FVs + heads.pt + fv_manifest.json + 29 selected_heads.json under
`results/function_vectors/gpt-j/train_varicl/`. Sanity passed.

**Step 3 (rebuilt):** after filling the 9 test-task activations (gap below), re-ran
`compute_all_task_fvs_varicl.py --overwrite` → 29 FVs (20 train + 9 test), norms 22–43, all sane.
**Step 4:** `write_fv_head_metadata.py --methods train_varicl --n_top_heads 10` → 29 metadata files.
**Sanity:** varicl top-10 vs `train_selected/heads.pt` = **8/10 overlap** (top-2 identical: (9,14),(15,5);
varicl swaps in (10,0),(21,2) for (9,2),(14,9)). Overlap-not-identity, as expected.

**Leftover dirs:** `results/_smoke_varicl/` (smoke), `results/_varicl_testtasks/` (test-task CIE +
activations + a harmless test-pooled head artifact; the 9 mean-activation .pt were copied into
`results/multitask_aie_heads_varicl/<task>/`). Keep for provenance or delete — not referenced by FVs.

---

## 2026-06-10 — Stream D: variable-ICL train-pooled FV method (RUNNING on pod)

**Status:** GPU available in THIS shell (A100 80GB, torch 2.8+cu128, transformers 4.49.0) —
contradicts the earlier "no transformers/GPU" note. Smoke test PASSED; full stage-1 launched.

**Smoke (step 1):** `compute_multitask_varicl_heads.py --tasks ag_news sentiment country-capital
--abstractive_only --query_split valid --demo_split train --n_top_heads 10 --batch_size 8
--max_prompts_per_task 6 --num_shards 1 --save_path_root results/_smoke_varicl --overwrite` →
exit 0. All 3 checks green: mean tensor shape (28,16,256) per task; fs_results_varicl_valid.json
present per task; cie_result confirms min_shots=1/max_shots=10/cap=170, metadata
n_shots="variable[1,10]"; deterministic sampler spans full 1–10 over 170 queries.

**Stage-1 (step 2): DONE.** `bash src/eval_scripts/run_multitask_varicl_all_tasks.sh 4` — 4 shards
on GPU 0, tasks[shard::4] over 20 train tasks, then --reduce. All 20 per-task cie_results written
(1189 prompts total) → `results/multitask_aie_heads_varicl/multitask_top_aie_heads.pt` (+metadata).
Single A100 = GPU-bound bottleneck; shard 3 drew the 5 heaviest tasks and ran solo for the tail
(commonsense_qa alone took 2h21m — long multiple-choice prompts × up to 10 ICL demos; the 170-cap
DID apply: 22 batches×8≈170 candidates, so the cost is sequence LENGTH not prompt count).

**Pooled top-10 heads (varicl):** L9H14 0.0553, L15H5 0.0546, L8H1 0.0242, L12H10 0.0226,
L11H0 0.0138, L8H0 0.0109, L14H0 0.0106, L24H6 0.0090, L21H2 0.0082, L10H0 0.0081.

**Stage-2 (step 3): PARTIAL → GAP FOUND.** `compute_all_task_fvs_varicl.py` built the **20 train**
FVs (norms ~22–43, sane) then crashed: `FileNotFoundError landmark-country_mean_head_activations_varicl.pt`.
The builder does NOT compute activations on the fly — it requires precomputed varicl mean
activations for ALL 29 tasks, but stage-1 (`--task_split_key train_tasks`) only produced them for
the 20 train tasks. The 9 test-task single-position varicl activations were never computed (gap in
the documented step 1→3 sequence).

**Fix (running):** generate the 9 test-task varicl mean activations via the stage-1 worker on the
test split, isolated dir so `writes_global` can't clobber the train head set:
`compute_multitask_varicl_heads.py --task_split_key test_tasks ... --save_path_root
results/_varicl_testtasks --num_shards 1`. (Computes CIE too — unavoidable, but test tasks are all
short-sequence so it's cheap.) Then: copy the 9 `*_mean_head_activations_varicl.pt` into
`results/multitask_aie_heads_varicl/<task>/`, re-run `compute_all_task_fvs_varicl.py --overwrite`
(rebuilds all 29 + manifest), then step 4 metadata. Sanity: compare pooled top-10 vs
train_selected/heads.pt.

---

## 2026-06-10 — Stream D: variable-ICL train-pooled FV method (IMPLEMENTED, not yet run)

**Status:** Code complete + py_compile/bash -n clean. NOT run (no `transformers`/GPU in the
coordinator shell; user runs the 3 shards on a separate GPU instance).

**Question / method:** A 4th FV head-selection method. Each prompt draws a RANDOM 1–10 ICL
count; keep only prompts the model answers correctly, capped at **170 successful/task**; read
both mean head activations AND the CIE intervention at the **query predictive (last) token**
(T=-1) so activations average over variable-length prompts at one consistent position; CIE uses
variable ICL + shuffled labels; head selection pools CIE across the **20 train tasks** (like
`train_selected`, new regime). FVs built for all 29 tasks → `train_varicl`.

**Files created:**
- `src/utils/varicl_utils.py` — `sample_variable_icl_count`, `build_varicl_prompt_data`,
  `get_last_token_mean_head_activations` (→ shape (n_layers,n_heads,head_dim)),
  `varicl_correctness_filter`, `batch_varicl_last_token_intervention` (single-position).
- `src/eval_scripts/compute_multitask_varicl_heads.py` — stage-1 engine (mirrors
  `compute_multitask_top_aie_heads.py`; reuses select_shard/aggregate/reduce; new args
  `--min_shots --max_shots --max_successful_prompts --cie_seed_offset`; train-only pooling).
- `src/eval_scripts/compute_all_task_fvs_varicl.py` — stage-2 (single-position indexing fix).
- `src/eval_scripts/run_multitask_varicl_all_tasks.sh` — 3 background shards + reduce, batch 8.

**Verified (coordinator):** reused signatures match (`_project_attention_inputs`,
`sample_demo_indices`, `word_pairs_to_prompt_data`, `get_answer_id`,
`compute_individual_token_rank`); `prepend_bos` expr matches engine; FV indexing is
`mean_activations[layer, head]` (no `[-1]`); advanced-indexing for last-token read +
head replacement is correct. One benign deviation: compat attrs set on `args`
(`n_shots="variable[1,10]"`, `shuffle_labels`, `mean_activations_root`) so the reused
`write_global_artifact` metadata writer works.

**Outputs (when run):** heads → `results/multitask_aie_heads_varicl/`; FVs →
`results/function_vectors/gpt-j/train_varicl/`.

**Next (run on GPU box):**
1. Smoke: `python src/eval_scripts/compute_multitask_varicl_heads.py --tasks ag_news sentiment country-capital --abstractive_only --query_split valid --demo_split train --n_top_heads 10 --batch_size 8 --max_prompts_per_task 6 --num_shards 1 --save_path_root results/_smoke_varicl --overwrite` → check mean tensor shape (28,16,256), fs_results_varicl_valid.json, shot-count spans 1–10.
2. Full: `bash src/eval_scripts/run_multitask_varicl_all_tasks.sh` (3 shards + reduce).
3. FVs: `python src/eval_scripts/compute_all_task_fvs_varicl.py` (defaults build all 29).
4. Metadata: `python src/eval_scripts/write_fv_head_metadata.py --model_root results/function_vectors/gpt-j --methods train_varicl --n_top_heads 10`.
5. Sanity: compare pooled top-10 vs `train_selected/heads.pt` (expect overlap, not identity).

**Blockers:** None. Plan: `/root/.claude/plans/immutable-finding-boole.md`.

---

## 2026-06-10 — Stream C: direct full-dim (4096→4096) activation→FV ridge per (token pos, layer)

**Owner:** Coordinator (this session). **Status:** DONE — all 899 cells computed, merged, heatmapped.

**Question:** How linearly recoverable is a task's `train_selected` FV from a *single* residual
activation, as a function of token position and layer? **No PCA** — full 4096→4096 ridge, λ by
leave-one-train-task-out CV, single 20-train standardizer, MSE on 7 test tasks (excl.
country-currency, product-company). 31 token positions (pre/first/last for ICL 1–10 + final
prompt token) × 29 layers = 899 cells. Sharded by ICL index (10 shards) for tmux.

**Commands run:**
- `bash src/eval_scripts/run_fulldim_ridge_shards.sh` (CONCURRENCY=3 → 3 tmux windows in session
  `fvridge`: w0=icl{1,4,7,10}, w1=icl{2,5,8}, w2=icl{3,6,9}).
- icl10 re-run after a fix (see Findings): `... --icl_index 10 --overwrite`.
- `python src/eval_scripts/merge_fulldim_ridge_results.py --input_dir results/fulldim_ridge_activation_to_fv`
  → `combined_metrics.csv` (899 rows), `combined_{test_mse,best_alpha}_heatmap.png`, `combined_summary.json`.

**Files changed:**
- NEW `src/eval_scripts/regress_activation_to_fv_fulldim_ridge.py` (worker; one shard = one ICL idx).
- NEW `src/eval_scripts/run_fulldim_ridge_shards.sh`, `src/eval_scripts/merge_fulldim_ridge_results.py`.
- Output: `results/fulldim_ridge_activation_to_fv/{shard_icl1..10,combined_*}`.

**Findings:**
- **BUG FIXED:** the final prompt token (`last_prompt_token`) is stored with
  `icl_example_index = None` (not 10) in the `4tokens` dir; only the 3 label roles use 10. Loader
  now resolves `None` for that role (`role_load_icl_index`). icl1–9 were unaffected.
- **Best cell: final prompt token @ layer 11, test_mse = 0.1161** (α≈3.2e4). Runner-up:
  query pre-label token @ L11 (0.1172). Both at the query position (ICL 10).
- **Layer profile (mean over all 31 positions):** clean bowl, **min at L11 (0.1325)**, best band
  L10–14; embedding L0 worst (0.196), slow degrade to L28 (0.149). Matches the prior layer≈8–12 result.
- **Best per role:** finaltok 0.1161 (L11) < pre 0.1172 (L11) < first 0.1252 (L11) < last 0.1257 (L13).
- **Query position dominates:** the strongest cells are all at ICL 10 (more accumulated context);
  early ICL demos decode worse. The query's final/pre-label tokens are the most FV-predictive reads.
- **α sanity:** bulk α ∈ 1e3–1e5 (peak 1e4), interior. The 11 "pinned" cells are all L0 (embedding)
  constant-feature positions (the ":" pre-label token / final token embed identically across tasks →
  "predict-the-mean" baseline 0.217, α irrelevant). No grid widening needed.

**Next:** Compare this direct-ridge floor (~0.116) against the joint-PCA reconstruction MSE
(different metric — see Open Q3); decide whether direct full-dim ridge becomes the canonical decoder.

**Blockers:** None. (Single A100 80GB; 3 shells time-shared → ~5s/cell-of-work, ≈ serial; the
split bought load-overlap + restartability, not GPU speedup. Real speedup needs >1 GPU.)

---

## 2026-06-10 — Stream B: direct sweep #3 — k_activations fixed=16, sweep k_FV

**Status:** DONE. Added `--fix_act_k` to `sweep_k_activation_to_fv_direct_log2.py` (mutually
exclusive with `--fix_fv_k`): pins k_activations and reinterprets the doubling grid as k_FV
(capped at fv_k_cap). Ran k_act=16 fixed, k_FV ∈ {1,2,4,8,16}, direct, 7 tasks, ICL 1–5.

**Command:**
- `sweep_k_activation_to_fv_direct_log2.py --output_dir .../activation_to_fv_direct_ols_multitask_top10_log2_fixedactk16_exclude_cc_pc --fix_act_k 16 --k_max 16 --icl_example_indices 1 2 3 4 5 --test_tasks <7>` → exit 0.

**Files changed:**
- `sweep_k_activation_to_fv_direct_log2.py`: NEW `--fix_act_k` mode (plot labels/title +
  run_config now mode-aware: sweep_variable, fix_act_k recorded). py_compile OK.
- NEW `results/k_sweeps/activation_to_fv_direct_ols_multitask_top10_log2_fixedactk16_exclude_cc_pc/`.
  k_sweeps now has 4 direct dirs.

**Findings (k_act=16, sweep k_FV, ICL5):**
- **Test MSE falls monotonically with k_FV; best = k_FV=16 (the cap) in 14/15 series.** No
  overfitting in the k_FV direction (unlike k_act). icl5/last: 0.1805(kFV1)→0.1259(kFV16).
- **Test MSE rides just above the recon floor** (gap ~0.01→0.03); the floor itself drops
  0.172→0.099 as k_FV grows. So k_FV is limited by how much FV variance you discard, NOT by
  overfitting — and the FV PCA caps at 16 (20 train tasks → rank ≤19, capped 16).
- **With 16 activation PCs as input, the regression recovers any k_FV target ~to the floor** —
  the gap only widens modestly as more FV directions are demanded.
- **Combined across all 3 sweep axes: joint optimum = (k_act=16, k_FV=16), the corner.** k_act
  peaks at 16 then overfits; k_FV improves monotonically up to its 16 cap.

**Next:** None pending — the three orthogonal cuts (coupled diagonal, fix k_FV, fix k_act) are done.

**Blockers:** None.

---

## 2026-06-10 — Stream B: all k_sweeps converted to DIRECT method; joint runs deleted

**Status:** DONE. Per user: overwrite every non-direct k-sweep with the direct method. Ran the
two missing direct configs and **deleted all three joint result dirs**. `results/k_sweeps/` is
now 100% direct (regression = project act→k_act PCs, FV→k_FV PCs, OLS between them).

**Commands:**
- `sweep_k_activation_to_fv_direct_log2.py --output_dir .../activation_to_fv_direct_ols_multitask_top10_log2 --icl_example_indices 1 2 3 4 5 --k_max 1024` (coupled, full 9 tasks) → exit 0.
- `sweep_k_activation_to_fv_direct_log2.py --output_dir .../activation_to_fv_direct_ols_multitask_top10_log2_fixedfvk16_exclude_cc_pc --fix_fv_k --icl_example_indices 1 2 3 4 5 --k_max 1024 --test_tasks <7>` → exit 0.
- `rm -rf` the 3 joint dirs (`activation_to_fv_ols_multitask_top10_log2{,_exclude_cc_pc,_fixedfvk16_exclude_cc_pc}`).

**Files changed:**
- NEW direct dirs: `activation_to_fv_direct_ols_multitask_top10_log2` (9-task coupled),
  `..._direct_ols_..._fixedfvk16_exclude_cc_pc` (7-task, k_FV pinned 16). The 7-task coupled
  direct dir already existed. DELETED the 3 joint dirs.
- `results/k_sweeps/` now holds exactly 3 dirs, all direct. (Joint *script*
  `sweep_k_activation_to_fv_ols_log2.py` kept as a tool; only its results were removed.)

**Findings (direct, k_FV fixed=16 — settles the "1 PC is enough?" question):**
- **k_act=1 → test MSE ≈ 0.18–0.19** (≈ predict-the-mean baseline ~0.21; floor@k_FV16 = 0.099),
  **k_act=16 → ≈ 0.126.** So a single activation-PC does NOT recover the 16-dim FV — you need
  ~16. The old joint plot's low-k_act minimum was ENTIRELY the 16 FV-basis features it appended
  to the inputs; with the honest direct features those vanish.
- Direct coupled optima unchanged from before (~0.126 at ICL5, k≈16; first-label edges to 32).

**Next:** None pending. All k-sweeps are direct.

**Blockers:** None.

---

## 2026-06-10 — Stream B: NEW "direct" k_activations→k_FV regression (coupled diagonal log2 sweep)

**Status:** DONE. User redefined the regression: instead of the joint-space setup (project both
activation and FV onto the concatenated [act-PCA | FV-PCA] basis), **project activations → k_act
PCs (input), FVs → k_FV PCs (target), regress R^{k_act}→R^{k_FV} directly**, reconstruct to
4096-d for MSE. First deliverable: the coupled diagonal log2 sweep (1,1),(2,2),(4,4),(8,8),
(16,16),(32,16),… i.e. k_FV = min(k,16). New standalone script; joint scripts untouched.

**Command:**
- `python src/eval_scripts/sweep_k_activation_to_fv_direct_log2.py
   --output_dir results/k_sweeps/activation_to_fv_direct_ols_multitask_top10_log2_exclude_cc_pc
   --icl_example_indices 1 2 3 4 5 --k_max 1024
   --test_tasks landmark-country word_length capitalize_first_letter synonym
   lowercase_first_letter capitalize antonym`  → exit 0; 15 series.

**Files changed:**
- NEW `src/eval_scripts/sweep_k_activation_to_fv_direct_log2.py` (also supports `--fix_fv_k`
  for the next step). py_compile OK.
- NEW `results/k_sweeps/activation_to_fv_direct_ols_multitask_top10_log2_exclude_cc_pc/`.
- WORKLOG (this entry), DECISIONS.

**Findings:**
- **Direct is ~0.002–0.003 HIGHER MSE than joint** everywhere (e.g. icl5/first 0.1261 vs
  0.1232; icl5/last 0.1259 vs 0.1239). Expected & correct: joint's feature vector secretly
  included the activation projected onto the FV basis (extra inputs), so joint numbers were
  mildly optimistic. Direct = the honest activation-space→FV-space regression.
- **Optimal k unchanged: ≈16** (first-label 32); diagonal bowl bottoms at (16,16), rises past.
  More ICL → better (ICL5 best ~0.126). pre-label noisier/flatter (icl2/pre wanders to k=256 on
  a near-flat curve), first/last-label clean minimum at 16–32.

**Next:** Per user's stated plan, sweep k_activations with k_FV fixed (use `--fix_fv_k` on the
new direct script) — the direct analogue of the joint `--fix_fv_k` run.

**Blockers:** None.

---

## 2026-06-10 — Stream B: sweep k_activations with k_FV pinned at 16 (`--fix_fv_k`)

**Status:** DONE. New flag `--fix_fv_k` on `sweep_k_activation_to_fv_ols_log2.py` holds the
FV-side PCs at `fv_k_cap` for EVERY k (instead of `fv_k = min(k, fv_k_cap)`), so the sweep
isolates **k_activations** with **k_FV pinned at 16**. Same k grid (1,2,4,…,1024), same
7-task test set (cc/pc excluded), layer 11, ICL 1–5.

**Command:**
- `python src/eval_scripts/sweep_k_activation_to_fv_ols_log2.py
   --output_dir results/k_sweeps/activation_to_fv_ols_multitask_top10_log2_fixedfvk16_exclude_cc_pc
   --fix_fv_k --icl_example_indices 1 2 3 4 5 --k_max 1024
   --test_tasks landmark-country word_length capitalize_first_letter synonym
   lowercase_first_letter capitalize antonym`  → exit 0; 15 series.

**Files changed:**
- `sweep_k_activation_to_fv_ols_log2.py`: NEW `--fix_fv_k` flag; plot tick labels + title +
  run_config + stdout now reflect the fv_k rule (derive fv_k from rows). py_compile OK.
- NEW `results/k_sweeps/activation_to_fv_ols_multitask_top10_log2_fixedfvk16_exclude_cc_pc/`.
- WORKLOG (this entry), DECISIONS.

**Findings (k_FV fixed = 16; best fv_test_mse per series):**
- **first/last-label tokens:** optimal **k_activations ≈ 16–32** (e.g. icl5/first k=32 0.1232,
  icl5/last k=16 0.1239). Clear bowl; rises past 32 (overfit).
- **pre-label token:** optimal **k_activations is tiny (1–8)** — adding activation PCs *hurts*
  (icl4/pre k=1 0.1299, icl3/pre k=1 0.1356, icl5/pre k=8 0.1248). The pre-label activation
  carries little task signal, so extra PCs are noise.
- **vs the coupled run:** identical for k≥16 (fv_k=16 in both); the only change is k<16, which
  is now *lower* (better) because the FV target is no longer shrunk. So pinning k_FV=16 mainly
  rescues the low-k_activations regime.
- **Headline:** with k_FV=16, the activation side saturates by **k_activations≈16**; first-label
  edges lowest at 32. Marginal value of activation PCs beyond ~16–32 is negative.

**Next:** If a single canonical config is wanted: layer 11, last/first-label token, ICL 5,
k_activations≈16, k_FV=16.

**Blockers:** None.

---

## 2026-06-10 — Stream B: log2 k-sweep excluding country-currency + product-company + k_sweeps reorg

**Status:** DONE. Re-ran `sweep_k_activation_to_fv_ols_log2.py` with the two weakest test
tasks dropped (`country-currency`, `product-company`) — the user's reason: their
train(multitask)-selected FVs perform much worse than task-specific head selection. Also
nested all k-sweep outputs under a new `results/k_sweeps/` parent for repo clarity.

**Command:**
- `python src/eval_scripts/sweep_k_activation_to_fv_ols_log2.py
   --output_dir results/k_sweeps/activation_to_fv_ols_multitask_top10_log2_exclude_cc_pc
   --icl_example_indices 1 2 3 4 5 --k_max 1024
   --test_tasks landmark-country word_length capitalize_first_letter synonym
   lowercase_first_letter capitalize antonym`
  (originally written to the flat path, then moved into k_sweeps/; fv_root, layer 11,
  fv_k_cap 16, k_min 1 all default = same as original 9-task run.) Exit 0; 15 series swept.

**Files changed:**
- NEW `results/k_sweeps/` parent; moved both runs in (orig 9-task + new 7-task), trimmed
  redundant `k_sweep_` prefix, fixed `output_dir`/`metrics_csv`/`plot_png` self-paths in both
  `run_config.json`.
- `sweep_k_activation_to_fv_ols_log2.py` + `sweep_k_activation_to_fv_ols.py`: default
  `--output_dir` now points inside `results/k_sweeps/`.
- WORKLOG.md (this entry), DECISIONS.md (reorg + finding).

**Findings:**
- **Dropping the two tasks barely moves the regression test-MSE** (best fv_test_mse within
  ~±0.001 of the 9-task run; some cells slightly *higher*). Best at ICL5: ~0.123–0.126.
- **Optimal-k structure is UNCHANGED:** bowl minimum at k≈16–32 (fv_k capped 16), overfit
  past 32. Same as the full-9-task run.
- **Key takeaway:** the tasks that are *bad for FV steering* (cc, pc) are NOT the tasks that
  are *hard to regress from activations*. The two performance notions are decoupled — so
  excluding them doesn't clean up the regression aggregate.

**Next:** Decide whether the regression metric is the right lens for the cc/pc weakness
(it isn't — that weakness shows up in steering, not activation→FV reconstruction MSE).

**Blockers:** None.

---

## 2026-06-10 — Coordinator: verified train+test build + corrected degeneracy wording

**Status:** Verified the train+test FV build is complete and sound; corrected two
imprecise claims in the prior entry's degeneracy finding. No new artifacts produced.

**Commands run:**
- `ls`/manifest inspection: 29/29 `train_test_selected/<task>/<task>_function_vector.pt`
  + `fv_manifest.json` (heads_path = `multitask_aie_heads_all_tasks`, n_top=10,
  fv_root=`gptj_fv`) + `heads.pt`/`heads_metadata.json`/per-task `selected_heads.json`.
- torch diff of `train_selected` vs `train_test_selected` over all 29 tasks:
  **global max|Δ| = 0** on the `function_vector` tensors (exactly equal).
- Confirmed head SET identical, rank ORDER differs (train: (9,14) first; train+test:
  (15,5) first). `cmp` on the `.pt` files reports DIFFER — that's dict metadata only.

**Files changed:**
- WORKLOG.md (this entry + tightened degeneracy bullet below).
- DECISIONS.md: table cell train+test = **BUILT**; degeneracy finding wording corrected.

**Findings:**
- Build correct; all three methods now complete for GPT-J.
- Degeneracy is real but the prior "byte-identical FVs" wording was wrong: the FV
  *tensors* are exactly equal; the *files* are not (metadata). Set-identical ≠ list-identical.

**Next:** To make train vs train+test meaningful, rebuild at n>10 (they first differ ~n=11).
Open Q1 (held-out eval head set) is moot at n=10 — train and train+test give the same FVs.

**Blockers:** None.

---

## 2026-06-10 — train+test FVs built; per-task head metadata; degeneracy finding

**Status:** All three methods now have FVs for all 29 tasks under
`results/function_vectors/gpt-j/`. Each task folder has a `selected_heads.json`.

**Commands run (by user + coordinator):**
- Built train+test FVs (user ran the build command) → 29 real
  `train_test_selected/<task>/<task>_function_vector.pt` + `fv_manifest.json`.
- `python src/eval_scripts/write_fv_head_metadata.py --model_root results/function_vectors/gpt-j --n_top_heads 10`
  → 87 `selected_heads.json` (29 × 3 methods).

**Files changed:**
- NEW `src/eval_scripts/write_fv_head_metadata.py` (writes per-task head metadata; idempotent).
- NEW `results/function_vectors/gpt-j/train_test_selected/` (29 FVs + manifest) and
  `selected_heads.json` in every task folder of all three methods.

**Findings:**
- FV norms sane (~30–47), consistent with the FV paper.
- **DEGENERACY at n=10:** `train_selected` and `train_test_selected` select the
  *same set* of top-10 heads (rank order + CIE scores differ slightly — train ranks
  (9,14) first, train+test ranks (15,5) first — but membership is identical through
  n=10). Because the FV is an order-independent sum over the selected heads, the
  `function_vector` tensors are **exactly equal for all 29 tasks** (verified: global
  max|Δ| = 0; norms match). NB: the `.pt` *files* are not byte-identical — `cmp`
  differs on the dict metadata (`top_heads` order, paths) — only the tensors match.
  They diverge only at larger n (34/40 overlap at n=40). So a train-vs-train+test
  comparison at top-10 is meaningless; to study the selection-leakage effect, raise
  n_top_heads or use a per-task metric. `task_specific` is genuinely distinct.

**Next:** When studying train vs train+test, use n>10 (they differ from ~n=11+).
Optionally rerun `write_fv_head_metadata.py` after building FVs for new models.

**Blockers:** None.

---

## 2026-06-10 — Coordinator: FV reorg + provenance resolution

**Status:** Located all three head-selection methods + their FVs; built an organized
FV folder; resolved the split-provenance question; staged (did not run) the
train+test FV build.

**Commands run:**
- Verified mean-activation cache: all 29 tasks have `*_mean_head_activations.pt` in
  `results/gptj_fv` (so train+test FV build needs no forward passes).
- Built `results/function_vectors/{task_specific,train_selected,train_test_selected}/`
  via relative symlinks into existing caches (29/29 FV links + head-set links resolve).

**Files changed:**
- NEW `results/function_vectors/gpt-j/{task_specific,train_selected,train_test_selected}/`
  tree (model-nested for future models) + `README.md`.
- `compute_all_task_fvs_from_multitask_heads.py`: added `--tasks` (subset sharding) and
  `--manifest_name` (so parallel shards don't clobber one manifest). py_compile OK.
- DECISIONS.md: 3-method table, FV-folder decision, provenance RESOLVED, 4-shard build commands.

**Findings:**
- **Three methods located** (see DECISIONS table). task-specific → `gptj_fv/`;
  train → `gptj_fv_multitask_top10/`; train+test → head set exists, **FVs not built**.
- **Provenance RESOLVED:** all-tasks head set was computed on `query_split=valid`
  (per-task files suffixed `_valid`; runner passes `--query_split valid`). Metadata
  `query_split=train` is a stale `--reduce` default. Both head sets share valid/train.
- All-tasks FV build is cheap (activations cached) → single-process, no big GPU needed.

**Next:** Run the train+test FV build command in DECISIONS.md to populate
`function_vectors/train_test_selected/`. Then the 3-way comparison is fully aligned.

**Blockers:** None.

---

## 2026-06-10 — Coordinator: state reconstruction

**Status:** Bootstrapped WORKLOG/DECISIONS from artifacts (both were empty). No
worker streams had registered. Reconstructed the two active experiment lines by
reading the uncommitted scripts in `src/eval_scripts/` and the `results/` tree.

**Findings (verified against artifacts, not just inferred):**

- Two distinct research lines are in flight (see streams below).
- `results/heldout_multitask_head_eval/` contains **all 9 test tasks** (antonym,
  capitalize, capitalize_first_letter, country-currency, landmark-country,
  lowercase_first_letter, product-company, synonym, word_length) + aggregate
  summary. Complete.
- **Split provenance mismatch (needs resolution):**
  - `results/multitask_aie_heads/` (20 train tasks): metadata `query_split=valid`,
    `demo_split=train`.
  - `results/multitask_aie_heads_all_tasks/` (29 tasks): metadata
    `query_split=train`, `demo_split=train` — **but** the runner
    `run_multitask_aie_all_tasks.sh` passes `--query_split valid`. Artifact
    metadata and the shell script disagree → the all-tasks artifact may not have
    been produced by that script, or the script was edited after the run. Confirm
    before trusting the all-tasks head set.
- Linear k-sweep (`sweep_k_activation_to_fv_ols.py`) **was never run** — no
  `results/k_sweep_activation_to_fv_ols_multitask_top10/`. Only the log2 variant ran.
- Two leftover smoke dirs: `results/joint_pca_activation_to_fv_regression_smoke`,
  `results/pca_abstractive_fv_activation_scatter_smoke`.

**Next:** Owners to claim the two streams below and resolve the open questions in
DECISIONS.md. Nothing committed yet — all work is uncommitted/untracked.

**Blockers:** None for coordination. See per-stream blockers.

---

## Stream A — Multitask AIE heads → function vectors

**Owner:** (unclaimed)
**Status:** Core runs complete; held-out eval complete on train-only heads.

**Question:** Do attention heads selected by causal indirect effect (CIE)
aggregated *across many tasks* yield function vectors that transfer to held-out
tasks as well as task-specific head selection?

**Pipeline (inputs → outputs):**
1. `compute_multitask_top_aie_heads.py` (MODIFIED) — computes per-task CIE per
   (layer, head), prompt-weighted aggregate, ranks top-N. New: sharding
   (`--num_shards/--shard_index`), `--reduce`, `--all_split_tasks`,
   `--abstractive_only`, `--save_per_prompt_effects`.
2. `run_multitask_aie_all_tasks.sh` (NEW) — orchestrates sharded run over all 29 tasks.
3. `select_heads_from_cie_subset.py` (NEW) — re-aggregate any task subset from
   cached per-task CIE without recompute (requires per-prompt effects on disk).
4. FV builders: `compute_task_fv_from_multitask_heads.py` (single),
   `compute_all_task_fvs_from_multitask_heads.py` (batch, one model load, manifest),
   `compute_fv_from_selected_heads.py` (flexible head source). **Overlap — pick one
   primary; see DECISIONS open Q.**
5. `evaluate_heldout_multitask_head_fvs.py` (MODIFIED) — multitask vs task-specific
   FV steering effectiveness by layer; now emits per-task PNGs.

**Outputs that exist:**
- `results/multitask_aie_heads/` — 20-task (train) head set, top-40. No per-prompt effects.
- `results/multitask_aie_heads_all_tasks/` — 29-task head set, top-40, per-prompt effects saved.
- `results/gptj_fv_multitask_top10/` — FVs for all 29 tasks from the train-only top-10 heads + manifest.
- `results/heldout_multitask_head_eval/` — 9 test tasks, multitask vs task-specific, + aggregate summary.

**Loose ends:**
- train+test FVs not yet built — run the command in DECISIONS.md (`To run`).
- Held-out eval used the **train-only** head set, not the newer all-tasks set. Re-run to compare?
- 3 overlapping FV builders; train-only baseline can't be re-subset (no per-prompt effects saved).
- (RESOLVED 2026-06-10) split provenance: all-tasks heads are on `query_split=valid`; metadata field is stale.

**Next:** Build train+test FVs into `function_vectors/train_test_selected/`; decide
whether to re-eval against all-tasks heads; consolidate FV builders.

**Blockers:** None known.

---

## Stream B — Regress layer activations → function vectors (joint PCA)

**Owner:** (unclaimed)
**Status:** Baseline, ICL, ICL-ridge, log2 k-sweep, and layer sweeps complete.

**Question:** Can a task's function vector be linearly predicted from intermediate
layer activations in a shared low-rank PCA space? What k, layer, token role, and
ICL position decode best, and does ridge help?

**Pipeline (inputs → outputs):**
- PCA bases: `pca_abstractive_fv_activation_scatter.py` (existing, layer 11),
  `pca_abstractive_icl_examples_fv_activation_scatter.py` (NEW, per-ICL 1–4).
- Regression: `regress_activation_to_fv_joint_pca.py` (existing, OLS, incl.
  last_prompt_token role), `_icl.py` (NEW, per-ICL, drops last_prompt_token),
  `_icl_ridge.py` (NEW, ridge w/ per-cell alpha via LOO-task CV).
- Sweeps: `sweep_k_activation_to_fv_ols.py` (linear k, NOT RUN),
  `sweep_k_activation_to_fv_ols_log2.py` (log2 k, decouples activation-k from
  fv_k cap=16), `sweep_layer_activation_to_fv_ols.py` (all 29 layers; ran at k=5,
  16, 32 as separate dirs).

**Outputs that exist:**
- `results/joint_pca_activation_to_fv_regression{,_icl,_icl_ridge}/`
- `results/k_sweep_activation_to_fv_ols_multitask_top10_log2/`
- `results/layer_sweep_activation_to_fv_ols{,_full_dim_k16,_full_dim_k32}/`
- `results/pca_abstractive_icl_examples_fv_activation_scatter{,_multitask_top10}/`

**Findings (provisional, from result CSVs — re-verify before citing):**
- Best layer ≈ 8–12 (layer 11 near-peak); embedding layer worst; later layers degrade.
- Activation-side k benefit saturates ≈ 16–32; test MSE plateaus above the fv_k=16
  reconstruction floor.
- Joint-PCA baseline: test MSE minimized around k≈6–10 depending on token role.

**Loose ends / inconsistencies to resolve:**
- Two non-comparable MSE metrics in use: joint-PCA-space MSE (regress scripts) vs
  reconstructed 4096-d FV-space MSE (sweep scripts). Pick one primary.
- ICL index ranges differ across scripts (e.g., 1–4 vs 2–5). Standardize.
- Some scripts use `gptj_fv`, others `gptj_fv_multitask_top10` (the `_multitask_top10`
  suffix). Decide which FV target is canonical, or document both as intentional.
- Layer sweep is split across 3 k-specific dirs — no single combined output.
- Linear k-sweep script unrun; decide if log2 coverage suffices.

**Next:** Decide canonical metric + FV target + ICL range; consider one combined
layer×k sweep; clean up smoke dirs.

**Blockers:** None known.
