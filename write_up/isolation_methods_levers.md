# Isolation Methods — Study Levers

Reference document for the study of *isolation methods*: procedures that isolate the parts
of the model / residual stream responsible for learning simple functions in context, as a
lower-dimensional representation (a vector, a set of heads, a subspace) that can be injected
back into the model to induce the task. Terminology and notation follow
`write_up/task_id_im_subspaces.md` ($h_A$, $v_A$, $v^j_A$, $z^t_\ell$, task imitation space
vs task identification space). Model: GPT-J-6B unless stated otherwise.

Motivating problem (2026-08): zero-shot steering on held-out tasks is bimodal — several
tasks (e.g. synonym, word_length, country-currency) get essentially no steering benefit from
any current isolation product, while others steer at their few-shot ceiling. We want to
understand whether the pipeline is sensible before building on it. The levers below define
the design space; the hypotheses at the end define what a result would mean.

---

## Lever 1 — Isolation algorithm

Three algorithms are currently in play. Each produces, for a task $A$, an injectable
product; all are injected additively into the residual stream at the final cue token
(the last prompt token, i.e. the position of `"A:"`).

### 1a. CIE top-40 hard head selection (FV paper)

The Function Vectors paper's algorithm, stated generically: for every attention head $h$,
compute the causal indirect effect (CIE) — patch the head's task-mean vector $h_A$ into
prompts drawn from the **chosen optimisation setting (Lever 2)** and measure the resulting
improvement in recovery of the correct answer. Average CIE over tasks, hard-select the
**top-40 heads** as the shared head set $H$. In the original FV paper (and in our canonical
`train_varicl_top40` run) the optimisation setting is the **same-task mixed-label 10-shot
prompt** — that is one point of Lever 2, not part of the algorithm. The isolation product
is the function vector

$$v_A = \sum_{h \in H} h_A,$$

one vector per task, injected at a single layer (best layer swept, or a fixed convention
layer). Head vectors $h_A \in \mathbb{R}^{d_{\mathrm{model}}}$ are averaged over
variable-ICL (1–10 shot) ICL-correct-filtered prompts. (Implementation note: the cached
`.pt` files store the 256-dim out_proj *inputs*; the build scripts apply the head's
$W_O$ slice to realize $h_A$ — the glossary's $h_A$ already lives in residual space.)

- Selection criterion is *implicit*: per-head answer recovery in corrupted (shuffled-label)
  10-shot context — the algorithm never sees a zero-shot prompt.
- Code: `src/eval_scripts/compute_multitask_varicl_heads.py` (selection),
  `compute_all_task_fvs_varicl.py` (FV build).
- Canonical artifact: `artifacts/function_vectors/gpt-j/train_varicl_top40` (repo FV
  definition per DECISIONS 2026-07-10); head ranking in
  `artifacts/multitask_aie_heads_varicl/multitask_top_aie_heads.pt`.

### 1b. Sparse optimisation over heads, thresholded at c > 0.8

Hu et al. 2025 (arXiv:2505.05145 §3.1), stated generically: learn a coefficient vector
$c \in [0,1]^{448}$ over all heads; the candidate vector $v_A(c) = \sum_h c_h\, h_A$ is
injected once at the cue token of prompts drawn from the **chosen optimisation setting
(Lever 2)**; loss is the negative log-likelihood of the correct answer,
$-\log p(\text{full label})$ (teacher-forced), plus the sparsity penalty
$\lambda \lVert c \rVert_1$, with $\lambda$ chosen by leave-one-task-out CV over the
fitting tasks. The isolation product used downstream is the hard set: **the heads with
$c > 0.8$, summed unweighted** — i.e. exactly the 1a construction with the head list
swapped.

- Selection criterion is *explicit and differentiable*, and generic over Lever 2.
- The existing run is the special case where the optimisation setting is the **zero-shot**
  `"Q: x\nA:"` prompt (injection at block-9 output), giving the 23-head set
  `vanilla_sparse_opt23`.
- Code: `src/sandbox/sparse_head_selection/train_sparse_heads.py`; head set in
  `artifacts/sandbox/sparse_head_selection/vanilla_sparse_opt23_heads.pt`. SANDBOX status —
  not the repo FV definition.

### 1c. Per-layer mean cue-token activation (average hidden state)

No selection at all: for each layer $\ell$, the isolation product is the mean residual
stream at the final cue token over 10-shot prompts of task $A$,

$$m^\ell_A = \frac{1}{|\mathcal{P}_A|} \sum_j z^{\text{cue}}_\ell(p^j_A),$$

**one vector per layer** (28 layers), each injected at its own layer $\ell$. This is the FV
paper's "average hidden state" baseline: it carries everything the residual stream carries
at that point (task identity, format, position, token statistics), not just head-mediated
task content.

- Code: `src/compute_avg_hidden_state.py`, using `get_mean_layer_activations`
  (`src/utils/extract_utils.py`). Artifacts currently exist for antonym and synonym only
  (`artifacts/gptj_avg_hs/`); the cached 10-shot residual activations in
  `artifacts/residual_activations/gptj_56tasks_170prompts_4tokens/` (role
  `last_prompt_token`) cover all 29 tasks, so extending is cheap.

### Compatibility note

1a and 1b are both generic over the "metric to optimise" lever (Lever 2): CIE can score
heads by answer recovery in any prompt setting, and the sparse objective can be evaluated
under any prompt setting. The *existing artifacts* pin particular choices — CIE top-40 was
fit under the same-task mixed-label 10-shot setting (as in the FV paper), sparse_opt23
under the zero-shot setting. 1c has no optimisation step and is a fixed point of the
crossing (though the prompts its mean is taken over could in principle also vary).

---

## Lever 2 — Metric to optimise (fitting objective)

The prompt setting the isolation algorithm is fit against (applies to both 1a and 1b;
see compatibility note):

1. **Zero-shot accuracy** — inject at the cue token of `"Q: x\nA:"` with no demos;
   objective in practice is $-\log p(\text{full label})$ (differentiable surrogate).
2. **Uplift on same-task mixed-label 10-shot prompts** — 10 demos from task $A$ with
   labels shuffled within the prompt (the FV paper's corrupted context); uplift = steered
   accuracy − unsteered accuracy on the same prompts.
3. **Uplift on mixed-task mixed-label 10-shot prompts** — demos drawn from *multiple
   tasks* with labels mixed; **no harness or precise construction exists yet** (the only
   mixed-demo script is the bespoke `mixed_icl_antonym_synonym_topk.py`). The construction
   (which tasks mix, whether labels are additionally shuffled, how uplift is defined) must
   be specified and user-adjudicated before first use.

---

## Lever 3 — Data to fit on

1. **Train/test task split** — fit on the 20 train tasks of
   `task_splits/abstractive_train_test_tasks_29.json`, evaluate on the 9 held-out test
   tasks (landmark-country, word_length, capitalize_first_letter, synonym,
   lowercase_first_letter, product-company, capitalize, country-currency, antonym).
2. **Task-specific** — fit on the evaluated task's own data (e.g. the per-task CIE top-10
   FVs in `artifacts/gptj_fv/<task>/`, which exist for all 9 test tasks). No
   generalization claim; upper-bounds what the algorithm can do when allowed to see the
   task.
3. **All tasks pooled** — fit on all 29 tasks together; no held-out tasks; measures
   capacity of a *shared* isolation product, not transfer.

---

## Lever 4 — Metric of success (evaluation)

Same three metrics as Lever 2, now as evaluations; crossed with *where* they are measured
(train tasks, held-out tasks, or the fitting task itself). Two readout conventions coexist
in the repo and are **not numerically comparable**:

- **FV-paper harness** (`evaluate_fv` / `n_shot_eval` via
  `evaluate_heldout_multitask_head_fvs.py`): first-token top-1 accuracy of the gold answer,
  layer swept 0–27 and best layer reported, queries filtered to items the model answers
  correctly with clean 10-shot ICL. Produces both zero-shot and shuffled-10-shot curves;
  the unsteered baseline sits in the raw per-layer JSONs (`clean_topk`).
- **Sandbox convention** (`train_sparse_heads.py` / `eval_pc_projection.py`): teacher-forced
  full-label accuracy at a fixed layer (block-9 output), unfiltered valid-split queries
  (cap 100 / min 80, train top-up).

Any cross-method table must hold the readout convention fixed; when quoting older numbers,
state which convention they came from.

---

## Failure-mode hypotheses

For tasks with poor held-out zero-shot steering (synonym, word_length, country-currency,
partially antonym/product-company):

- **H1 — Overfitting to train tasks.** The isolation method (head set / basis) fits the
  train tasks and fails to transfer. *Test:* task-specific fitting (Lever 3.2) should
  rescue the failing tasks if H1 holds.
- **H2 — A single direction is not enough.** No single injected vector can induce these
  tasks zero-shot, regardless of how it is found. *Test:* directly optimise an
  unconstrained $v \in \mathbb{R}^{4096}$ per task against the zero-shot loss (small delta
  on `train_sparse_heads.py`'s machinery — does not currently exist); if even this upper
  bound fails, no head-selection method can succeed at 0-shot on that task.
- **H3 — Wrong success metric.** Zero-shot induction is simply the wrong bar for some
  tasks; the honest metric is uplift in corrupted 10-shot context (where the format
  scaffold exists and steering supplies task identity). *Test:* score the same isolation
  products under Lever 4.2/4.3; if failures vanish there, adopt the mixed-context metric
  as primary.

These are not exclusive: H2 and H3 can both be true (a task can need context for the format
*and* more than one direction for the content).

## Current evidence snapshot (2026-08-12)

- **Task-specific FVs also fail zero-shot on the failing tasks** (zs_task_specific, FV-paper
  harness, best layer: synonym 0.18, word_length 0.09, country-currency 0.36 —
  `results/steering_vector_comparison/heldout_varicl_nheads_sweep/nheads_sweep_best_layer.csv`).
  Weakens H1 as the sole explanation.
- **Head sets differ zero-shot but tie in corrupted context**: on the 9 held-out tasks,
  sparse23 0.40 vs top-40 0.25 zero-shot (best layer), but both 0.78 on shuffled-10-shot
  (sparse-heads sandbox artifact, Result 2 + appendix figure, PNG extracted to
  `results/sandbox/sparse_head_selection/heldout_zs_bylayer_top10_top40_sparse23.png`).
  Keeps H3 live: the metric, not the head set, drives the "failure".
- **Zero-shot steerability is bimodal across held-out tasks** under both head sets and both
  readout conventions; the failing half is stable (see 2026-08-11 test-task eval,
  `results/sandbox/sparse_pc_selection/top29pc_projection_vs_fullfv_testtasks.csv`).
- **Train-task PC subspaces of per-prompt FVs do not transfer** (coverage, not compression:
  held-out FV steering content sits in the low-variance tail of the train stack) —
  `results/sandbox/sparse_pc_selection/`, WORKLOG 2026-08-11. Relevant to H1/H2: even
  within one head set, task content is spread beyond a small shared subspace.
- **No instruction-prompt zero-shot baselines exist anywhere** (all prompts use
  `instructions: ""`): "can GPT-J do this task zero-shot at all, under any prompt" is
  currently unmeasured — needed to separate H2 from "the model can't do it, period".
