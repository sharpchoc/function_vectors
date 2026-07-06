# DECISIONS

Reusable conventions, project decisions, and open questions. Append; date entries.
Resolved questions move from "Open" to "Decided" with the rationale.

---

## 2026-07-04 — Pair-diff pre-images: invert the FV DIFFERENCE, not per-task pre-images (Stream S)

- **When a direction is needed for a task-pair difference under the ridge maps, damp the inversion
  of the DIFFERENCE target** `fv_A − fv_B` directly (`fit_ridge_preimages_multicell.py`,
  `pairdiff_preimages/` banks). The Tikhonov-damped inverse is linear in the target only at FIXED
  gamma; subtracting two per-task damped pre-images mixes two different gammas (the norm-cap
  selection rule picks per-target) and injects damping artifacts into the difference direction.
  Exact (gamma=0) inversion commutes with subtraction, so only the damped variant cares.
- **Unit-normalizing does NOT make exact vs damped pre-images interchangeable.** cond(W)~1e9-1e10
  means the exact inverse's *direction* (not just norm) is dominated by the smallest singular
  values — Stream S confirms it empirically: exact pre-image direction explains ~random-floor
  levels of the two-shot pair-diff variance while the damped direction explains up to 0.43
  (uncentered, digits query_final L4).
- **Reusable intermediates:** `artifacts/preimage_pairdiff/train_varicl_max4_top40/<role>_icl{k}/maps/`
  holds materialized 4096x4096 ridge maps (W_std fp16 + scaler) for 6 cells x 28 layers with the
  varicl-max4-top40 FV target — any future pre-image/decoding study at those cells can skip the
  ~25 s/layer refits. Same layout as Stream R's `artifacts/preimage_steering/` maps (icl10
  pre_label, train_selected_top40 target).
- **Centered vs uncentered explained-variance for paired diffs diverge wildly** (digits demo2_label
  L4: top-1-direction bound 0.85 uncentered vs 0.13 centered): the pair-diff mean is a huge shared
  component. Report both; say which one a claim uses.

## 2026-07-04 — FV .pt files are dicts (weights_only) + FV-projection-ablation design notes (Stream T)

- **`compute_function_vectors.py` saves each FV as a DICT** `{'function_vector' [resid_dim],
  'top_heads', 'n_top_heads'}` — NOT a bare tensor. `top_heads` holds numpy int64 scalars, so on
  torch≥2.6 `torch.load(p)` fails (default `weights_only=True`). Load pattern:
  `obj = torch.load(p, weights_only=False); fv = obj['function_vector'].reshape(-1).float()`.
- **GPT-J task-specific FVs live in `artifacts/gptj_fv/<task>/<task>_function_vector.pt`** (top-10,
  n_shots=10). The `_digits` variants didn't exist and were computed 2026-07-04 (mean-acts 100 trials +
  indirect-effect 25 trials ≈ 9 min/task on an RTX 4000 Ada, 20 GB).
- **Logit readout without the full lm_head:** run `model.transformer(...)` under the edit hook, then
  `model.lm_head(last_hidden[:, -1, :])`. GPTJModel already applies `ln_f`, so `last_hidden_state` is
  post-norm — applying lm_head to the qfinal slice gives correct next-token logits while avoiding the
  `[B, seq, 50400]` allocation.
- **FV-projection-ablation ordering:** when steering (add) and ablation (project-out `u`) target the
  SAME token (qfinal), apply steer-THEN-ablate within each layer's hook.
- **Localize the steer, not just the read (Stream T lesson):** "steer across all layers" can mean two
  very different experiments. Adding the steer vec at ALL 29 layers at once (incl. directly at qfinal)
  swamps a single-direction ablation → retention≈1, misleadingly looks like the FV doesn't mediate.
  Injecting at ONE layer at a time (swept 0..28, as in the heatmap studies) while ablating F'⊥F at
  qfinal reveals the real picture: peak steering is MID-NETWORK (L10–14) and the ablation removes
  ~55–67 % of the localized gain at α=2 (peak-layer retention 0.33–0.48 across the 4 directions).
  Retention rises with α (steering brute-forces through other directions), so the gentle-α regime is
  the cleaner test. Default to a per-layer sweep when asking whether a direction *mediates* an
  intervention — an all-layers intervention conflates "does this direction matter" with "can I
  overwhelm the readout." (Supersedes the first pass, which used an all-layers steer and wrongly
  concluded mediation≈0.)

## 2026-07-03 — 10-shot strip steering (Stream Q): lm_head OOM fix + long-prompt batch tuning

- **When you only need hidden states, call `model.transformer(...)`, NOT `model(...)`.** `model(...)`
  computes `lm_head(hidden_states)` → a `[batch, seq, vocab≈50400]` fp32 tensor (e.g. 1.25 GB at
  batch 64 × seq 97), which we never use (we read residual streams via baukit TraceDict). This is what
  OOM'd the 10-shot run at the unsteered pass. `model.transformer(input_ids, attention_mask)` runs the
  embedding + blocks (where the hooks live) and skips lm_head/final head — identical captured outputs,
  ~1.25 GB less, and faster. The 2-shot script used `model(...)` and got away with it only because its
  prompts were short.
- **Batch vs the 24 GB 4090 for these 29-layer `retain_output` captures:** batch 64 pins memory at the
  ~23.4 GB ceiling (a longer-prompt batch then OOMs); **batch 48 sits at ~21.7 GB** with margin — use 48.
  The 29-layer retain_output (holds every block's `[B,seq,D]` output each forward) is the memory AND
  time driver, so runs are slow (~80 s/grid) regardless of the short prompts. Set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Kill remote runs by PID, not `pkill -f <pattern>`** when the pattern also appears in the killing
  shell's own command line — `pkill -f steer_tenshot_strip` matches the ssh shell running it and
  self-terminates (SSH exits 255, though it does kill the target first). Safe form:
  `ps -eo pid,args | grep '[s]teer_tenshot_strip_cos' | awk '{print $1}' | xargs -r kill -9` (the `[s]`
  regex trick prevents grep from matching itself).
- **Unmatched-demo mean-difference steering** (10-shot, no matched labels): steer_vec at a structural
  slot = difference of the two tasks' MEAN activations there. Result: the steerable signal lives almost
  entirely in demo LABEL tokens (input/pre-label ≈ 0), mid-layer→late-read, distributed across demo
  positions (not just the last). Only the read token (qfinal) is byte-matched, keeping baseline_cos valid.

## 2026-06-30 — Two-shot token-pair cosine heatmaps (Stream P): conventions + GPU-pod gotchas

- **Steering ≠ patching ⇒ byte-identity is NOT required.** The readout adds `α·mean_pairs[tgt−src]` to
  a residual stream; that direction is well-defined at any token. So the search space can include the
  demo-2 INPUT token (`t2 input2`) even though it differs across functions. BUT its steer vector then
  carries a **lexical/content** component (not pure function context; at layer 0 it is literally a mean
  token-embedding diff) and its READ baseline cos is < 1 (vs ≈1 at the 5 clean tokens). Keep it, but
  flag it (suptitle + summary `note_input2`); don't compare its magnitude naively against clean tokens.
- **Structural asserts that hold exactly:** lower-triangle `k ≤ i` ≡ 0 (a position-i edit at layer i
  reaches a later position only at read layers `> i`, even across different tokens); embedding column
  `i=0` ≡ 0 for CLEAN source tokens (same token ⇒ identical `transformer.drop` output ⇒ steer_vec=0; GPT-J
  has no additive positional emb at the embedding, so position shifts don't matter). `input2` is the
  documented exception (assert skipped for it).
- **Token positions for the 6-token 2-shot search space** come from the inlined `selected_token_records`
  (loops both demos, icl 1 & 2): label1=`last_label_token@1`, prelabel2=`pre_label_token@2`,
  label2=`last_label_token@2`, qfinal=`last_prompt_token@None`; the two INPUT tokens are `prelabel2−1`
  and `qfinal−1` (no named role). Assert strictly-increasing positions per prompt.
- **Memory-lean pattern for ≤20 GB cards:** keep captured acts `[N,6,29,D]` in fp16; compute steer_vec
  and baseline cos in a chunked loop over pairs (CB≈64) to avoid full-N float copies; convert only
  per-read-token slices to float in the inner loop. Fits gpt-j-6b + acts at ~18.5/20 GB. Full-precision
  copies OOM.
- **GPU-pod gotchas (RunPod):** (1) a fresh pod created with `--networkVolumeId` but the DEFAULT
  `--volumePath` mounts the shared volume at **`/runpod`**, NOT `/workspace` (the templated pods use
  `/workspace`). Find the repo at `/runpod/function_vectors`, HF cache `/runpod/.cache/huggingface`.
  (2) Other pods only trust the USER's SSH key; to reach a self-created pod, pass our pubkey via
  `--env PUBLIC_KEY="$(cat ~/.ssh/runpod_gpu.pub)" --startSSH`. (3) The project docker image is now the
  upgraded **torch 2.8 / cu128** stack (runs on Ada AND Blackwell sm_120) — but `matplotlib` is ABI-broken
  against numpy 2.1.2; `pip install -U matplotlib` (→3.11) before plotting. Compute scripts (numpy+torch
  only) are unaffected.
- **Plotting is CPU-only** — the plot scripts just read saved `.npy/.csv` grids. Install `numpy matplotlib`
  on the editing (CPU) pod (`python3 -m pip install numpy matplotlib`; note the system has TWO pythons —
  `/usr/bin/python3.10` is the one the scripts use, NOT the `/usr/local/bin/pip` python3.12) and iterate on
  figures locally — no GPU pod needed once the grids exist on the shared volume.
- **Comparability across many sub-plots:** when per-figure color scales span a wide dynamic range (here
  ~100× across token-pairs), per-figure scaling is misleading. Add (a) a single GLOBAL vmax across all
  grids, (b) a token×token "matrix of heatmaps" master figure (spatially arranged, one colorbar), and
  (c) a scalar peak-Δcos overview matrix (annotated). Keep per-pair per-scale figures for drill-down.

## 2026-06-29 — Attention KNOCKOUT (Stream O): monkeypatch GPTJAttention._attn, pre-softmax mask

- **GPT-J (transformers 4.49.0) uses EAGER attention by default** (no `GPTJSdpaAttention` class), so
  pre-softmax attention scores are directly editable. `load_gpt_model_and_tokenizer` already loads eager;
  assert `model.config._attn_implementation == "eager"`.
- **Knockout recipe:** monkeypatch `GPTJAttention._attn` with a faithful copy of the 4.49.0 source, and
  right AFTER `attn_weights = attn_weights + causal_mask` and BEFORE softmax, set the target (query,key)
  entries to `torch.finfo(attn_weights.dtype).min`. After softmax the key's weight is 0 and the row
  renormalizes to sum 1 — "ablate by zeroing the score, attention still sums to 1". Per-row, all-heads
  indexing: `attn_weights[rows[:,None], heads[None,:], q_idx[:,None], k_idx[:,j][:,None]] = mn`.
- **Threading per-row indices:** stash `(q_idx[B], k_idx[B,K])` on each `model.transformer.h[l].attn._ko`
  before the forward (helper `set_knockout`/`clear_knockout`); the patched `_attn` reads `_ko` (None =
  clean). No baukit needed — read logits from the model output. Editing the 4D `attention_mask` does NOT
  work for per-row key positions (it's shared across the batch); editing `self.bias` fails under
  left-padding (same for all rows). Monkeypatch is the way.
- **Verify** with one `output_attentions=True` forward: knocked-out key weight ≈ 0, edited rows sum to 1.
- **Result:** qfin reads the task directly from the LABEL tokens, not from the demo-2 pre-label (cutting
  the pre-label edge ≈ cutting a structural-token edge; cutting the label edges collapses the task).

## 2026-06-27 — Patch-onset layer (`--patch_from_entry`) + regime folders; all-layers check

- The two patching experiments take `--patch_from_entry` (default 6 = "L6 and above"; 0 = all entries
  incl. embedding). Output goes to a per-experiment regime subfolder `<exp>/{L6_and_above,all_layers}/`
  (`"all_layers" if pe==0 else f"L{pe}_and_above"`); plotters take `--regime`. Implemented by rebinding
  the module global `PATCH_FROM_ENTRY` at the top of `main()` so the existing hooks/asserts pick it up.
- **Patching the embedding (entry 0) is a no-op for byte-identical positions** (labels, query, pre-label
  in the paired design) — so the label-follow result is unchanged between regimes. It is NOT a no-op for
  the demo INPUT tokens (they differ across base/target): patching entry 0 there swaps the literal input
  word, which is why the interval grid's demo2-input cells jump in all_layers. Keep this in mind when
  reading any "all layers" patch that touches positions whose tokens differ across the pair.
- Conclusion: the label-token findings (Streams M/N) are robust to patch onset; document the all_layers
  run as a completeness check, not a new result.

## 2026-06-26 — Isolating a token's DIRECT effect (Stream N): pin-everything-else-to-base

- **"Open" set-patching does NOT isolate the patched tokens.** Overwriting token positions ← target with
  nothing else frozen lets the patched info be **relayed**: later tokens attend to the patched positions
  (from entry 7+) and pass the signal on. To measure a token set's DIRECT contribution to the output,
  also **pin every other non-output token to its base value** at the patched entries
  (`h[:, :-1, :] = base_full[:, :-1, e]`, then set the patched roles → target; leave the last/output
  column free). Then only the patched positions carry target and only the direct patched→output path
  remains. The gap (open recovery − isolated recovery) = the relayed share.
- Requires capturing the base **full** residual stack (all positions, patched entries) — keep it
  per-chunk (seq differs per chunk under left-pad); ~3GB at N=544 on the 4090, fits with batch 64 +
  `expandable_segments:True`. (Stream M only needed the 6 named positions; isolation needs all positions.)
- **Result:** label tokens recover 74% (words) / 95% (numbers) of the task switch even fully isolated →
  the labels drive the output largely directly; pre-label ≈ 0. See [[stream-m-interval-patch]] WORKLOG.

## 2026-06-26 — Six-token interval activation-PATCHING (Stream M): mechanics + gotchas

- **Interval-patch mechanic** (vs Δ-vector steering): run the base prompt and, at residual entries
  `6..28`, **overwrite** (assign, not `+=`) token `i` ← the *other* prompt's activation and token `j` ←
  the base's *own clean* activation. `i` then carries the other function for L6+; `j` is pinned to
  original (blocks relay). Hooks fire in forward order, so a downstream token `k` first differs at the
  entry **after** the first patched entry → `Δcos(k, entry 6) ≡ 0` (good cheap invariant; asserted <1e-3).
- **Pinning the OUTPUT token (`qfin`) to original zeroes any logit effect exactly** — the entire `j=qfin`
  column is +0.000. Expected, not a bug: the final-position residual is restored to baseline so logits
  are unchanged. Keep the column (it's the causal boundary), don't special-case it.
- **Reuse, don't reinvent the trace plumbing:** the exact 2-arg `(output, layer_name)` factory closure,
  tuple-vs-tensor dispatch (`output[0]` for `transformer.h.*`, bare tensor for `transformer.drop`), and
  left-pad per-row position table all carry over verbatim from `steer_label_cos_heatmap.py`. The ONLY
  change for patching is assign instead of add, and a per-row position **vector** (`pos6[:,i]`) so the
  hook can edit two positions at once.
- **Positions are NOT fixed even when every word is single-token.** In-context tokenization shifts the
  query block by ±1 token for some words (e.g. `acceptable`), so the "all single-token ⇒ fixed index"
  shortcut fails. Always compute positions **per row** and align by ROLE (the named 6 positions), never
  by absolute index — base and target prompts of a pair can also differ in length.
- **`retain_output` over all 29 layers is the memory bottleneck**, not the stored captures: it holds
  every position × every layer for the whole batch. batch 256 OOMs the 4090 (~23 GB); **batch 64 +
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** is ample and still ~17 short forwards/task.
- **Single-token-input filter is free** for antonym/synonym: 100% of demo-input options for the 544
  used labels and 100% of the shared query pool are single-token (verified), so requiring it drops 0
  tuples. Demo-input-1 and demo-input-2 sample from the SAME distribution (`o2i[L]`); only the query
  comes from the narrower `shared_in` pool.

## 2026-06-25 — Layer×layer label→query-final cosine-shift heatmap; env + reuse gotchas

New deliverable `src/eval_scripts/steer_label_cos_heatmap.py` (Stream L): 29×29 (intervention-layer ×
read-layer) heatmap of how injecting `α·steer_vec(i)` at the demo **label** token pushes the source
prompt's **query-final** activation toward the (unsteered) target's, measured as `Δcos = cos(steered
src_final, tgt_final) − cos(src_final, tgt_final)`, meaned over overlapping prompt pairs. Layer 0 =
embedding (`transformer.drop`), 1–28 = block outputs. Reusable facts:

- **The grid is structurally upper-triangular and the embedding row/col is exactly 0.** A label-token
  edit reaches the query-final token only via *later* blocks' attention, so `Δcos = 0` for read
  `k ≤ intervene i` (incl. diagonal) — use this as a free correctness check. Embedding diff is 0
  because the label + query-final tokens are byte-identical across f1/f2 (the overlapping design), so
  baseline cos at layer 0 = 1 and `steer_vec[0] = 0`.
- **baukit `edit_output` hooks MUST be exact 2-arg `(output, layer_name)` closures** (already noted
  2026-06-11). Hit again here: passing per-chunk state via extra default-kwargs made the hook silently
  no-op (Δcos = 0 everywhere). Fix = a factory returning a 2-arg closure that captures state via scope.
  Also handle BOTH output shapes: `transformer.h.{l}` returns a tuple (edit `output[0]`), but the
  embedding module `transformer.drop` returns a bare tensor (edit `output`).
- **`matplotlib` 3.7.1 is ABI-incompatible with this box's `numpy` 2.1.2** (`numpy.core.multiarray
  failed to import` on `import matplotlib`). Fixed by `pip install -U "matplotlib>=3.8"` → 3.11.0.
  Any plotting script will fail to import until this is in place.
- **Shared output dir → name summaries per task** (`<task_pair>_summary.json`, not a single
  `summary.json`) so a second task's run doesn't clobber the first's metadata. Grids/PNGs/CSVs are
  already per-task-named.
- **Finding:** mid-layer intervention band (words L5–12 peak L7–9; digits earlier, L4–8) feeding read
  L16–18, dead by intervene ~L16 — mirrors the 1-D causal window. Digits steer ~2× words; α=2 > α=4 at
  the peak (the toward-target-cosine metric overshoots sooner than the logit-flip readout). Absolute
  shifts are small (baseline cos near ceiling ~0.98–0.99); the *location* in the grid is the signal.

---

## 2026-06-19 — Results layout: gitignored `artifacts/` vs tracked, direction-bucketed `results/`

**Convention (use for all new scripts):** import paths from `src/utils/paths.py`; never hardcode
`results/...`. Roots are REPO_ROOT-anchored and env-overridable (`FV_ARTIFACTS_ROOT`, `FV_RESULTS_ROOT`,
`FV_LOGS_ROOT`):
- **`ARTIFACTS_ROOT`** (`artifacts/`, gitignored) — intermediate caches: activations, captures, function
  vectors, head selections, scratch. Write all recomputable `.pt`/`.jsonl` here.
- **`RESULTS_ROOT`** (`results/`, TRACKED) — study deliverables, in 5 bucket constants by research
  direction: `AMBIGUOUS_DIR` (d1), `LABEL_GEOMETRY_DIR` (d2), `FV_FORMATION_DIR` (d3),
  `STEERING_COMPARISON_DIR`, `GENERAL_DIR`. Put figures in `<bucket>/<exp>/figures/` or `<bucket>/figures/`.
- **`LOGS_ROOT`** (`logs/`, gitignored) — run logs.

**What's committed:** under `results/` only `*.png/*.pdf` + summary `*.csv`/`*.json` (binaries `*.pt/*.npy/
*.npz/*.jsonl/*.log` and raw dumps `per_query.csv`/`raw.json`/`recreate_fig8/**/*.md` are globally
ignored). **Exception — head selections ARE committed** even though they live in `artifacts/`: the top-N
head rankings are the output of the expensive CIE computation, so `multitask_top_aie_heads{.pt,_metadata.json}`,
`heads.pt`, `heads_metadata.json`, `fv_manifest*.json`, `selected_heads.json` are re-included via
`.gitignore` negations (so FVs can be rebuilt without recomputing CIE). Scratch dirs (`artifacts/_*/**`)
are re-ignored.

**Gotchas:** (1) `.gitignore` has NO inline comments — `pattern  # note` makes the comment part of the
pattern; put comments on their own line. (2) To track files under an ignored dir, ignore by glob
(`artifacts/**`) + re-include directories (`!artifacts/**/`) + re-include specific filenames, with the
negations LAST so they beat earlier rules (incl. the global `*.pt`).

---

## 2026-06-19 — TWO-shot matched-label paired captures (Stream K): construction + 5-role schema

Extension of the 1-shot paired-difference design to **two demos**. Use this construction for any 2-shot
paired study:
- **Matched labels, distinct within a prompt:** demos carry labels `(L1,L2)`, both from the shared-output
  pool, with `L1≠L2`. The label SEQUENCE is identical across the two functions (f1 prompt and f2 prompt
  use the same `(L1,L2)` and the same query); only the demo INPUT tokens differ by function (antonym-of-L
  vs synonym-of-L; for digits next-input `L−1` vs prev-input `L+1`). So at every demo-label / pre-label
  position the token is byte-identical across f1/f2 — pure contextualization, as in the 1-shot design.
- **Enumeration:** one tuple per shared-output label word `w` → `L1=w`, `L2`=random distinct label, query
  from shared-input pool minus `{L1,L2,4 demo inputs}`, deterministic per `(seed,task_pair,w)`. Yields
  n_prompts ≈ |label pool| (≈544 ant/syn, ≈198 digits) — comparable to the 1-shot run.
- **NO train/test split** for activation gathering (full-pool capture). The split only matters where Δ is
  fit-then-applied (steering); geometry is descriptive. (User-confirmed.)
- **5 captured roles per prompt:** `demo1_prelabel`, `demo1_label`, `demo2_prelabel`, `demo2_label`,
  `query_final`. The `*_prelabel` roles are the `A:` token before each demo label (`pre_label_token` @
  icl_example_index 1/2); `demo2_prelabel` was the specifically-requested new position. `selected_token_records`
  numbers demos **1-based** (icl_example_index 1,2) and handles 2 demos with no special-casing.
- **Script:** `src/eval_scripts/capture_and_grade_twoshot_paired.py` (sibling of the 1-shot version;
  reuses `get_residual_stack`/`selected_token_records`/`flush_shard` + `word_pairs_to_prompt_data`). Output
  is an INTERMEDIATE → `artifacts/twoshot_paired_graded/<pair>/` (git-ignored `ARTIFACTS_ROOT`), NOT
  `results/`. First-token rank grading stamped per row.
- **Finding:** a second matched demo ~doubles first-token top-1 on word tasks (antonym 0.232→0.439,
  synonym 0.066→0.221) and cuts the 1-shot copy-the-query failure; digits at ceiling (0.98–0.99).

---

## 2026-06-19 — Logit-readout switch-steering + clean train/test split (preferred over sample+judge)

For task-switch steering where the gold answers are **single tokens**, prefer the logit readout over
the sample+judge harness — it's exact, ~100× cheaper, and needs no OpenAI key.

- **Metric:** steer source→target, then in ONE forward read `logit(target_gold) - logit(source_gold)`
  at the query final token. α=0 = clean baseline (negative = prompt favors source answer); steering
  pushes it up across zero = the switch. Script: `src/eval_scripts/steer_switch_logit.py`,
  plot `plot_switch_logit.py` (shaded band = 95% CI of the mean over the n=100 test queries = mean ±
  1.96·SEM/√n, NOT per-query spread).
- **DIGIT numerals are all single GPT-J tokens** (verified 0–250, with leading space) — vs word-form
  numbers where compound forms >20 are multi-token. So for number tasks use the digit datasets
  (`dataset_files/abstractive/{next,prev}_number_digits.json`, inputs 1–200) to get single-token
  labels/golds. Word next/prev dropped for single-token analyses.
- **CLEAN TRAIN/TEST SPLIT (no leakage in either the ICL example or the final query):** the two
  positions Δ is read at are the demo **label token** (Δ_label) and the query **final token**
  (Δ_final). So hold out a TEST set of 100 label-overlap (shared-output) words AND 100 input-overlap
  (shared-input, single-token-gold-both) words; estimate Δ ONLY from disjoint TRAIN words, and draw
  the 1-shot demo for test prompts only from non-test words too (so a test query can't sneak in via the
  demo slot). Use **equal n_train (=100 prompt pairs) for every task pair** so Δ quality is comparable.
- **Derive Δ from our own forward passes on train pairs**, NOT `load_capture_diffs` over the full-pool
  `oneshot_paired_graded` capture — the latter pools over the whole shared vocabulary and overlaps the
  eval queries (train-on-test). Only hold out the **test query words** from Δ (the positions read);
  excluding their *golds* too is unnecessary and decimates the small digit pool (collapsed Δ to 5
  words in one run).
- **No correctness filtering on Δ collection (deliberate):** activations are averaged over all train
  pairs regardless of whether the model answers correctly — this diverges from the standard
  ICL-correct-filtered FV recipe (`get_mean_head_activations`) and likely leaves Δ noisier (1-shot
  competence is modest), but the user chose to keep it blind. Revisit if Δ looks weak.
- **Finding:** two distinct causal windows by injection site — demo-label peaks early (L4–8, dead by
  ~L14); final-prompt-token peaks later (L10–12) and reaches larger logit-diffs. All effects gone by
  ~L16–20.

---

## 2026-06-18 — Behavioral switch-steering harness + GPT-4 judge must be parallelized

`src/eval_scripts/steer_switch_judge.py` (Stream E) is the behavioral analog of the logit-level
`steer_label_to_query.py`: inject `sign·α·Δ_site(L)` into a source-task 1-shot prompt, **generate**
n=10 temp-1 samples, **GPT-4-judge** them for the *target* task, sweep layer×α → accuracy curves.

- **Reuse, don't recompute:** the per-site steering vectors come straight from the existing
  `results/oneshot_paired_graded/<pair>/` captures via `load_capture_diffs` (`source` role →
  Δ_label, `target` role → Δ_final). No new capture. sign = +1 if target is f1 (Δ is f1−f2) else −1.
- **Steering hook during generation:** baukit `edit_output`, add the vector at a per-row position
  **only on the prompt forward** (`output[0].shape[1] > 1`); cached single-token steps are skipped so
  the perturbation propagates through the KV cache. Replicate prompts ×n_samples explicitly
  (prompt-major order) + left-pad; label-site idx = pad_len + label_pos, final-site idx = −1.
- **α=0 baseline is layer/site-independent** → generate & judge it once per direction, draw flat.
  Sanity: an unsteered *source*-task prompt judged for the *target* task scores LOW (antonym-demo →
  synonym ≈ 0.08), confirming headroom for steering to lift.
- **GPT-4 judge MUST be concurrent.** The stock sequential `judge()` in `judge_oneshot_paired.py`
  runs ~7 batches/min (~8s/request) → ~27h for this sweep's 11k batches. A `ThreadPoolExecutor`
  (`judge_parallel`, `--judge_workers 40`, order-preserving, with exp-backoff retry on transient
  HTTP/JSON errors) cuts it to ~40 min. Use this pattern for any large judge run.
- **Throughput seen on the 24GB Blackwell:** GPT-J generation ~140 samples/s (gen_batch=20 queries ×
  n_samples=10 = 200 rows/generate call); the full 564k-sample sweep generated in ~50 min.
- **OPENAI_API_KEY is NOT in this session's env or /proc/1/environ** (unlike earlier streams). An
  interactive-shell `export` does NOT reach the agent's fresh shells. Workable route: write the key to
  a gitignored file (`.openai_key`, chmod 600) and `set -a; . ./.openai_key; set +a` before the run.

---

## 2026-06-18 — variable-ICL FV variant capped at max 4 demos (top-40); zero-shot steering cost is small

New `train_varicl` sibling that caps the per-prompt random ICL count at **1–4 demos** (`--max_shots 4`)
instead of 1–10, top-40 heads pooled over the 20 train tasks. Single behavioral change; isolated roots
so the original max-10 artifacts are untouched. NEW scripts: `src/eval_scripts/run_varicl_max4_pipeline.sh`
(build driver — handles the stage-2 gotcha: test-task activations computed to `results/_varicl_testtasks_max4/`
then copied into the main root before the 29-task FV build) and `src/eval_scripts/evaluate_heldout_varicl_max4.py`
(steering eval + 3-series plot; reuses `evaluate_fv`/`get_filter_set`/`summarize_results`, reads the
max-10 N=40 curve from `results/heldout_varicl_nheads_sweep/<task>/nheads_sweep_by_layer.json` and the
task-specific curve from `results/heldout_multitask_head_eval/<task>/comparison_summary.json` — no
recompute of those lines, no no-FV-baseline recompute). Artifact roots:
`results/multitask_aie_heads_varicl_max4/`, `results/function_vectors/gpt-j/train_varicl_max4_top40/`
(29 FVs, n_top_heads=40), `results/heldout_varicl_max4_top40/` (9 per-task PNGs + AGGREGATE + summaries).

**FINDING:** shrinking the extraction ICL window to ≤4 demos costs a little **zero-shot** steering
(mean best-layer top-1 0.353 vs max-10's 0.400; task-specific 0.483) and **~nothing few-shot**
(10-shot-shuffled 0.777 vs 0.784). Same mid-layer (L6–12) causal window, max-4 a notch below max-10
throughout; no task collapses. So the variable-ICL FV is fairly robust to a shorter extraction window —
context at inference time compensates almost entirely, and even zero-shot the FV is only modestly weaker.
The top-40 pooled head set is the canonical varicl ranking ((9,14)(15,5)(8,1)…), so the shot-cap changes
the *mean activations* (and thus FV magnitude/direction) more than the head *selection*.

**PCA-ridge activation→FV heatmap (k_act=16, k_fv=16) on the max-4 FVs:** reran
`regress_activation_to_fv_pca_ridge.py` (×10 ICL shards) + `merge_fulldim_ridge_results.py` with the
FV target re-pointed to `train_varicl_max4_top40` (cached residual activations in
`results/residual_activations/` are FV-target-agnostic → only `--fv_root`/`--output_dir` change). Out:
`results/pca_ridge_activation_to_fv_varicl_max4_top40/`. Same canonical mid-layer (L9–15) bowl; best
cell icl09/pre @ L13 = 0.1596 (FV-PC floor 0.1433). **Compare via gap-above-floor** (the
FV-target-agnostic regression-quality metric, since each FV set has its own 16-PC variance floor):
train_selected 0.0155, varicl_max4 0.0162, varicl_top40 (max10) 0.0214 → the max-4 FVs are as
regressable from the residual stream as train_selected and a touch tighter than the max-10 varicl FVs.
Do NOT compare absolute test_mse across FV targets (different floors); use gap-above-floor.

## 2026-06-18 — Paired 1-shot captures carry extra correctness notions (n=5 temp=1 fraction + prob-gated)

The paired 1-shot activation rows (`results/oneshot_paired_graded/{antonym_synonym,
next_number_prev_number}/shard_*.pt` + `grading.json`) now hold these per-prompt correctness labels,
all GPT-4.1-judged with the strict same-word-is-false prompts:
- `judge_top1` — greedy decode (original).
- `frac_correct_temp1_n5` (float) + `n_correct_temp1_n5` (int 0..5) — **5 temperature=1 samples**
  (seed 42), each judged; the row stores the FRACTION correct (k/5). This REPLACED an earlier
  single-sample `judge_top1_temp1` (one noisy Bernoulli draw) — the n=5 fraction is the robust signal;
  the single-sample field was dropped from all rows. `sample_judge_oneshot_paired.py` (draws via
  `num_return_sequences`, reuses the judge helpers).
- `greedy_answer_prob` (float) — model's probability of the greedy answer = product of greedy
  per-token (top-1) probs over the answer span (tokens before first "\n"); equals first-token prob for
  single-token answers, joint product for multi-token (numbers). NB it is the probability of the
  *greedy path itself*, NOT the gold answer's probability (the answer judged IS the greedy output).
- `judge_top1_p50` (bool) = `judge_top1` AND `greedy_answer_prob` ≥ 0.50. (Threshold walked
  0.70 → 0.60 → 0.50; each step a free instant re-gate from the stored float, no model rerun — the
  payoff of saving the continuous prob. p70/p60 fields removed. To change again, re-gate from the float.)

**Activations are decode-independent** (same forward pass), so all decode-dependent labels (temp=1
fraction, greedy prob/gate) are computed retroactively and stamped by match key
`(function_task, output_word, query)` — same scheme as the original `judge_top1` tagging. The prob
gate makes NO API calls (reuses greedy `judge_top1`).

**Lesson / caveat:** in 1-shot, open-ended word tasks (antonym/synonym) are rarely high-confidence
(mean greedy answer prob ~0.17) so even a 0.50 gate keeps ≤19 prompts (synonym 0) — too few for a
correct-vs-incorrect geometry contrast on those tasks; use the continuous `greedy_answer_prob` or
`frac_correct_temp1_n5` as covariates, or the number pair (gate keeps 47–98). Confidence ≠ correctness
(synonym: 8 prompts ≥0.50 but 0 judge-correct; prev_number 58 ≥0.50, 47 correct). Under temp=1 sampling
numbers drop vs greedy (greedy argmax was right; sampling injects errors) while synonym RISES (.143 →
mean frac .229 — many valid non-argmax answers). any-correct ≫ all-correct (most prompts are sometimes
but not always right), which is exactly the spread the single sample missed and the fraction captures.

## 2026-06-18 — magnitude/identity datasets DOUBLED with decimals (canonical files overwritten)

`dataset_files/ambiguous/{magnitude,identity}.json` were extended from 300 → **600 entries**
(300 integer + 300 decimal, ≤3 dp) by `dataset_files/generate/add_decimals_magnitude_identity.py`.
Integer-only originals backed up at `{magnitude,identity}.int_only.json` (restore from these to
revert). Decimal entries mirror the integer structure: +d overlap (mag=id=d), −d differentiator
(mag=d, id=−d). Split is now overlap=300 / differentiator=300. **Impact:** any re-run of FV
extraction / disambiguation evals on these tasks now includes decimals; previously-saved FV `.pt`
files (e.g. `gptj_fv_ambiguous_constrained*`) are integer-derived and unchanged. When judging
numeric-answer correctness, use **numeric equality** (`float(a)==float(b)`), NOT `normalize_answer`
(it strips `.`). Decimal labels are multi-token in BOTH tasks (e.g. ` 97.96`→`[' 97','.','96']`),
so the first/last-label-token capture finally matters for magnitude too.

## 2026-06-16 — Qwen3-8B 3+1+1 disambiguation + CAVEAT: first-token top-k invalid for digit-splitting tokenizers

Ran the Stream-F 3+1+1 ambiguous eval for `Qwen/Qwen3-8B-Base` (`eval_ambiguous_disambiguation.py`
already takes `--model_name`). **CAVEAT:** the token-level `topk` scoring (`score_topk`, first answer
token via `tok(" "+ans).input_ids[0]`) is **invalid for Qwen3 numeric tasks** — Qwen3 tokenizes
`" 47"` as `[space, "4", "7"]`, so the first token is a bare space for every numeric answer →
gold and partner both "match" → round/truncate/first_digit/last_digit/count_* spuriously read
top1=partner@1=1.00. GPT-J merges `" 47"`→single token, so it's fine. **Use the whole-answer
`wordtopk` (beam) scoring for cross-model comparison** (`eval_wordtopk_3plus1plus1_qwen3.json`).
Result (whole-answer): Qwen3 mean top1 0.678 vs GPT-J 0.561, mean partner@1 0.170 vs 0.294 — Qwen3
disambiguates better and is less prior-bound (round 0.06→1.00, british 0.33→0.87, reverse 0.00→0.28),
but the prior asymmetry persists (last_letter 0.10/partner 0.60, largest_city 0.11/partner 0.70).
Lesson: any first-token metric is tokenizer-dependent; prefer whole-answer scoring across models.

## 2026-06-16 — BUG: `ICLDataset` int-indexing upcasts mixed-dtype rows (float input corrupts int gold)

`ICLDataset.__getitem__(int)` does `self.raw_data.iloc[i].to_dict()`, building a **1-row pandas
Series**. A Series is single-dtype, so a row with a float column (e.g. `round`/`truncate` inputs
like `3.7`) **upcasts an int output gold `4` → `4.0`**. List-indexing (`ds[[i]]`, orient='list')
preserves per-column dtype and is safe. Symptom: `round`/`truncate` scored ~0.00 because golds were
`'4.0'`; `normalize_answer` then strips the '.' → `'40'`, so the model's correct `'4'` never matched.
Only these two tasks are affected (only float-input + int-output tasks); demos were unaffected
(they use list-indexing). Fix in `recreate_fig8.py`: extract the query via `ds[[idx]]`. After fix:
round 0.00→~0.77, truncate 0.00→~1.00. **Lesson:** never grab a single ICLDataset row with int
indexing when columns have mixed dtypes; use list-indexing or coerce ints before `str()`.

## 2026-06-16 — GOTCHA: task-name collisions across dataset folders; key results by `<folder>/<task>`

4 task names exist in BOTH `dataset_files/abstractive` and `dataset_files/ambiguous`:
`count_consonants`, `count_vowels`, `identity`, `magnitude` (the ambiguous copies are the
overlap/differentiator-structured variants; `load_dataset` only scans abstractive+extractive and
would silently grab the abstractive one). Any results store keyed by task name alone
(`results/.../<task>.json`) will have the second folder's run clobber the first. Convention going
forward: **key per-folder results by `<model>/<folder>/<task>`** (recreate_fig8.py does this).

## 2026-06-16 — Recreate Fig 8 for two models: generation-until-newline, not first-token rank

For the Fig 8 recreation (Stream H), accuracy is **greedy generation stopped at "\n"** then scored
with normalized `exact_match_score` over the full answer — matching Stream G's correctness
definition, NOT the cached first-token `clean_topk`. Efficient multi-shot readout: per trial sample
10 demos + 1 query, and for k=0..10 use the first k demos (nested prefixes of the 10-shot prompt) so
shots 1..9 come from the same draw. Batched generation with token-budget batching (left-pad) keeps
GPT-J's full-MHA KV cache under 80GB. All 2200 responses/task are saved to `<task>_responses.jsonl`
for inspection. (A single-forward "read every demo answer position" trick is ~11× cheaper but only
works for first-token scoring; generation requires real decoding, so we batch instead.)

## 2026-06-15 — Per-prompt correctness for the residual-activation captures (Stream G)

To mark capture prompts correct/incorrect, **do not** reuse `fs_results_layer_sweep.json`: it scores
a different dataset partition (the 504-example `valid` split) than the captures (130 train + 40 test
sampled by `stable_rng(seed, task, split, "query_indices")`), and stores only first-token ranks.

Instead, `compute_capture_prompt_correctness.py` regenerates the *exact* capture prompts by importing
the capture's own deterministic helpers (`sample_query_indices` / `sample_demo_indices` / `make_prompt`
from `extract_targeted_residual_stream_activations.py`, seed 42) and **asserts** the regenerated
`query_source_index` order equals the capture `index.json`'s `query_indices` (byte-identical prompts).

**"Correct" = full-answer match, not first-token rank** (per request): greedy generate (`do_sample=False`,
`max_new_tokens=16`), then the repo's `parse_generation(out, [target], exact_match_score)` — its regex
`([\w. ]+)[\nQ]*` truncates at the **newline** / next `Q:`, and `exact_match_score` is SQuAD-normalized.
Join correctness to activations by `(split, query_source_index)` (robust; verified zero misses).
Correctness is one bit per prompt, shared across all 31 token positions of that prompt.

## 2026-06-15 — Cosine(activation, task-specific FV) heatmap (Stream G), companion to the MSE study

Same 31-token-position × 29-layer grid as the full-dim ridge MSE study, metric = per-example mean
**raw** cosine (no centering) between each residual activation and that task's OWN task-specific FV
(`results/gptj_fv/<task>/...`), averaged over all 29 tasks. `cosine_activation_to_task_fv.py` is
model-free; reuses the ridge loader constants + the merge script's `position_key/label` axis ordering;
diverging heatmap centered at 0, shared scale across the all/correct PNGs. Each activation directory
targets one ICL index, so `token_role` alone uniquely buckets a prompt's rows (no icl filter needed).
**Result:** all-positive, mid-layer bowl (peak L10), role banding `last_prompt_token`≈`pre_label_token`
≫ `first/last_label_token`; correct-only ≈ all at the aggregate level (max|Δ|=0.0016).

## 2026-06-14 — `ambiguous` task-disambiguation datasets + 3+1+1 eval; FINDING: prior-bias asymmetry

New dataset family `dataset_files/ambiguous/` (generator `create_ambiguous_datasets.py`) for the
task-DISAMBIGUATION study: pairs (f1,f2) that AGREE on an overlap region and DISAGREE on a
differentiator region (matched input set; overlap entries byte-identical in both files). Four pairs:
`magnitude|identity` (50/50), `past_tense|past_participle` (50/50), `first_letter|last_letter`
(50/50), `capital_city|largest_city` (50 overlap / **35** differ — only ~35 real capital≠largest
countries exist, ~20 GPT-J-plausible; recall caveat, trim/restrict downstream).

Eval `eval_ambiguous_disambiguation.py` (GPT-J, n=100/task, **cross-prompt batched** greedy gen,
token-level exact match; `matches_partner` = model emitted the OTHER function's answer). Prompt =
3 overlap demos + 1 differentiator demo (task's output) + 1 differentiator query.

**FINDING — a single disambiguating demo only redirects GPT-J in the prior-aligned direction.**
magnitude/identity are ~perfect both ways (.98/1.00). The other pairs are strongly asymmetric: the
model nails the prior-aligned task and ignores the 4th demo for the anti-prior task, emitting the
prior answer instead — past_tense .93 vs **past_participle .36** (partner .60); first_letter .97 vs
**last_letter .04** (partner .89); capital .57 / largest .47 (noisy, recall-limited). So 3 ambiguous
+ 1 disambiguating demos are enough only when the two functions are equally "natural"; a strong prior
(past-tense over participle, first- over last-letter) is not overcome by one differentiating example.
The `matches_partner` rate is the key diagnostic (separates "did the other rule" from "neither").
Next probes: a control arm (4th demo also ambiguous → prior baseline) and k>1 disambiguating demos.

## 2026-06-14 — `train_selected` FVs now also at top-20 and top-40 heads

The train-pooled (`train_selected`) FVs exist at three head counts: top-10 (original,
`results/gptj_fv_multitask_top10/`), top-20 (`..._top20/`), top-40 (`..._top40/`), all 29 tasks.
Built from the same `results/multitask_aie_heads/multitask_top_aie_heads.pt` (stores top-40) +
cached mean activations — no CIE/forward recompute. Organized views:
`results/function_vectors/gpt-j/train_selected_top{20,40}/`. FV norms grow with n (more out_proj
terms summed): top-10 ~30–47, top-20 ~34–54, top-40 ~41–66. NB the n=20/40 head sets are strict
prefixes of the top-40 ranking, so head sets nest (top-10 ⊂ top-20 ⊂ top-40). Relevant to the
n=10 degeneracy note below: train vs train+test first diverge ~n=11, so the top-20/40 train_selected
FVs are the right inputs for a non-degenerate train-vs-train+test comparison (rebuild train+test at
matching n first).

**Held-out steering finding (9 test tasks, `evaluate_heldout_multitask_head_fvs.py --n_top_heads
{10,20,40}`; combined `results/heldout_multitask_head_eval_nheads_comparison.json`):** adding heads
helps ZERO-SHOT steering but not ICL-context steering. Best-layer zero-shot+FV mean top-1 rises
0.376 (n10) → 0.381 (n20) → **0.446 (n40)**, closing most of the gap to task-specific (0.483); some
tasks jump a lot (capitalize 0.75→0.96, capitalize_first_letter 0.70→0.95). But 10-shot-shuffled+FV
is flat/down (0.796→0.785→0.780; task-specific 0.812) — with real context present the extra heads add
nothing. Caveat: at n40 several tasks' best zero-shot layer drifts to L0–1 (larger-norm FV), so verify
those before citing; the mid-layer L8–13 optimum is stable for antonym/synonym/capitalize_first_letter.

## 2026-06-11 — FINDING (Phase 2): the mean label-token function axis is CAUSAL — steering it flips synonym→antonym

`steer_label_to_query.py`, results in `results/oneshot_steering/`. Injecting `α·Δ_label` (mean
antonym−synonym label-token difference, computed on the 530 Phase-1 words) at the demo label token of a
synonym-context 1-shot prompt, scored on 1,003 disjoint queries with single-token gold ant+syn:

- **Geometric propagation:** the induced query-token shift aligns with the natural syn→ant direction —
  cos(shift, Δ_final) ≈ **0.71–0.76** (steer L6, read L8; >0.5 through L15), matched-norm **random
  control ≈ 0.01–0.06**. α-stable over 0.5–4 (linear regime), degrades at 8.
- **Behavioral flip:** flip rate 0.283 (clean) → **0.64** (L6, α=8); mean logit(ant)−logit(syn)
  −1.44 → **+1.12**, zero-crossing at α≈2. Random control stays negative (−0.92 @ α=8) — effect is
  direction-specific. Best steer layers 6≈9 > 11.
- **Reconciles Phase 1's negative map result:** the dominant shared axis propagates causally and
  predictably label→query; what the (weak, high-rank) linear map could not capture is the per-word
  high-dim tail. So "label-token arithmetic moves the next position predictably" is TRUE for the
  function axis specifically, FALSE for the full per-word geometry.
- **Geography (landmark↔park) propagates weakly** (cos 0.13–0.23, n=84, geometry-only — no behavioral
  flip exists since both functions output the same country).
- **Causal window [L4, ~L15] (full L1–L24 sweep):** steering is null at L1–2 (flip ≈ baseline despite
  maximal downstream depth; natural ‖Δ_label‖ ~0.7–0.9, the axis doesn't exist yet), onsets sharply
  L3→L4 (flip 0.34→0.57 @α4), plateaus L4–9 (peak cos 0.76, flip 0.60), decays L11–12, and is
  **exactly baseline-dead by L16** (L16/20/24: flip 0.282–0.284, cos ≤0.09). Interpretation: the
  function fingerprint is written into the label token ~L3–6 and transported to the query position
  before ~L16 — congruent with the CIE top-head band (L8–15). Profile:
  `results/oneshot_steering/fig_steer_layer_profile_L1to24.png`.
- **Engineering gotchas (reusable):** baukit `edit_output` hooks MUST be exact 2-arg
  `(output, layer_name)` closures — extra default-kwargs are mis-bound positionally by
  `invoke_with_optional_args`. The query-position shift at L_read==L_steer is identically zero
  (cross-position effects only enter via later blocks' attention) — read strictly downstream.

## 2026-06-11 — FINDING: function is decodable at the query token in FV space, but the label→query map is not low-rank/rotation-like (1-shot)

Phase-1 of the function-geometry experiment (Stream E; scripts `capture_oneshot_paired.py` +
`analyze_oneshot_geometry.py`; results in `results/oneshot_paired_analysis/`). Same-output-word,
different-function paired 1-shot prompts; source = demo label token, target = query final token.

- **Function fingerprint is present and linearly decodable at the QUERY token** along the FV difference
  axis `(fv_f1−fv_f2)`: peak **L11 AUC 0.941 / d 2.22** (antonym↔synonym), **0.894** (landmark↔park),
  mid-layer band L9–15. This is the robust, positive result.
- **The label-token difference = one dominant shared axis + a broad high-dim tail.** STABLE rank
  (Σσ²/σ₁²) of the L11 `D_label` matrix is **5.2** (antonym↔syn, 544×4096) / **3.3** (geography, 84×4096):
  σ₁ alone holds ~19%/~30% of the energy → a single clear "function axis." BUT 90% of the energy needs
  k≈315 dims (antonym↔syn) — the residual is high-dimensional/per-word. Stable rank is **lowest at mid
  layers (min ~4.7 @ L9, ~5 through L15)**, coinciding with peak query-token FV separation; rises to ~9
  by L24. (The entropy "effective rank" was ~150; the two disagree precisely because of the fat tail —
  stable rank keys on the dominant axis.) Centered stable rank (mean axis removed) = 18.8 / 9.4.
  **Robust to magnitude:** unit-normalizing each difference vector before stacking leaves stable rank
  essentially unchanged (5.4 / 3.5 vs raw 5.2 / 3.3) — so the dominant axis is DIRECTIONAL, not driven
  by a few high-norm vectors. Magnitudes are themselves tight (CV ~23%/18%) BUT 14/544 antonym↔syn rows
  are exactly zero (degenerate: demo input collided across functions, e.g. borrow←lend in both → identical
  prompts; excluded when normalizing → 530 rows). Artifacts: `D_label.pt`, `fig_Dlabel_svd{,_unit}_L11.png`,
  `fig_Dlabel_magnitude_hist_L11.png`, `fig_Dlabel_stable_rank_by_layer{,_unit}.png`.
- **But the demo-label → query-token linear map is weak / high-rank / not rotation-like** (held-out
  R² ≤0.16 mid-layers, eff rank ~190–280, Procrustes gap large). So the original "low-rank manifold +
  *predictable rotation* from label to next position" hypothesis is **not supported in 1-shot** — even
  though a single dominant function axis DOES exist at the label token.
- **Method notes (reusable):** (1) hold the OUTPUT token fixed and vary function → the source-token
  activation difference is pure contextualization (the label token is literally identical across f1/f2;
  assertion enforced). (2) For a W-sample map into d=4096 with W≪d, the full-matrix M structural metrics
  (eff-rank, reduced-rank, ‖MᵀM−I‖) are rank-limited/regularization-dominated — trust **held-out map_R2**
  and the **FV-separation AUC**, not in-sample reduced-rank R². (3) Project the source→target map into the
  ≤2W-dim data span before any d×d op — all reported metrics are invariant, ~30× faster.
- **Decision for Phase 2 (if pursued):** steer along the FV difference axis at/near the query token
  (well-motivated by the separation); do NOT assume steering the demo label token propagates predictably
  to the query (Phase 1 contradicts it). A multi-shot variant (stronger function identification) is the
  natural next probe before drawing strong conclusions.

## 2026-06-12 — Criterion for "confusable task pairs": matched input AND output marginals

For the mixed-task ICL / function-geometry substrate, a good pair (f1, f2) must encode
the function ONLY in the (input, output) relation: neither the input token alone nor the
output/label token alone may be predictive of which task it is. Formally
P(input|f1)=P(input|f2) AND P(output|f1)=P(output|f2). Shared input domain + same output
*type* is NOT enough — the output *marginal* must match too.

This rejects pairs that looked superficially fine:
- capitalize_first ↔ last: **output** leak (first- vs last-letter distributions differ).
- english-fr ↔ -de, country-capital ↔ -currency, person-occupation ↔ -sport: **output**
  leak (you can ID the task from one output token: a language / a city vs currency / etc.).
- hypernym ↔ hyponym: **input** leak (specific vs general words) — inverse pairs only
  satisfy the criterion over a *symmetric* domain (input set = output set).

Two construction recipes that satisfy it by design: (1) inverse functions over a
symmetric/closed domain (next/prev over a cycle; +k/−k; Caesar ±n); (2) two distinct
relations whose outputs land in the same word pool as the inputs (antonym/synonym;
synonym/rhyme). Operational test: a probe predicting the task from input-token-only or
output-token-only embeddings should be at chance.

**A second, sharper constraint — in-context learnability — makes the design space much
smaller than it looks.** Matched output marginals already require the two functions to
share an output *pool* (a lone output token can't reveal the task). That alone kills
factual-attribute pairs (capital vs largest-city → disjoint answer sets), translation
(disjoint by language), and morphology (suffix leak). On top of that, the function must
be inferable from ~5 ICL demos — which kills otherwise-legal ideas like alphabetical
neighbor word (model can't know the candidate lexicon). What survives BOTH = "a general
rule the model already knows, over a domain with a shared output pool" → arithmetic on
numbers, or meaning/sound relations over words. That is *exactly* the antonym/synonym/
rhyme/number region — the apparent "everything looks similar" is the constraint, not a
lack of imagination. Decision: ship **3 pairs** (antonym|synonym, synonym|rhyme,
antonym|rhyme, next_number|prev_number); do NOT force a 4th. Letters were generated then
dropped (26-cap + no new legal+learnable axis). If more variety is ever needed, the only
honest routes are to *relax* a constraint deliberately and *measure* the cost (e.g.
morphology with the input/output leakage probe), or curate an association corpus offline.

Datasets live in `dataset_files/paired_tasks/` (generator
`dataset_files/generate/create_paired_tasks_datasets.py`): next/prev_number (words),
rhyme (CMUdict, outputs in syn/ant vocab), plus copied synonym/antonym. Note:
number-words avoid digit clumping but compound forms >20 are multi-token (word-based) —
score at first/last label-token positions.

## 2026-06-12 — Geography under the matched-marginal rule: only spatial place→place survives

Geography mostly LEAKS: (a) attribute relations (country→capital vs →currency vs
→continent) have **disjoint output pools** → output-token leak; (b) entity-type→country
relations leak via the **input** — verified the repo's park-country has "Park"/"National"
in 747/749 inputs, so landmark|park is identifiable from the input alone (why Stream E
could only use it geometry-only). The single legal family is **place→place spatial
relations over one shared pool** — the geographic analog of next/prev_number. Built
`east_neighbor` / `west_neighbor` (`dataset_files/generate/create_geography_neighbor_datasets.py`):
country → nearest country in the E/W bearing quadrant, from most-populous-city coords
(geonamescache, offline; deterministic). Both country→country → matched input AND output
marginals; identical input set (244 countries with both an E and W neighbour). Multi-token
country names allowed (first/last label-token, like compound numbers). Approximate for
island/edge countries (e.g. Bermuda→Cabo Verde) — inherent to nearest-in-quadrant. North/south
and bordering (needs adjacency data, not available offline) are possible extensions.
Installed `pycountry-convert` (continents) + `geonamescache` (city coords), and `nltk`+WordNet
(unused so far — semantic co-hyponym/hypernym/meronym pairs remain an option).

**General principle (what makes geography/periodic-table work):** take a domain the model
knows that has internal STRUCTURE (1D order, 2D grid, hierarchy) and pair two "neighbour"-type
relations over it; the shared pool + the structure give matched marginals for free. Instances:
numbers (1D line: next/prev), geography (2D map: east/west), **periodic table (2D grid:
next_in_period=right / next_in_group=below)**. Built `next_in_period`/`next_in_group` via
`mendeleev` (offline): 66 main-group+transition elements with both a right and below neighbour,
identical input set, multi-token names allowed. f-block excluded (no group_id; model knows them
worst). Skipped US-presidents (ordinal over a known list, ~46 — too small). Open structured
domains not yet built: north/south geo neighbours, WordNet taxonomy (co-hyponym; is-a vs has-part).

## 2026-06-10 — Mixed-task ICL probe: synonym pays for diluted demos, not antonym

Probe (`src/eval_scripts/mixed_icl_antonym_synonym_topk.py`): n=200 prompts, 10 ICL
demos = 5 antonym + 5 synonym, query from one task; top-1/2/3 = correct first token
in model's top-k. Ran both demo orders + a same-task 10-shot reference overlay.
(Query word is excluded from the ICL demos; n=50 SE≈0.06 was too noisy → use n≥200.)

- **Antonym query ≫ synonym query** (top-1 0.47–0.55 vs 0.12–0.20), every cut. This is
  task difficulty — synonyms have many valid answers so the single gold first-token
  frequently misses top-3. First-token top-k *understates* synonym competence; use
  any-acceptable-synonym scoring if synonym is the focus.
- **Recency is real (~+0.08 top-1):** each task scores higher when *its* demo block is
  the last one before the query (antonym 0.47→0.55, synonym 0.12→0.20). An earlier
  n=50 run wrongly called recency "weak/inconsistent" — that was sampling noise.
- **Antonym is fairly robust to halved demos** (mixed 0.47–0.55 vs same-task ref 0.565,
  matching the ref when its demos are recent); **synonym degrades hard** (mixed
  0.12–0.20 vs ref 0.365) → synonym pays most for diluted demos.

Repro convention for model-loading scripts on this box: deps not preinstalled —
`pip install transformers==4.49.0 accelerate` (per fv_environment.yml); baukit is
not on PyPI, so scripts that only need `eval_utils`' pure helpers should inline them
rather than import the baukit-laden module. GPT-J weights cached under
`/workspace/.cache/huggingface`; run with `HF_HOME=… HF_HUB_OFFLINE=1`.

---

## Conventions (observed in the codebase)

- **Model:** GPT-J (29 layers, d=4096). FP16 + Qwen variants exist (see git log).
- **Task split:** `task_splits/abstractive_train_test_tasks_29.json` = 20 train + 9 test.
- **Results layout:** one dir per experiment under `results/`; per-task subdirs hold
  `.pt` artifacts + a `*_metadata.json` / `*_summary.json` describing the run config.
- **Head set artifact:** `multitask_top_aie_heads.pt` (+ `_metadata.json`) — ranked
  (layer, head) list by prompt-weighted CIE. Stored top-40; top-10 typically used.
- **FV artifact:** `{task}_function_vector*.pt` = sum over selected heads of
  out_proj applied to the head's mean last-token activation.
- **`_multitask_top10` suffix** on a result/FV dir means it was built from the
  multitask top-10 head set rather than per-task FVs.
- **Save intermediates (within storage reason).** When a big experiment has an expensive stage
  (forward passes, CIE, activation capture), persist the *general* intermediate — not just the
  final answer — so variations rerun in minutes instead of hours. Concretely: store the full
  per-(layer,head) CIE grid + a generously-sized ranking (top-40, even if top-10 is used) rather
  than only the chosen heads; store mean activations for **all** heads/positions needed by any
  plausible N/k, not just the selected ones; keep per-prompt effects when subset re-aggregation
  might matter; cache residual-stream activations once and reuse across regression targets.
  Payoff observed 2026-06-11: top-20/30/40 varicl FVs + a whole new PCA-ridge heatmap cost
  minutes because stage-1 saved all-head activations and the top-40 ranking. Guardrail: this is
  for O(GB) per-task tensors and grids, not for dumping every forward pass — if an intermediate
  would cost ≫ recomputing it on demand, skip it and note the recompute command in the metadata
  instead.

---

## Decided

### 2026-06-12 — FVs round 2: east/west_neighbor + next_in_period/group — built, but LOW-N

Same 3-method derivation as the trio below (manifest now `task_splits/paired_tasks_7.json`;
builder manifests `fv_manifest_paired2.json`; auto no-filter retry available but unused — all 4
tasks pass the filter). All 12 FVs in the organized folder, norms 24–44. **CAVEAT: GPT-J is weak
at all four** — ICL-correct counts: canonical 10-shot east 5/51, west 7/51, period 2/14, group
5/14; varicl east 2, west 3, period 1, group 1 → the correct-only mean activations (esp. varicl,
1–3 prompts) are HIGH-VARIANCE. Filter kept ON for consistency with the 29 originals. If these
tasks matter downstream, prefer no-filter variants or bigger datasets. (Effective-n hierarchy among
paired tasks: next/prev_number excellent (42/42) ≫ elements/geo (1–7) ≥ rhyme (0 → no-filter).)

### 2026-06-12 — FVs derived for the 3 paired tasks (rhyme, next_number, prev_number) × 3 methods

The new `dataset_files/paired_tasks/` trio now has FVs under all of `task_specific`,
`train_selected`, `train_varicl` (organized folder + underlying caches; `fv_manifest_paired.json`
in each builder root so the original 29-task manifests are untouched). Conventions adopted:
- The 3 task JSONs are **symlinked into `dataset_files/abstractive/`** (user-approved; the loader
  hardcodes abstractive/extractive). Split seed 42 → 140/18/42; valid is only 18.
- `task_splits/paired_tasks_3.json` = manifest for stage-2 builders (they validate `--tasks`).
- **rhyme FVs are NO-FILTER** (`--no_filter_to_correct_icl`, both regimes): GPT-J is 0/18
  ICL-correct on rhyme valid, so the standard correct-only averaging is impossible. Treat rhyme's
  FV as "attempted-task" activations; flag in any cross-task comparison. next_number (18/18) and
  prev_number (15/18) are standard.
- Repro trick: running varicl stage-1 for extra tasks with `--num_shards N>1` keeps
  `writes_global=False` → per-task files only, existing pooled head artifact safe. We still used an
  isolated root (`results/multitask_aie_heads_varicl_paired/`, kept for provenance) and copied the
  `*_mean_head_activations_varicl.pt` into `results/multitask_aie_heads_varicl/<task>/`.

### 2026-06-11 — PCA-space (direct) activation→FV ridge sweep + 16-PC vs full-dim comparison

Companion to the full-dim ridge decoder (Stream C below), run in a 16-PC bottleneck across all 31
token positions × 29 layers. Per cell: activation PCA (k_act=16) fit per-cell on 20 train; FV PCA
(k_fv=16) fit once on 20 train FVs; **ridge 16→16, λ by leave-one-train-task-out CV, single
20-train standardizer**; predict 7 test tasks, reconstruct to 4096-d, score there. Direct
projection (not the deprecated joint). FV target `train_selected`. cc/pc excluded.

- **Scripts:** NEW `regress_activation_to_fv_pca_ridge.py`; launcher `run_fulldim_ridge_shards.sh`
  generalized with a `WORKER` env override (drives both sweeps); merge reused unchanged.
- **Output:** `results/pca_ridge_activation_to_fv/` (combined_metrics.csv = 899 rows + heatmaps).
- **HEADLINE:** the 16-PC bottleneck is **free at the optimum** — PCA best `icl10/finaltok @ L13 =
  0.1147` vs full-dim `@ L11 = 0.1161` (PCA marginally *better*; it denoises). In the mid-layer
  sweet spot (L8–13) PCA ≈/< full-dim; in later/embedding layers PCA is worse (L28 0.158 vs 0.149).
  Net mean Δ(pca−full)=+0.003. → 16 activation PCs hold all recoverable activation→FV signal where
  it's concentrated; the regression target genuinely lives in a ~16-dim subspace.
- **Metric identity:** `fv_test_mse = (k_fv/4096)·pca_test_mse + floor` (FV-PCs orthonormal),
  verified to 4e-8. FV-PC reconstruction floor (test) ≈ 0.099. So selecting α on PCA-space CV MSE ==
  selecting on reconstructed MSE (differ by the constant floor).
- **Comparable metric note:** this reconstructed-4096-d MSE is the same unit as the full-dim ridge
  (0.116) and the k/layer sweeps — NOT the joint-PCA-space MSE (Open Q3).

### 2026-06-10 — `train_varicl` RAN to completion; GOTCHA: stage-2 needs test-task activations precomputed

The variable-ICL method is fully built for GPT-J: 29 FVs + heads.pt + manifest + per-task
`selected_heads.json` under `results/function_vectors/gpt-j/train_varicl/`. Pooled top-10 (varicl)
= [(9,14),(15,5),(8,1),(12,10),(11,0),(8,0),(14,0),(24,6),(21,2),(10,0)]; **8/10 overlap with
`train_selected`** (top-2 identical), so the variable-ICL regime selects nearly the same head
subspace as fixed-ICL train pooling — overlap, not identity, as expected.

**GOTCHA (pipeline gap, now handled):** the documented step sequence (stage-1 `--task_split_key
train_tasks` → stage-2 `compute_all_task_fvs_varicl.py`) is INCOMPLETE. Stage-2 builds FVs for all
29 tasks and **requires `<task>_mean_head_activations_varicl.pt` to already exist for every task**
(it raises `FileNotFoundError`, does NOT compute on the fly). Stage-1 over `train_tasks` only writes
the 20 train tasks' activations, so the build crashes on the first test task. **Fix:** run stage-1
once more on the test split to an isolated dir (so `writes_global` can't clobber the train-pooled
head artifact), then copy the 9 activation files in:
```
python src/eval_scripts/compute_multitask_varicl_heads.py --task_split_key test_tasks \
  --abstractive_only --query_split valid --demo_split train --n_top_heads 40 --batch_size 8 \
  --min_shots 1 --max_shots 10 --max_successful_prompts 170 --filter_to_correct_icl \
  --save_per_prompt_effects --save_path_root results/_varicl_testtasks --num_shards 1
# then: cp results/_varicl_testtasks/<task>/<task>_mean_head_activations_varicl.pt
#            results/multitask_aie_heads_varicl/<task>/   (for the 9 test tasks)
python src/eval_scripts/compute_all_task_fvs_varicl.py --overwrite
```
This recomputes CIE for the 9 test tasks too (unavoidable; they're short-sequence so it's cheap and
the resulting test-pooled head artifact is harmless/unused). Future fix option: teach stage-2 to
compute missing activations itself, or have the runner also process `test_tasks` for activations only.

### 2026-06-10 — 4th FV method: `train_varicl` (variable-ICL, train-pooled) — IMPLEMENTED, not yet run

A new head-selection + FV method alongside task_specific / train / train+test. Same two-stage
shape, but the prompt regime changes:
- **Variable ICL:** each prompt draws a random 1–10 demonstration count (deterministic per
  (task_index, query_idx); shard-invariant via global task_index).
- **Correctness filter, capped:** keep only correctly-answered prompts, **≤170 per task**.
- **Read position:** mean head activations AND the CIE intervention both read the **query
  predictive (last) token, T=-1** (user-confirmed). Single, length-independent position → mean
  activations stored as `(n_layers, n_heads, head_dim)` (no token axis), so a new save filename
  `<task>_mean_head_activations_varicl.pt` and adapted indexing are required (the canonical
  builder's `[...,-1]` would grab a scalar — see the three corrections below).
- **CIE:** variable ICL + shuffled labels, intervention at the query token, on the same
  correctly-answered query set; CIE prompts use seed `args.seed + cie_seed_offset` (default
  500000) so they differ from the activation pass but stay reproducible.
- **Pooling:** CIE averaged across the **20 train tasks** (same as `train_selected`), top-N heads.

**New sibling scripts (existing engines untouched):** `src/utils/varicl_utils.py`,
`src/eval_scripts/compute_multitask_varicl_heads.py`,
`src/eval_scripts/compute_all_task_fvs_varicl.py`,
`src/eval_scripts/run_multitask_varicl_all_tasks.sh`. New args: `--min_shots`, `--max_shots`,
`--max_successful_prompts`, `--cie_seed_offset`.

**Three corrections vs the canonical code (handled):** (1) the fixed-n_shots correctness filter
can't vary shots → new `varicl_correctness_filter`; (2) CIE `avg_activations[L,H,token_idx]` →
`avg_activations[L,H]` for single-position; (3) FV builder `mean_activations[L,H,-1]` →
`mean_activations[L,H]`.

**Outputs (when run):** `results/multitask_aie_heads_varicl/` (head set + per-task CIE/activations);
`results/function_vectors/gpt-j/train_varicl/` (FVs + heads.pt + selected_heads.json). Run commands
+ smoke verification: WORKLOG Stream D and plan `/root/.claude/plans/immutable-finding-boole.md`.

### 2026-06-10 — Direct full-dim (4096→4096) activation→FV ridge decoder (Stream C)

New experiment line, **deliberately PCA-free** (distinct from `regress_activation_to_fv_joint_pca*`).
For each (token position, layer) cell, fit one ridge map from the raw 4096-d residual activation
to the raw 4096-d `train_selected` FV. λ chosen by leave-one-train-task-out CV (20 folds); a single
standardizer fit on the pooled 20-train rows is reused everywhere; MSE reported natively in 4096-d
on the 7 test tasks (cc/pc excluded). 31 token positions × 29 layers = 899 cells.

- **Scripts:** `regress_activation_to_fv_fulldim_ridge.py` (worker, one shard = one ICL idx; GPU
  ridge via eigendecomposition reuse so the α grid is ~free), `run_fulldim_ridge_shards.sh` (tmux
  sharding, round-robin), `merge_fulldim_ridge_results.py` (combined CSV + heatmaps + summary).
- **Outputs:** `results/fulldim_ridge_activation_to_fv/`.
- **GOTCHA (now handled):** in the `4tokens` dir the final prompt token (`last_prompt_token`) has
  `icl_example_index = None`; the 3 label roles use `10`. Filter the final-token role on `None`.
- **Result:** best decode = final prompt token @ **layer 11**, test_mse **0.116**; clean layer bowl
  min at L11 (band L10–14); query position (ICL 10) beats earlier ICL demos. Embedding-layer
  pre-label/final tokens are constant across tasks (→ predict-the-mean baseline 0.217).
- **Metric note:** this 4096-d MSE is NOT comparable to the joint-PCA-space MSE (Open Q3); it *is*
  comparable to the reconstructed-4096-d MSE used by the k/layer sweeps.

### 2026-06-10 — Three head-selection methods + organized FV folder

FV derivation is two stages: (1) select top-N heads by CIE; (2) per task, sum
`out_proj(mean_head_activation[L,H,-1])` over the selected heads. The three methods
differ **only in stage 1** — which tasks' per-task CIE is pooled to rank heads.
Stage 2 (mean head activations) is shared. All three use top-10, GPT-J-6B,
`query_split=valid`, `demo_split=train` → directly comparable.

| Method | Stage-1 pool | Head set | Per-task FVs |
|---|---|---|---|
| task-specific | 1 task | `gptj_fv/<task>/<task>_indirect_effect.pt` | `gptj_fv/<task>/<task>_function_vector.pt` |
| train | 20 train tasks | `multitask_aie_heads/multitask_top_aie_heads.pt` | `gptj_fv_multitask_top10/` (29) |
| train+test | 29 tasks | `multitask_aie_heads_all_tasks/multitask_top_aie_heads.pt` | **BUILT** → `function_vectors/gpt-j/train_test_selected/` (29 + manifest) |

**Organized access point:** `results/function_vectors/{task_specific,train_selected,train_test_selected}/`
— a clean view with uniform `<task>_function_vector.pt` naming so any `--fv_root`
points straight at a method. See `results/function_vectors/README.md`.

**Symlinks, not hard moves.** `results/gptj_fv` is dual-role (task-specific FVs +
the shared `*_mean_head_activations.pt` cache + CIE + eval JSONs) and ~15 scripts
default to the old paths. Hard-moving would break the pipeline, so the FV folder
symlinks into the existing caches. Underlying dirs remain source of truth.
Reversible; revisit if we want a true single-source-of-truth migration (would
require rewiring all `--fv_root` defaults).

### 2026-06-10 — k-sweep outputs nested under `results/k_sweeps/`

All activation→FV k-sweep result dirs now live under `results/k_sweeps/` (was flat in
`results/`). Names trimmed of the redundant `k_sweep_` prefix (the parent conveys it):
- `results/k_sweeps/activation_to_fv_ols_multitask_top10_log2/` — original, 9 test tasks.
- `results/k_sweeps/activation_to_fv_ols_multitask_top10_log2_exclude_cc_pc/` — 7 test tasks
  (drops `country-currency`, `product-company`).
Default `--output_dir` in `sweep_k_activation_to_fv_ols_log2.py` and
`sweep_k_activation_to_fv_ols.py` updated to match; `run_config.json` self-paths fixed.

### 2026-06-10 — Direct k-sweep, third axis: fix k_act, sweep k_FV → joint optimum (16,16)

`sweep_k_activation_to_fv_direct_log2.py` now has `--fix_act_k K` (mutually exclusive with
`--fix_fv_k`): pins k_activations=K and sweeps k_FV (doubling grid reinterpreted as k_FV, capped
at fv_k_cap). Run at k_act=16 (7 tasks): **test MSE falls monotonically with k_FV, best at the
k_FV=16 cap (14/15 series), riding just above the recon floor (gap ~0.01–0.03).** No k_FV
overfitting — k_FV is bounded by discarded-FV-variance (floor) and by the FV-PCA rank cap (16),
not by overfitting. With 16 activation PCs the regression recovers any k_FV target nearly to the
floor. **Combined over all three direct cuts (coupled diagonal / fix k_FV / fix k_act): the joint
optimum is (k_act≈16, k_FV=16) — the corner.** k_act peaks ~16 then overfits; k_FV climbs
monotonically to its 16 cap. The four direct dirs in `results/k_sweeps/`:
`..._log2`, `..._log2_exclude_cc_pc`, `..._log2_fixedfvk16_exclude_cc_pc`,
`..._log2_fixedactk16_exclude_cc_pc`.

### 2026-06-10 — All k_sweeps are DIRECT; joint results removed

Per user decision, the k-sweep result dirs are now **direct-method only**. The three joint runs
were deleted and replaced by direct equivalents; `results/k_sweeps/` holds exactly:
- `activation_to_fv_direct_ols_multitask_top10_log2/` — coupled diagonal, full 9 test tasks.
- `activation_to_fv_direct_ols_multitask_top10_log2_exclude_cc_pc/` — coupled diagonal, 7 tasks.
- `activation_to_fv_direct_ols_multitask_top10_log2_fixedfvk16_exclude_cc_pc/` — k_FV pinned 16,
  sweep k_act, 7 tasks.
The joint *script* (`sweep_k_activation_to_fv_ols_log2.py`) is retained as a tool; only its
results were removed. **Settling finding (direct, k_FV=16):** k_act=1 → test MSE ≈ 0.18–0.19
(≈ predict-the-mean ~0.21), k_act=16 → ≈ 0.126. A single activation-PC does NOT recover the
16-dim FV — the joint method's apparent "1 PC suffices" was the 16 appended FV-basis features,
not genuine low-rank recoverability. Optimal remains k_act≈16 (first-label 32), k_FV=16, ICL5.

### 2026-06-10 — Regression definition: "direct" k_activations→k_FV (vs "joint")

Two regression definitions now coexist for the activation→FV PCA decode:
- **joint** (`sweep_k_activation_to_fv_ols_log2.py` + the `regress_*_joint_pca*` family): X and Y
  both projected onto the SAME concatenated [act-PCA (k_act) | FV-PCA (k_fv)] basis; OLS maps the
  (k_act+k_fv)-dim joint projection of the activation → the joint projection of the FV. The FV-PC
  half of the prediction is reconstructed for the 4096-d MSE.
- **direct** (NEW `sweep_k_activation_to_fv_direct_log2.py`): X = activation→act-PCA (k_act) only;
  Y = FV→FV-PCA (k_FV); OLS regresses R^{k_act}→R^{k_FV}; predicted FV-PCs reconstruct to 4096-d.

**Why they differ:** joint's feature vector additionally includes the activation projected onto
the FV basis (k_fv extra inputs), so joint fits slightly better. Measured: direct is
~0.002–0.003 higher test MSE than joint at the optimum (icl5/first 0.1261 vs 0.1232). The
**direct** setup is the cleaner "regress activation space → FV space" and should be preferred as
the headline definition; joint numbers were mildly optimistic. Optimal k (≈16, first-label 32),
ICL monotonicity, and token-role ordering are all unchanged between the two. The new script
mirrors all log2 flags incl. `--fix_fv_k`.

### 2026-06-10 — k is two knobs: k_activations vs k_FV (`--fix_fv_k`)

The joint-PCA regression has two independent dimensionalities: **k_FV** (FV-target PCs) and
**k_activations** (input PCs). The original log2 sweep coupled them below the cap
(`fv_k = min(k, 16)`), conflating the two. New flag `--fix_fv_k` on
`sweep_k_activation_to_fv_ols_log2.py` holds k_FV at `fv_k_cap` for every k, isolating
k_activations. **Finding (k_FV=16):** first/last-label tokens optimal at k_activations≈16–32;
pre-label token optimal at k_activations≈1–8 (more PCs hurt — little task signal there).
Beyond ~16–32 the activation side overfits regardless of token. Pinning k_FV=16 only changes
the k<16 regime vs the coupled run (it's strictly better there since the FV target isn't shrunk).
Canonical config: layer 11, last/first-label, ICL 5, **k_activations≈16, k_FV=16**.

### 2026-06-10 — Excluding cc/pc barely changes the regression k-sweep (FINDING)

Re-ran the log2 sweep without `country-currency` + `product-company` (whose
train-multitask-selected FVs steer much worse than task-specific). The activation→FV
regression test-MSE moved by only ~±0.001 and the optimal-k structure (k≈16–32, fv_k 16,
overfit past 32) was unchanged. **Implication:** "bad for steering" ≠ "hard to regress from
activations" — the two task-quality notions are decoupled, so the cc/pc weakness is not
visible in (and not fixable via) this reconstruction-MSE metric. If their weakness matters,
measure it on the steering/FV-effectiveness side (cf. `evaluate_heldout_multitask_head_fvs.py`).

### 2026-06-10 — Split provenance of all-tasks head set: RESOLVED (was Open Q1)

The all-tasks head set was computed on `query_split=valid` (not `train`). Evidence:
per-task files are suffixed `_valid` (`*_mean_indirect_effect_over_valid.pt`,
`*_per_prompt_indirect_effect_valid.pt`) — the suffix is emitted from the actual
split — and the runner passes `--query_split valid`. The metadata top-level
`query_split=train` is a **stale default**, almost certainly written by the
`--reduce` step (which didn't receive the arg). Both head sets therefore share
`query=valid, demo=train` and are comparable. **Follow-up (low priority):** fix the
`--reduce` path in `compute_multitask_top_aie_heads.py` to record the true split.

### DONE — train+test function vectors built (GPT-J)

Built 2026-06-10 with (reuses cached activations; zero forward passes):

```
python src/eval_scripts/compute_all_task_fvs_from_multitask_heads.py \
  --heads_path results/multitask_aie_heads_all_tasks/multitask_top_aie_heads.pt \
  --n_top_heads 10 --fv_root results/gptj_fv \
  --output_root results/function_vectors/gpt-j/train_test_selected \
  --task_manifest task_splits/abstractive_train_test_tasks_29.json
```

→ `results/function_vectors/gpt-j/train_test_selected/<task>/<task>_function_vector.pt`
(29) + `fv_manifest.json`. All three methods now complete for GPT-J. The builder also
supports `--tasks` (subset shard) and `--manifest_name` for **future models with no
cached activations** (each shard runs prompts at batch size).

### 2026-06-10 — Per-task head metadata (`selected_heads.json`)

Every task folder under `results/function_vectors/<model>/<method>/<task>/` carries a
`selected_heads.json` listing the `[layer, head, mean_indirect_effect]` heads that built
its FV, plus `selection_pool`. Generated (idempotent) by:

```
python src/eval_scripts/write_fv_head_metadata.py --model_root results/function_vectors/gpt-j --n_top_heads 10
```

`task_specific` reads each task's own `top_heads` (unique per task); the two multitask
methods read the shared `heads.pt` (top-N). Rerun after building FVs for a new model.

### 2026-06-10 — IMPORTANT: train vs train+test is degenerate at n=10

At top-10, `train_selected` and `train_test_selected` select the **same head set**
(same membership; rank order and CIE scores differ slightly). Since the FV is an
order-independent sum over selected heads, the `function_vector` tensors are **exactly
equal for all 29 tasks** (verified: global max|Δ|=0; norms match). NB the `.pt` *files*
are not byte-identical — `cmp` differs on dict metadata (`top_heads` order, paths) — only
the tensors match. They diverge only at larger
n (34/40 overlap at n=40; first differences ~n=11+). **Implication for experiments:** a
train-vs-train+test comparison at n=10 measures nothing — raise `n_top_heads` (rebuild
with a larger `--n_top_heads`) or use a per-task metric to study the selection-leakage
effect. `task_specific` is genuinely distinct from both at every n.

---

## Open questions (coordinator-flagged 2026-06-10)

1. **Canonical head set for held-out eval.** `evaluate_heldout_multitask_head_fvs.py`
   used the train-only (20-task) head set. Should held-out eval use the all-tasks
   set instead, or both as a comparison?

2. **One FV builder, not three.** `compute_task_fv_from_multitask_heads.py`,
   `compute_all_task_fvs_from_multitask_heads.py`, and `compute_fv_from_selected_heads.py`
   overlap. Leaning: `compute_all_task_fvs_from_multitask_heads.py` is the canonical
   batch builder (one model load, all 29 tasks, standard `<task>_function_vector.pt`
   filenames + `fv_manifest.json`) — it's the one in the build command above.
   `compute_fv_from_selected_heads.py` is the flexible single/ad-hoc builder (arbitrary
   head source, `--tasks`, non-standard `_fv_<tag>.pt` filename). Mark
   `compute_task_fv_from_multitask_heads.py` (single-task) legacy.

3. **Canonical regression metric.** Joint-PCA-space MSE vs reconstructed 4096-d
   FV-space MSE are not comparable. Pick one as the headline metric; report the other
   as secondary only.

4. **Canonical FV target for regression.** Some regression scripts target `gptj_fv`,
   others `gptj_fv_multitask_top10`. Decide which is the primary target (now also
   reachable as `function_vectors/task_specific` vs `function_vectors/train_selected`).

5. **Standardize ICL index range** across regression/sweep scripts (currently mixes
   1–4 and 2–5).

6. **Save per-prompt CIE effects on the baseline too?** The train-only head set lacks
   per-prompt effects, so `select_heads_from_cie_subset.py` cannot re-subset it. If
   subset analysis matters for the baseline, re-run with `--save_per_prompt_effects`.

7. **Smoke-test dirs.** `results/joint_pca_activation_to_fv_regression_smoke` and
   `results/pca_abstractive_fv_activation_scatter_smoke` — keep or remove?

---

## Process notes

- Workers: register a stream in WORKLOG.md before editing; don't co-edit the same
  source file without coordinating (per CLAUDE.md). `compute_multitask_top_aie_heads.py`
  and `evaluate_heldout_multitask_head_fvs.py` are the currently-modified shared files.
- Nothing is committed yet; all experimental scripts + results are uncommitted.
