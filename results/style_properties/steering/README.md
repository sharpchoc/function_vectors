# results/style_properties/steering — STEERING SANDBOX

> **Nothing in this folder is canonical.** There is no default, headline, or recommended
> steering result. Every subfolder is one exploratory variant. Promotion of a variant to
> repo standard requires an **explicit user decision**, recorded in DECISIONS.md.
> (User instruction 2026-09-06.)

The variant space is a grid over three axes:

| axis | values |
|---|---|
| vector construction | `meandiff` (alt mean − nat mean) · `meanact` (raw alt mean, no subtraction) · `sparsehead` (sum of sparse-selected heads' contributions) |
| k filter | `kall` (all cue sites) · `k4` (only sites with k ≥ 4 prior manifestations) |
| success filter | `succno` (no behavioural filter) · `succyes` (only sites where the model actually followed that context's convention) |

## Coverage (4 of 12 cells have data)

| technique | `kall` / `succno` | `kall` / `succyes` | `k4` / `succno` | `k4` / `succyes` |
|---|---|---|---|---|
| `meandiff` | [data](variants/meandiff__kall__succno/spec.md) | not run | not run | [data](variants/meandiff__k4__succyes/spec.md) |
| `meanact` | [data](variants/meanact__kall__succno/spec.md) | not run | not run | not run |
| `sparsehead` | [data](variants/sparsehead__kall__succno/spec.md) | not run | not run | not run |

Empty cells and what each would require: [`variants/NOT_RUN.md`](variants/NOT_RUN.md).

## Shared protocol (identical across variants, so cells are comparable)

first cue token per document · 200 docs · 32-token T=1 seeded rollouts · LLM coherence judge · strict metric (unscorable = not adopted).

- **Injection site**: the cue token only — the first cue of each document, so the context
  contains no prior manifestation of either convention to override.
- **Primary metric**: strict rate. An *unscorable* rollout (the model never produced the
  feature in 32 tokens) counts as **not adopting** — reported per cell, since it is often
  the deciding statistic.
- **Coherence**: an LLM judge labels each rollout fluent/gibberish, ignoring the
  manipulated convention; gibberish is dropped and the rate reported.
- **Controls** (user decision: every variant should eventually carry both): counterfactual
  = another property's vector of the same construction; reverse = negated vector on
  alt-convention documents.
- **Shared sweep grid**: layers [2, 4, 6, 8, 10, 12, 16, 20, 24], doses ['0.5', '1', '2', '4', '8', '16', '32'].
  Cells that searched less say so in their spec and in the comparison figure's flags —
  their peaks are **not** directly comparable.

## Files

| path | contents |
|---|---|
| `comparison_table.png` / `.csv` | all populated cells side by side, alphabetical and unranked, with caveat flags |
| `variants/<cell>/spec.md` | exact vector formula, filters, site, grids searched, arms present/absent, provenance, caveats |
| `variants/<cell>/results.csv` | per-property strict rate, conditional adherence, unscorable %, incoherent %, n, controls |
| `variants/<cell>/adherence.png` | that cell alone: unsteered / steered / reading-genuine-context reference |
| `variants/NOT_RUN.md` | the 8 empty cells |
| `archive/README.md` | earlier runs under a *different* protocol, excluded from the grid |

## Code

`src/sandbox/ext_styleprops/`: `variants.py` (the registry — single source of truth for the
grid), `variant_metrics.py` (the one metric definition), `build_variant_results.py`
(assigns run arms into cells; `--check` verifies extraction is lossless),
`compare_variants.py`, `steer_adherence.py` (runner), `judge_coherence.py`.

Adding a technique = adding one `Variant` entry, not a new script.
