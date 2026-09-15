# results/style_translation/qwen25_base/code/read_features — read feature (evidence-token mean) for the 55 coding-convention families

Qwen2.5-7B (base). Step 6 of the style-translation study applied to code (`../../code_pool_full.json`), protocol as for the text families
(`../../read_features/`) with the code-specific evidence rule decided 2026-09-15 (DECISIONS.md):

- **Evidence tokens** = in each of the 4 earlier in-context opportunities of a k = 4 prompt, the tokens that actually differ between the
  nat and alt renderings (token-level diff inside the opportunity span; shared content tokens inside merged spans are dropped —
  `evidence_stats.csv` column `shared_tokens_dropped`). Whitespace-only evidence (indentation, blank lines, operator spaces) counts.
  `evidence_tokens.csv` lists the five most frequent evidence strings per pole and family; `evidence_stats.csv` the per-family audit
  (median 1–3 tokens per instance, comment_language 8; `outside_prompt` ≈ 0 everywhere).
- **Read feature** r_pole(L) = mean residual over all evidence tokens of the prompt at layer L (0 = embeddings, 1..28 = block outputs),
  averaged over the paired pool (documents with evidence in both poles; 193–200 per family). Read vector = r_nat − r_alt.
- Split-half cosine (halves by document) ≥ .95 at every layer for every family: the read vectors are stable.

## Read–write geometry (`read_features.csv`, `read_feature_summary.png`, `read_write_cos_by_layer.png`)

Cosine between the read vector at layer L and the paired cue-token write vector (`../steering/`, step 4):

| | L0 | L6 | L12 | L20 | L24 | L26 | L28 |
|---|---|---|---|---|---|---|---|
| cos(read at L, write at L24), median over 55 families | −.01 | −.01 | .02 | .05 | .20 | .20 | .07 |
| cos(read at L, write at L), median | – | .12 | .28 | .27 | .20 | .23 | .20 |

- The evidence-token difference is nearly orthogonal to the L24 write vector at early and middle layers and aligns with it only from
  L23 on (median .20 at L24–L26; 19 families reach ≥ .3 somewhere, 4 reach ≥ .5: comment_language .84, py_snake_camel .59,
  js_camel_snake .57, py_const_naming .53). Single-token families (quotes, `let`/`var`, `self`/`this`, `<-`/`=`, `===`/`==`) stay near 0.
- Same-layer alignment peaks around L11–L16 (median ≈ .3) — the read and write differences share a component in the middle of the
  network that is not the L24 write direction.
- Whether the read vector *works* (steering at the evidence tokens flips the continuation) is tested in `../read_steer/` (k = 3) and
  `../read_steer_k1/` (k = 1).
