# Read Directions — Evaluation Levers

Reference document for the design space of *read-direction evaluations*: the choices we
have to make when testing whether a candidate read direction $r_A$ (any point of the
definition crossing in `write_up/read_direction_levers.md`) is actually the direction the
model reads to identify task $A$. Terminology and notation follow
`write_up/task_id_im_subspaces.md` ($h_A$, $v_A$, $r^h_A$, $z^t_\ell$, task
identification vs task imitation space). Model: GPT-J-6B unless stated otherwise.

Two complementary test families, each with its own levers:

- **Steering** (sufficiency): write task-$A$ content along $r_A$ into a prompt that does
  not already determine the task, and ask whether the model behaves as if doing task $A$.
- **Ablation** (necessity): remove the $r_A$ component from prompts where the model *is*
  doing task $A$, and ask whether task behaviour (or downstream FV formation) degrades.

A good read direction should pass both; each alone is a weaker claim (a direction can be
sufficient without being the one the model uses, and redundant encodings can mask
necessity).

---

## Part I — Steering levers

### Lever S1 — Which read direction to steer with

The candidate vector itself is a lever: every point of the definition crossing in
`write_up/read_direction_levers.md` is a distinct steering vector. Listing the choices
by name (see that document for the exact definitions):

1. **Definition levers** (from `read_direction_levers.md`): optimisation metric (dot
   product vs cosine), inversion (literal pseudo-inverse vs $\tau$-truncated),
   aggregation across heads (summed circuit $M$ vs per-head-then-sum, with the
   equal-weight vs magnitude-weighted sub-choice), and final normalisation (unit norm vs
   natural magnitude).
2. **Target level: task-level vs averaged per-prompt.** The read direction can be built
   from the task-level target directly ($r_A$ from $v_A$ or $h_A$), or per prompt and
   then aggregated: compute $r^j_A$ from each per-prompt function vector $v^j_A$, then
   average over the task's prompt set,
   $$r_A \;=\; \frac{1}{|\mathcal{P}_A|} \sum_j r^j_A$$
   (unit-normalised or not, per the normalisation lever). The two differ whenever the
   construction is nonlinear in the target — which every pseudo-inverse-plus-normalise
   variant is — so averaging read directions is not the same as reading the averaged
   function vector.

**Note on the $\alpha$ sweep.** The steering-strength sweep applies only to
**unit-normed** read directions, where the construction has discarded scale and $\alpha$
must supply it. **Natural-magnitude** read directions already carry their own scale
(e.g. $\|M^{+} v_A\|$, the input scale needed to drive a $v_A$-sized output) and are
injected at that native magnitude ($\alpha = 1$).

### Lever S2 — Intervention layer(s)

Where in depth the write happens.

1. **Single layer, swept.** Inject at one layer $\ell$, sweep $\ell$ over all layers,
   report the best (or a fixed convention layer). Cheapest; but the read direction is
   *read by heads*, and the selected FV heads sit at many layers — an injection at layer
   $\ell$ is only visible to heads at layers $> \ell$, so a single-layer sweep partially
   confounds "which layer carries the identification signal" with "how many reader heads
   sit downstream".
2. **Layer band.** Inject the same content at every layer in a contiguous band
   $[\ell_1, \ell_2]$ (with or without renormalising $z$ per layer). Tests the signal as
   a *persistent* feature of the stream rather than a one-shot pulse; note that residual
   accumulation means the effective dose grows with band width (dose-matching against
   the single-layer case is a sub-choice).
3. **All layers up to the readers.** The limit of 2: hold the $r_A$ component clamped
   from the embedding up to the last FV-head layer.

### Lever S3 — Prompt pattern (what the demos carry)

The steering prompt is a $k$-shot scaffold (input–output demo pairs plus a query); the
lever is what content fills the demo **inputs** and the demo **outputs**, each
independently one of three options, giving a $3 \times 3$ grid:

| demo inputs \ demo outputs | in-distribution | out-of-distribution | dummy (e.g. `_`) |
|---|---|---|---|
| **in-distribution** | clean ICL — prompt already determines the task; steering has nothing to add (ceiling / control) | conflicting mapping evidence | format + input statistics, no mapping |
| **out-of-distribution** | conflicting | fully OOD content, format intact | OOD inputs, no mapping |
| **dummy (e.g. `Input`)** | outputs alone hint the range | inputs empty, outputs misleading | pure format scaffold — all task identity must come from the injection |

- **In-distribution**: real task-$A$ pairs (or real inputs / real outputs separately).
- **Out-of-distribution**: content from other tasks or random English words — plausible
  text, wrong task.
- **Dummy**: a constant placeholder token, e.g. every input is the literal string
  `Input`, or every output is `_` — no semantic content at all, only the format.

Interpretation: moving away from the in-distribution/in-distribution corner removes the
prompt's own evidence about the task, so the steering injection has to supply more of
the identification signal. The dummy/dummy cell is the cleanest sufficiency test; the
in-distribution-inputs row tests whether the injection can *override or complete* partial
evidence; OOD cells test robustness to distractors. (Sub-choices, held fixed within a
sweep: the number of shots $k$; the query input itself — normally a real task-$A$ input,
since success is judged by the model producing the correct $A$-output for it; and
whether the zero-shot prompt, $k = 0$, is included as the degenerate row/column.)

### Lever S4 — Intervention type

How the write is performed at the chosen site $z$ (token position(s) $\times$ Lever S2
layer(s)).

1. **Patching.** Replace the site wholesale with a donor activation:
   $z \leftarrow z^{\ast}$, where $z^{\ast}$ is the activation at the same site under a
   clean task-$A$ prompt. Maximal signal, minimal specificity — carries everything the
   donor stream carries, not just the read direction; upper-bounds what any
   direction-level intervention can do.
2. **Single-direction steering.** Additively inject along the candidate direction:
   $$z \leftarrow z + \alpha\, r_A,$$
   with $\alpha$ swept for unit-normed $r_A$ and fixed at $1$ for natural-magnitude
   $r_A$ (see the Lever S1 note). Tests the direction itself; specificity controls are
   dimension-matched random directions of the same norm.
3. **Multi-direction (subspace) steering.** Given a read subspace $R$ with orthonormal
   basis $U$ (e.g. the span of read directions across tasks) and projector
   $P_R = U U^\top$: first *ablate* the prompt's own content in the subspace, then
   *write* the target content into it,
   $$z \leftarrow (I - P_R)\, z + \alpha\, r_A
     \quad\text{(or } + P_R z^{\ast}\text{, the patched variant)}.$$
   Distinguishes "the subspace is the channel" from "this direction added on top of
   whatever was already in the channel"; strictly stronger than 2 when the prompt
   pattern (Lever S3) leaves competing task evidence in $R$.

---

## Part II — Ablation levers

### Lever A1 — Ablation layer(s)

Same options as Lever S2 (single swept layer / band / all layers up to the readers), but
the redundancy concern points the opposite way: a signal ablated at one layer can be
*recomputed* by later layers from surviving upstream copies, so single-layer ablations
under-estimate necessity. The clamped all-layer ablation (remove the component at every
layer from the first ablation layer onward) is the honest necessity test; the
single-layer sweep is diagnostic of *where* the signal is written.

### Lever A2 — Ablation tokens

Which token positions are ablated:

1. **All tokens.** Strongest removal; least localisation.
2. **Demo tokens only** (all positions before the query). Tests whether the direction
   carries task identification accumulated from the examples.
3. **Label/output tokens only** vs **input tokens only.** Splits where in the demos the
   identification signal lives.
4. **Query / final cue token only.** Tests the direction at the point where the FV heads
   actually read (per-token granularity of the read claim: identification is
   hypothesised to occur at earlier tokens, so an effect that only appears when the cue
   token is ablated is a different claim than one driven by demo tokens).

### Lever A3 — Zero-projection vs mean ablation

What value the ablated component is set to.

1. **Zero projection.** Remove the component entirely:
   $$z \leftarrow z - (r_A^\top z)\, r_A.$$
   Simple, but sets the site to a value (zero along $r_A$) that may itself be
   off-distribution — degradation can then reflect the OOD-ness of the ablated
   activation rather than the loss of task information.
2. **Mean ablation.** Replace the component with its mean over a reference
   distribution:
   $$z \leftarrow z + \left(r_A^\top(\mu - z)\right) r_A,
     \qquad \mu = \mathbb{E}_{\text{ref}}[z].$$
   Keeps the site on-distribution while destroying the task-specific *variation* along
   $r_A$. Sub-choice: the reference distribution for $\mu$ (all tasks pooled at that
   site; the same prompt-pattern cell; the same task — the last removes nothing
   task-identifying and is a null control, not an ablation).

### Lever A4 — Single-direction vs multi-direction ablation

1. **Single direction.** Project out $r_A$ alone (either variant of Lever A3). Subject
   to redundancy: if task identity is encoded in several directions, removing one may
   show no effect even when the direction is genuinely used.
2. **Multi-direction (subspace).** Project out a read subspace $R$ with projector
   $P_R = U U^\top$ (rank $k$):
   $$z \leftarrow (I - P_R)\, z \qquad\text{or its mean-ablated analogue}
     \quad z \leftarrow (I - P_R)\, z + P_R\, \mu.$$
   Stronger necessity test; the cost is collateral damage risk, controlled by
   dimension-matched *random* rank-$k$ subspaces (the effect of ablating $R$ should
   exceed the effect of ablating random subspaces of the same rank at the same sites).

---

## Success metric (both parts)

What is measured after intervening is a further choice, shared with the isolation-method
study (`write_up/isolation_methods_levers.md`, Lever 4): task accuracy of the steered /
ablated model (which readout convention), uplift or degradation relative to the
unintervened baseline on the *same* prompts, and — for ablation — optionally the effect
on downstream quantities ($h_A$ at the FV heads, or $v_A$ formation) rather than
behaviour. The metric, the prompt set it is computed on, and the controls (random
directions / random subspaces, dose-matching) must be fixed before any sweep over the
levers above is run.
