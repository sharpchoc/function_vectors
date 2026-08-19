# Read Directions — Definition Levers

Reference document for the design space of *read directions*: the input-side directions
that the OV circuits of the selected FV heads map onto the task imitation space.
Companion to `write_up/isolation_methods_levers.md`; terminology and notation follow
`write_up/task_id_im_subspaces.md` ($h_A$, $v_A$, $v^j_A$, $r^h_A$, summed circuit $M$).
Model: GPT-J-6B ($d_{\mathrm{model}} = 4096$, $d_{\mathrm{head}} = 256$, so each
$W^h_O W^h_V \in \mathbb{R}^{4096 \times 4096}$ has rank $\le 256$).

Motivation: there is more than one defensible definition of a read direction, and the
choices interact. Rather than committing to one, we enumerate the levers so a sweep can
pick the best definition empirically. The glossary's current definitions (eqs. 1–5 in
`task_id_im_subspaces.md`) are one point in this crossing, marked below.

Throughout, "target" means the vector the read direction should reproduce through the
circuit — $h_A$ for a single head, $v_A$ or $v^j_A$ for the summed construction. The
task-level vs per-prompt choice of target is orthogonal to the levers below and is held
fixed within any sweep.

---

## Lever 1 — Optimisation metric: dot product vs cosine similarity

What "the input direction whose image is maximally aligned with the target" means. Let
$W$ stand for the relevant circuit ($W^h_O W^h_V$ or $M$) and $t$ for the target.

1. **Dot product.** $r = \arg\max_{\|z\|=1} \langle W z,\, t \rangle$. Closed form:

   $$r \;\propto\; W^\top t.$$

   No kernel constraint is needed ($W^\top t$ lies in the row space of $W$
   automatically). Rewards output *magnitude* along $t$: directions the circuit
   transmits strongly are favoured even if their image is less purely aligned with $t$.

2. **Cosine similarity.** $r = \arg\max_{\|z\|=1,\; z \perp \ker(W)} \cos(W z,\, t)$.
   Closed form:

   $$r \;\propto\; W^{+} t$$

   (truncated pseudo-inverse; the kernel constraint makes the argmax unique). Indifferent
   to output magnitude: rewards purity of alignment, but $W^{+}$ scales by $1/\sigma_i$,
   so near-zero singular values can dominate the solution (the near-kernel caveat in the
   glossary). Lever 2 exists to control this.

The two are genuinely different optimisation problems (the glossary's "Previous
misunderstanding" note, point 1): the unit-norm constraint fixes the *input* norm, but
cosine additionally normalises by the *output* norm $\|Wz\|$, which varies with $z$.

---

## Lever 2 — Inversion: literal pseudo-inverse vs $\tau$-truncation (cosine metric only)

Applies when Lever 1 = cosine (the dot-product solution involves no inversion, so this
lever collapses there).

1. **Literal pseudo-inverse $W^{+}$.** Invert every nonzero singular value. Exact
   maximiser of the cosine objective, but amplifies whatever the circuit barely
   transmits: components of $t$ along small-$\sigma$ output directions get weight
   $1/\sigma_i$ in the solution.

2. **$\tau$-truncated pseudo-inverse $W^{+}_\tau$.** Invert only singular values
   $\sigma_i \ge \tau$ (equivalently: additionally constrain $z$ to be orthogonal to the
   right-singular subspace with $\sigma_i < \tau$). Trades a small loss in achievable
   cosine for robustness to directions the circuit does not meaningfully transmit.
   $\tau$ (or, equivalently, a rank cut $k$) is a hyperparameter to sweep; $\tau \to 0$
   recovers option 1, and truncating to the top-1 singular direction is the other
   extreme.

(A truncated variant of the dot-product solution — projecting $W^\top t$ onto the
top-$k$ right-singular subspace — is definable but out of scope unless the sweep asks
for it.)

---

## Lever 3 — Aggregation across heads: summed circuit vs per-head-then-sum

How the multiple selected heads $H$ combine into one read direction per task.

1. **Summed circuit ("big $M$").** Form
   $M = \sum_{h \in H} W^h_O W^h_V$, target $= v_A$ (or $v^j_A$), and solve the Lever 1
   problem once against $M$. This is the glossary §3 construction. It treats every head
   as reading the *same* residual-stream vector $z$, even though the heads sit at
   different layers — a physical inconsistency accepted for tractability. Rank of $M$
   can reach $256\,|H|$, so with $|H| \gtrsim 16$ it can be full rank and the kernel
   constraint does nothing (making Lever 2 truncation the only regulariser).

2. **Per-head, then sum and normalise.** For each head $h \in H$, solve the Lever 1
   problem for that head's own circuit $W^h_O W^h_V$ against that head's own mean
   activation $h_A$, giving $r^h_A$; then

   $$r_A \;=\; \mathrm{normalise}\!\left(\sum_{h \in H} r^h_A\right).$$

   Respects each head's own input geometry; loses cross-head interactions (no head can
   compensate for another's blind spot, unlike in $M$).

   *Sub-choice (3′):* each $r^h_A$ is unit-norm by definition, so the plain sum weights
   heads equally. An alternative weights each head by its unnormalised solution
   magnitude (e.g. $\|(W^h_O W^h_V)^{+} h_A\|$, or $\|(W^h_O W^h_V)^\top h_A\|$ under
   the dot-product metric) before summing. Flag which is used in any run.

---

## Lever 4 — Final normalisation: unit norm or natural magnitude

Whether the final $r_A$ is unit-normalised.

1. **Unit norm.** $r_A$ is a pure direction; any injection/ablation strength is
   delegated entirely to the $\alpha$ sweep. The glossary's definitions (eqs. 2 and 5)
   take this option.

2. **Natural magnitude.** Keep the norm the construction produces —
   $\|W^{+} t\|$ (cosine), $\|W^\top t\|$ (dot product), or the norm of the per-head
   sum (Lever 3.2). These magnitudes carry meaning (e.g. $\|M^{+} v_A\|$ is the input
   scale needed to produce a $v_A$-sized output through $M$), and they differ *across
   tasks and heads*, so option 2 changes relative dosing in any multi-task comparison
   at fixed $\alpha$.

This lever is moot for purely correlational uses (cosine of $r_A$ against residual
activations is scale-invariant); it bites for causal uses (injection, ablation,
projection-removal at fixed $\alpha$).

---

## The crossing

- Dot product (Lever 2 collapses): $1 \times 2 \times 2 = 4$ definitions.
- Cosine: $\{$literal, $\tau$-truncated over a $\tau$ grid$\}$ $\times\ 2 \times 2$
  definitions.

Existing named points:

- Glossary §2 $r^h_A$ (single head) = cosine + literal $W^{+}$ + (per-head, no
  aggregation) + unit norm.
- Glossary §3 $r^j_A$ / $r_A$ = cosine + literal $M^{+}$ + summed circuit + unit norm.
- The retired *d_payload* transpose formula = dot product (the glossary's "previous
  misunderstanding" was mislabelling it as the cosine solution — as a *dot-product*
  read direction it is a legitimate lever point, not an error).

## Before any sweep

The success metric for "best definition" is itself a definitional choice (correlational
alignment at early tokens? causal effect of ablating/injecting along $r_A$? which prompt
setting?) and must be user-adjudicated before compute, per the standing rule on
definitional geometry choices.

Implementation note: pseudo-inverses and truncations here run through SVD; on CUDA, pin
`torch.linalg.svd(..., driver="gesvd")` — the default gesvdj is only ~1e-3 accurate in
fp32, which is material when inverting small singular values (DECISIONS conventions).
