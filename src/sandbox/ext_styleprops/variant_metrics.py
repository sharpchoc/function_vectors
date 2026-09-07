#!/usr/bin/env python
"""Shared metric implementation for the steering sandbox (one definition, all variants).

overall     = successes / ALL rollouts   <- PRIMARY (user decision 2026-09-07).
              A rollout counts as a success ONLY if it is judged coherent AND adopts the
              target convention. Incoherent rollouts and unscorable rollouts (the model
              never produced the feature) both count as FAILURES.
strict      = P(target convention | rollout coherent)   <- secondary: drops incoherent
              rollouts from the denominator instead of failing them.
conditional = P(target | coherent AND scorable)         <- secondary, conditional on the
              model reaching for the feature at all.
unscorable  = share of coherent rollouts the classifier cannot score.
incoherent  = share of all rollouts the LLM judge called gibberish (judge-fail counts as
              coherent so a judge outage cannot silently delete data).
"""
import numpy as np

from src.sandbox.ext_styleprops.properties import PROPS


def stats(prop: str, cond: dict, tgt: str = "alt") -> dict:
    tails = cond["tails"]
    coh = cond.get("coherent") or [None] * len(tails)
    labs = [PROPS[prop].classify(t) for t in tails]
    n = len(tails)
    coherent_idx = [i for i in range(n) if coh[i] is not False]
    good = [labs[i] for i in coherent_idx if labs[i] is not None]
    successes = sum(1 for i in coherent_idx if labs[i] == tgt)
    return dict(
        overall=successes / n if n else np.nan,
        strict=float(np.mean([labs[i] == tgt for i in coherent_idx])) if coherent_idx else np.nan,
        conditional=float(np.mean([l == tgt for l in good])) if good else np.nan,
        unscorable=sum(labs[i] is None for i in coherent_idx) / max(len(coherent_idx), 1),
        incoherent=sum(c is False for c in coh) / max(n, 1),
        n=n, n_coherent=len(coherent_idx), n_scored=len(good),
        judged=any(c is not None for c in coh),
    )
