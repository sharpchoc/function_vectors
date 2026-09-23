"""Cue-token steering hook (step 4); model-agnostic via models.arch (GPT-J, Qwen2.5).

Layer convention: L in 1..28 = output of transformer block h[L-1] = `hidden_states[L]` from
`output_hidden_states=True` (index 0 is the embedding output). NOTE hidden_states[28] has ln_f
applied in HF's GPT-J, so L = 28 is not used for steering (screen layers stop at 24).

Injection (user decision 2026-09-07): add alpha * v to the residual stream at layer L at the CUE
TOKEN ONLY = the last prompt position of the prefill pass. With left padding every batch row's cue
is at index -1. Decode steps (sequence length 1) are untouched.
"""
import torch

from src.sandbox.style_translation.models import arch


class CueSteer:
    def __init__(self, model, layer, vec, alpha):
        A = arch(model)
        self.block = A["blocks"][layer - 1]
        self.vec = torch.as_tensor(vec, dtype=A["dtype"], device=next(model.parameters()).device)
        self.alpha = float(alpha)
        self.handle = None
        self.calls = 0

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1 and self.alpha != 0.0:          # prefill pass only -> cue token
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + self.alpha * self.vec
            self.calls += 1
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return output

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()


def unit_test(model, tok, layer=6):
    """alpha = 0 leaves logits unchanged; alpha = 1 shifts exactly the last position of layer L by v."""
    enc = tok(["Spanish:\nHola mundo.\n\nEnglish:\nHello", "Spanish:\nAdiós.\n\nEnglish:\nGood"],
              return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        base = model(**enc, output_hidden_states=True)
        v = torch.randn(arch(model)["hidden"], device=model.device, generator=torch.Generator(device=model.device).manual_seed(0)) * 0.1
        with CueSteer(model, layer, v, 0.0):
            zero = model(**enc, output_hidden_states=True)
        with CueSteer(model, layer, v, 1.0) as s:
            one = model(**enc, output_hidden_states=True)
    assert torch.equal(base.logits, zero.logits), "alpha=0 must be an identity"
    d = (one.hidden_states[layer] - base.hidden_states[layer]).float()
    # dtype-aware tolerance: hidden entries at late layers reach the hundreds; spacing = eps(dtype) * |h| (fp16 1e-3, bf16 8e-3)
    rel = 1.5 * torch.finfo(arch(model)["dtype"]).eps
    h = base.hidden_states[layer][:, -1, :].float().abs()
    assert ((d[:, -1, :] - v.float()).abs() <= 2e-2 + rel * h).all(), "last position must move by v"
    assert d[:, :-1, :].abs().max().item() == 0.0, "other positions must be unchanged"
    assert s.calls == 1
    return True


class PositionSteer:
    """Step 7: add alpha * v at an arbitrary list of positions per batch row (prefill pass only).

    positions: list (one per batch row) of lists of absolute indices into the left-padded sequence
    (caller offsets the prompt-relative indices by the padding). Decode steps (length 1) untouched.
    """
    def __init__(self, model, layer, vec, alpha, positions):
        # layer 0 = the embedding output (no positional vector in the residual stream for GPT-J / Qwen2), L >= 1 = output of block L
        A = arch(model)
        self.block = A["embed"] if layer == 0 else A["blocks"][layer - 1]
        self.vec = torch.as_tensor(vec, dtype=A["dtype"], device=next(model.parameters()).device)
        self.alpha = float(alpha)
        self.positions = positions
        self.handle = None
        self.calls = 0

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1 and self.alpha != 0.0:
            h = h.clone()
            for r, pos in enumerate(self.positions):
                if pos:
                    h[r, pos, :] = h[r, pos, :] + self.alpha * self.vec
            self.calls += 1
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return output

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()


def unit_test_positions(model, tok, layer=6):
    """Works for layer = 0 (embedding output) as well as block outputs."""
    """alpha = 0 identity; alpha = 1 shifts exactly the listed positions of layer L by v, nothing else."""
    enc = tok(["Spanish:\nHola mundo.\n\nEnglish:\nHello there my", "Spanish:\nAdiós.\n\nEnglish:\nGood"],
              return_tensors="pt", padding=True).to(model.device)
    L = enc.input_ids.shape[1]
    positions = [[L - 3, L - 2], [L - 1]]                 # row 0: two evidence tokens before the cue; row 1: last token
    with torch.no_grad():
        base = model(**enc, output_hidden_states=True)
        v = torch.randn(arch(model)["hidden"], device=model.device, generator=torch.Generator(device=model.device).manual_seed(1)) * 0.1
        with PositionSteer(model, layer, v, 0.0, positions):
            zero = model(**enc, output_hidden_states=True)
        with PositionSteer(model, layer, v, 1.0, positions) as s:
            one = model(**enc, output_hidden_states=True)
    assert torch.equal(base.logits, zero.logits), "alpha=0 must be an identity"
    d = (one.hidden_states[layer] - base.hidden_states[layer]).float()
    for r, pos in enumerate(positions):
        hmag = base.hidden_states[layer][r, pos, :].float().abs()
        rel = 1.5 * torch.finfo(arch(model)["dtype"]).eps
        assert ((d[r, pos, :] - v.float()).abs() <= 2e-2 + rel * hmag).all(), "listed positions must move by v"
        others = [j for j in range(L) if j not in pos]
        assert d[r, others, :].abs().max().item() == 0.0, "other positions must be unchanged"
    assert s.calls == 1
    return True


class CueAblate:
    """Write-feature ablation at the CUE TOKEN (2026-09-23, code-convention families): at layer L, last prompt position of the prefill
    pass, remove the component of the residual along the unit direction w (mode "zero": h -= (h.w) w) or replace it by a fixed value
    (mode "mean": h += (m - h.w) w). Decode steps untouched. `proj` collects the pre-ablation projections (h.w) per batch row."""
    def __init__(self, model, layer, w_unit, mode, mean_value=0.0):
        A = arch(model)
        self.block = A["blocks"][layer - 1]
        w = torch.as_tensor(w_unit, dtype=torch.float32, device=next(model.parameters()).device)
        self.w = (w / w.norm()).to(A["dtype"]); self.w32 = (w / w.norm())
        self.mode, self.m = mode, float(mean_value)
        self.handle = None; self.calls = 0; self.proj = None

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1:
            h = h.clone()
            last = h[:, -1, :].float(); p = last @ self.w32                      # [B]
            self.proj = p.detach().cpu()
            target = torch.zeros_like(p) if self.mode == "zero" else torch.full_like(p, self.m)
            h[:, -1, :] = (last + (target - p)[:, None] * self.w32[None, :]).to(h.dtype)
            self.calls += 1
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return output

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook); return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()


def unit_test_ablate(model, tok, layer=6):
    """zero mode leaves |h.w| ~ 0 at the last position; mean mode sets h.w = m; other positions untouched."""
    enc = tok(["Spanish:\nHola mundo.\n\nEnglish:\nHello", "Spanish:\nAdiós.\n\nEnglish:\nGood"], return_tensors="pt", padding=True).to(model.device)
    w = torch.randn(arch(model)["hidden"], generator=torch.Generator().manual_seed(3)); w = w / w.norm()
    with torch.no_grad():
        base = model(**enc, output_hidden_states=True)
        with CueAblate(model, layer, w, "zero") as a0:
            z = model(**enc, output_hidden_states=True)
        with CueAblate(model, layer, w, "mean", 7.0) as a1:
            m = model(**enc, output_hidden_states=True)
    wd = w.to(model.device)
    tol = 0.05 + 3 * torch.finfo(arch(model)["dtype"]).eps * base.hidden_states[layer][:, -1, :].float().abs().max().item()
    pz = z.hidden_states[layer][:, -1, :].float() @ wd; pm = m.hidden_states[layer][:, -1, :].float() @ wd
    assert pz.abs().max().item() <= tol, f"zero mode projection {pz}"
    assert (pm - 7.0).abs().max().item() <= tol, f"mean mode projection {pm}"
    assert torch.equal(z.hidden_states[layer][:, :-1, :], base.hidden_states[layer][:, :-1, :]), "other positions must be unchanged"
    assert a0.calls == 1 and a1.calls == 1 and a0.proj is not None
    return True


class MultiCueAblate:
    """CueAblate at SEVERAL layers at once (2026-09-23, user decision: ablate at every layer at the cue token): one hook per layer, each
    with that layer's own unit direction w[l-1] and mean value m[l-1]. `proj_at(layer)` = pre-ablation projections at that layer."""
    def __init__(self, model, layers, w_by_layer, mode, m_by_layer=None):
        self.hooks = [CueAblate(model, l, w_by_layer[l - 1], mode, 0.0 if m_by_layer is None else float(m_by_layer[l - 1])) for l in layers]
        self.layers = list(layers)

    def proj_at(self, layer):
        return self.hooks[self.layers.index(layer)].proj

    def __enter__(self):
        for h in self.hooks: h.__enter__()
        return self

    def __exit__(self, *exc):
        for h in self.hooks: h.__exit__(*exc)


class PositionAblate:
    """Read-feature ablation at a list of positions per batch row (2026-09-23): at layer L (0 = embedding output), remove the component of
    the residual along the unit direction w at those positions (mode "zero") or set it to a fixed value m (mode "mean"). Prefill only.
    `proj` = per-row mean of the pre-ablation projections over the row's positions."""
    def __init__(self, model, layer, w_unit, mode, mean_value, positions):
        A = arch(model)
        self.block = A["embed"] if layer == 0 else A["blocks"][layer - 1]
        w = torch.as_tensor(w_unit, dtype=torch.float32, device=next(model.parameters()).device); self.w32 = w / w.norm()
        self.mode, self.m, self.positions = mode, float(mean_value), positions
        self.handle = None; self.calls = 0; self.proj = None

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1:
            h = h.clone(); pr = []
            for r, pos in enumerate(self.positions):
                if not pos:
                    pr.append(float("nan")); continue
                x = h[r, pos, :].float(); p = x @ self.w32; pr.append(p.mean().item())
                target = torch.zeros_like(p) if self.mode == "zero" else torch.full_like(p, self.m)
                h[r, pos, :] = (x + (target - p)[:, None] * self.w32[None, :]).to(h.dtype)
            self.proj = pr; self.calls += 1
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return output

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook); return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()


class MultiPositionAblate:
    """PositionAblate at several layers at once, each with its own direction w_by_layer[l] and mean m_by_layer[l] (index = layer, 0 = embed)."""
    def __init__(self, model, layers, w_by_layer, mode, m_by_layer, positions):
        self.layers = list(layers)
        self.hooks = [PositionAblate(model, l, w_by_layer[l], mode, 0.0 if m_by_layer is None else float(m_by_layer[l]), positions) for l in self.layers]

    def proj_at(self, layer):
        return self.hooks[self.layers.index(layer)].proj

    def __enter__(self):
        for h in self.hooks: h.__enter__()
        return self

    def __exit__(self, *exc):
        for h in self.hooks: h.__exit__(*exc)


def unit_test_position_ablate(model, tok, layer=6):
    enc = tok(["Spanish:\nHola mundo.\n\nEnglish:\nHello", "Spanish:\nAdiós.\n\nEnglish:\nGood"], return_tensors="pt", padding=True).to(model.device)
    w = torch.randn(arch(model)["hidden"], generator=torch.Generator().manual_seed(4)); w = w / w.norm(); wd = w.to(model.device)
    pos = [[2, 5], [3]]
    with torch.no_grad():
        base = model(**enc, output_hidden_states=True)
        with PositionAblate(model, layer, w, "zero", 0.0, pos) as a0:
            z = model(**enc, output_hidden_states=True)
        with PositionAblate(model, 0, w, "mean", 3.0, pos) as a1:
            m = model(**enc, output_hidden_states=True)
    hb = base.hidden_states[layer]; hz = z.hidden_states[layer]; hm = m.hidden_states[0]
    tol = 0.05 + 3 * torch.finfo(arch(model)["dtype"]).eps * hb.float().abs().max().item()
    for r, p in enumerate(pos):
        assert (hz[r, p, :].float() @ wd).abs().max().item() <= tol, "zero mode"
        assert ((hm[r, p, :].float() @ wd) - 3.0).abs().max().item() <= tol, "mean mode at the embedding"
        keep = [i for i in range(hb.shape[1]) if i not in p]
        assert torch.equal(hz[r, keep, :], hb[r, keep, :]), "other positions must be unchanged"
    assert a0.calls == 1 and a1.calls == 1 and len(a0.proj) == 2
    return True
