"""Cue-token steering hook for GPT-J (step 4).

Layer convention: L in 1..28 = output of transformer block h[L-1] = `hidden_states[L]` from
`output_hidden_states=True` (index 0 is the embedding output). NOTE hidden_states[28] has ln_f
applied in HF's GPT-J, so L = 28 is not used for steering (screen layers stop at 24).

Injection (user decision 2026-09-07): add alpha * v to the residual stream at layer L at the CUE
TOKEN ONLY = the last prompt position of the prefill pass. With left padding every batch row's cue
is at index -1. Decode steps (sequence length 1) are untouched.
"""
import torch


class CueSteer:
    def __init__(self, model, layer, vec, alpha):
        self.block = model.transformer.h[layer - 1]
        self.vec = torch.as_tensor(vec, dtype=torch.float16, device=next(model.parameters()).device)
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
        v = torch.randn(model.config.n_embd, device=model.device, generator=torch.Generator(device=model.device).manual_seed(0)) * 0.1
        with CueSteer(model, layer, v, 0.0):
            zero = model(**enc, output_hidden_states=True)
        with CueSteer(model, layer, v, 1.0) as s:
            one = model(**enc, output_hidden_states=True)
    assert torch.equal(base.logits, zero.logits), "alpha=0 must be an identity"
    d = (one.hidden_states[layer] - base.hidden_states[layer]).float()
    # fp16-aware tolerance: hidden entries at late layers reach the hundreds, where the fp16 spacing is 0.25
    h = base.hidden_states[layer][:, -1, :].float().abs()
    assert ((d[:, -1, :] - v.float()).abs() <= 2e-2 + 1.5e-3 * h).all(), "last position must move by v"
    assert d[:, :-1, :].abs().max().item() == 0.0, "other positions must be unchanged"
    assert s.calls == 1
    return True


class PositionSteer:
    """Step 7: add alpha * v at an arbitrary list of positions per batch row (prefill pass only).

    positions: list (one per batch row) of lists of absolute indices into the left-padded sequence
    (caller offsets the prompt-relative indices by the padding). Decode steps (length 1) untouched.
    """
    def __init__(self, model, layer, vec, alpha, positions):
        # layer 0 = the embedding output (GPT-J adds no positional vector to the residual stream), L >= 1 = output of block L
        self.block = model.transformer.wte if layer == 0 else model.transformer.h[layer - 1]
        self.vec = torch.as_tensor(vec, dtype=torch.float16, device=next(model.parameters()).device)
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
        v = torch.randn(model.config.n_embd, device=model.device, generator=torch.Generator(device=model.device).manual_seed(1)) * 0.1
        with PositionSteer(model, layer, v, 0.0, positions):
            zero = model(**enc, output_hidden_states=True)
        with PositionSteer(model, layer, v, 1.0, positions) as s:
            one = model(**enc, output_hidden_states=True)
    assert torch.equal(base.logits, zero.logits), "alpha=0 must be an identity"
    d = (one.hidden_states[layer] - base.hidden_states[layer]).float()
    for r, pos in enumerate(positions):
        hmag = base.hidden_states[layer][r, pos, :].float().abs()
        assert ((d[r, pos, :] - v.float()).abs() <= 2e-2 + 1.5e-3 * hmag).all(), "listed positions must move by v"
        others = [j for j in range(L) if j not in pos]
        assert d[r, others, :].abs().max().item() == 0.0, "other positions must be unchanged"
    assert s.calls == 1
    return True
