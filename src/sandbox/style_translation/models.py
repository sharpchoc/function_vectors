"""Model registry for the style-translation study (2026-09-11): GPT-J-6B is the default; other base
models get their own tokeniser-specific cue tokens, prompts, rollouts and results folder.
    paths("gptj")        -> the original locations (pairs carry the cues; artifacts/style_translation/{prompts,rollouts}; results/style_translation)
    paths("qwen25_base") -> artifacts/style_translation/qwen25_base/{cues,prompts,rollouts}; results/style_translation/qwen25_base
"""
from pathlib import Path
from src.utils.paths import ARTIFACTS_ROOT, STYLE_TRANSLATION_RESULTS

MODELS = {
    "gptj": dict(tokenizer="EleutherAI/gpt-j-6B", hub="models--EleutherAI--gpt-j-6b", label="GPT-J-6B", dtype="float16"),
    "qwen25_base": dict(tokenizer="Qwen/Qwen2.5-7B", hub="models--Qwen--Qwen2.5-7B", label="Qwen2.5-7B (base)", dtype="bfloat16"),
}
DEFAULT = "gptj"


def paths(model=DEFAULT):
    """Per-model locations. Feature-pipeline roots (added 2026-09-14 for the Qwen port): steering/{vectors,screen,
    confirm,common_layer}, read_features(+/evidence), read_steer, prompt_pairs — GPT-J keeps its historical folders."""
    m = MODELS[model]
    if model == DEFAULT:
        root = ARTIFACTS_ROOT / "style_translation"
        return dict(m, name=model, cues=None, prompts=root / "prompts", rollouts=root / "rollouts", results=STYLE_TRANSLATION_RESULTS,
                    steering=root / "steering", read_features=root / "read_features", evidence=root / "read_features" / "evidence",
                    read_steer=root / "read_steer", prompt_pairs=root / "prompt_pairs")
    root = ARTIFACTS_ROOT / "style_translation" / model
    return dict(m, name=model, cues=root / "cues", prompts=root / "prompts", rollouts=root / "rollouts",
                results=STYLE_TRANSLATION_RESULTS / model, steering=root / "steering", read_features=root / "read_features",
                evidence=root / "read_features" / "evidence", read_steer=root / "read_steer", prompt_pairs=root / "prompt_pairs")


def arch(model):
    """Architecture handles shared by the hooks and captures: blocks[L-1] = transformer block L, embed = token embedding
    (layer 0 = its output; neither GPT-J nor Qwen2 adds a positional vector to the residual stream), trunk = the base
    model (hidden_states[0..n_layers], the last one normed), hidden size, layer count, parameter dtype."""
    cfg = model.config
    if hasattr(model, "transformer"):                       # GPT-J
        t = model.transformer
        return dict(blocks=t.h, embed=t.wte, trunk=t, hidden=cfg.n_embd, n_layers=cfg.n_layer, dtype=next(model.parameters()).dtype)
    m = model.model                                         # Qwen2 / Llama family
    return dict(blocks=m.layers, embed=m.embed_tokens, trunk=m, hidden=cfg.hidden_size, n_layers=cfg.num_hidden_layers,
                dtype=next(model.parameters()).dtype)


def snapshot_dir(model=DEFAULT):
    snaps = sorted(Path("/workspace/.cache/huggingface/hub").joinpath(MODELS[model]["hub"], "snapshots").glob("*"))
    snaps = [s for s in snaps if (s / "config.json").exists()]
    assert snaps, f"no complete snapshot for {model}; pass --model_dir"
    return snaps[-1]
