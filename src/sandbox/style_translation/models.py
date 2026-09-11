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
    m = MODELS[model]
    if model == DEFAULT:
        root = ARTIFACTS_ROOT / "style_translation"
        return dict(m, name=model, cues=None, prompts=root / "prompts", rollouts=root / "rollouts", results=STYLE_TRANSLATION_RESULTS)
    root = ARTIFACTS_ROOT / "style_translation" / model
    return dict(m, name=model, cues=root / "cues", prompts=root / "prompts", rollouts=root / "rollouts",
                results=STYLE_TRANSLATION_RESULTS / model)


def snapshot_dir(model=DEFAULT):
    snaps = sorted(Path("/workspace/.cache/huggingface/hub").joinpath(MODELS[model]["hub"], "snapshots").glob("*"))
    snaps = [s for s in snaps if (s / "config.json").exists()]
    assert snaps, f"no complete snapshot for {model}; pass --model_dir"
    return snaps[-1]
