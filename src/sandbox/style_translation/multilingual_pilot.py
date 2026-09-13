#!/usr/bin/env python
"""Pilot (2026-09-13): can GPT-J translate short English texts into German / Portuguese / French / Spanish (and Dutch,
Romanian as expected-fail references) well enough for the style-translation pipeline? Prompt mirrors the study's
header layout ("English:\\n{text}\\n\\n{Lang}:\\n"), k = 0 (no in-text convention examples), one seeded T = 1 sample of
up to 160 tokens, and a second arm with a one-sentence demo pair in front (headroom check).
Sources: the first 20 house-style English texts of the `ampersand` family (bicycle topics, ~160 words), truncated to
their first 3 sentences (~60 words) so the translation fits in 160 tokens.
Output: artifacts/style_translation/multilingual_pilot/rollouts.json (list of dicts: lang, arm, doc_id, source, output)."""
import json, re, sys
from pathlib import Path
import torch
_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import ARTIFACTS_ROOT
from src.sandbox.style_translation.rollout import load_model

LANGS = {"de": "German", "pt": "Portuguese", "fr": "French", "es": "Spanish", "nl": "Dutch", "ro": "Romanian"}
DEMO = {"de": ("The weather is nice today, so we are going for a walk in the park.", "Das Wetter ist heute schön, deshalb machen wir einen Spaziergang im Park."),
        "pt": ("The weather is nice today, so we are going for a walk in the park.", "O tempo está bom hoje, por isso vamos dar um passeio no parque."),
        "fr": ("The weather is nice today, so we are going for a walk in the park.", "Il fait beau aujourd'hui, alors nous allons nous promener dans le parc."),
        "es": ("The weather is nice today, so we are going for a walk in the park.", "Hoy hace buen tiempo, así que vamos a dar un paseo por el parque."),
        "nl": ("The weather is nice today, so we are going for a walk in the park.", "Het weer is vandaag mooi, dus we gaan een wandeling maken in het park."),
        "ro": ("The weather is nice today, so we are going for a walk in the park.", "Vremea este frumoasă astăzi, așa că mergem la o plimbare în parc.")}
OUT = ARTIFACTS_ROOT / "style_translation" / "multilingual_pilot"

def first_sentences(t, n=3):
    s = re.split(r"(?<=[.!?])\s+", t.strip()); return " ".join(s[:n])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = json.load(open(_BOOT / "dataset_files" / "style_translation" / "english" / "ampersand.json"))[:20]
    srcs = [(r["doc_id"], first_sentences(r["text_nat"])) for r in recs]
    model, tok = load_model()
    out = []
    for code, lang in LANGS.items():
        for arm in ("header", "demo"):
            prompts = []
            for doc, text in srcs:
                p = f"English:\n{text}\n\n{lang}:\n"
                if arm == "demo":
                    p = f"English:\n{DEMO[code][0]}\n\n{lang}:\n{DEMO[code][1]}\n\n" + p
                prompts.append(p)
            for i in range(0, len(prompts), 10):
                batch = prompts[i:i + 10]
                enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
                torch.manual_seed(0)
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=160, pad_token_id=tok.eos_token_id)
                for j, g in enumerate(gen):
                    txt = tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True)
                    txt = txt.split("\n\n")[0].strip()          # stop at the next blank line (next header)
                    out.append(dict(lang=code, arm=arm, doc_id=srcs[i + j][0], source=srcs[i + j][1], output=txt))
            print(f"{lang:10s} {arm:6s} done; sample: {out[-1]['output'][:120]!r}", flush=True)
    json.dump(out, open(OUT / "rollouts.json", "w"), ensure_ascii=False, indent=1)
    print("saved", len(out))

if __name__ == "__main__":
    main()
