#!/usr/bin/env python
"""Generate the Spanish source corpus: N candidate paragraphs per style family (Gemini 2.5
Flash via OpenRouter, all families in one parallel pool), then a deterministic convention
post-edit + audit and a regex pre-count of the family's opportunities.

Output: dataset_files/style_translation/spanish/<family>.json
  [{doc_id, family, topic, angle, text_es, words, sentences, regex_k, violations:[...],
    verify: null | {...}, pass: null | bool}]
Resumable per doc_id. `--audit_only` re-runs post-edit/audit/regex on stored texts.

Conventions and per-family instructions: families.py (user decisions 2026-09-07).
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.utils.paths import STYLE_TRANSLATION_DATA
from src.sandbox.style_translation.families import FAMILIES, FAMILY, CONVENTIONS

OUT_DIR = STYLE_TRANSLATION_DATA / "spanish"
URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_PATH = Path.home() / ".openrouter_key"

TOPICS = [
    "mantener una bicicleta", "organizar un mercado de agricultores", "aprender a tejer una bufanda",
    "preparar una mudanza", "cuidar un huerto urbano", "planificar un viaje en tren",
    "hacer pan en casa", "adoptar un perro", "montar una estantería", "ahorrar para las vacaciones",
    "aprender un idioma nuevo", "correr una carrera popular", "restaurar un mueble antiguo",
    "organizar una fiesta de cumpleaños", "cuidar plantas de interior", "elegir un colchón",
    "reducir el desperdicio de comida", "aprender a nadar de adulto", "pintar una habitación",
    "empezar a meditar", "preparar una entrevista de trabajo", "hacer una hoguera segura",
    "montar un acuario", "cambiar una rueda pinchada", "visitar un museo con niños",
    "organizar el armario", "hacer conservas de tomate", "aprender fotografía",
    "cuidar de un gato mayor", "preparar un picnic", "arreglar un grifo que gotea",
    "escribir un diario", "empezar a tocar la guitarra", "cultivar hierbas aromáticas",
    "hacer senderismo en la montaña", "reciclar en casa", "elegir unas gafas nuevas",
    "aprender a patinar", "organizar un club de lectura", "hacer compost", "cuidar la espalda en la oficina",
    "preparar una cena para invitados", "limpiar una bicicleta después de la lluvia", "elegir una mochila",
    "aprender a coser un botón", "montar una tienda de campaña", "cocinar arroz perfecto",
    "planificar un jardín pequeño", "hacer ejercicio en casa", "visitar a los abuelos",
    "organizar las fotos digitales", "preparar el coche para el invierno", "hacer una tarta de manzana",
    "aprender ajedrez", "cuidar un bonsái", "elegir un libro para regalar", "hacer voluntariado en el barrio",
    "aprender a hacer malabares", "planificar la semana de comidas", "cuidar las plantas en verano",
    "hacer un presupuesto familiar", "elegir un instrumento para un niño", "preparar un maratón",
    "reparar una cremallera", "hacer yogur casero", "instalar una lámpara", "aprender a bailar salsa",
    "recorrer una ciudad en bicicleta", "organizar un torneo de fútbol amateur", "elegir pintura para la fachada",
    "cuidar de una tortuga", "aprender caligrafía", "preparar una barbacoa", "cambiar el aceite de la moto",
    "montar un puzle grande", "hacer velas en casa", "cuidar un jardín de rosas", "aprender a esquiar",
    "elegir un regalo de boda", "organizar una excursión escolar", "hacer mermelada de fresa",
    "aprender a hacer croquetas", "planificar un fin de semana en la costa", "limpiar el horno",
    "empezar una colección de sellos", "elegir zapatillas para correr", "hacer un mapa del barrio",
    "aprender a silbar", "cuidar un olivo en maceta", "preparar un examen de conducir",
]
ANGLES = [
    ("a", "an informative explanation of the topic"),
    ("b", "a first-person anecdote about the topic"),
    ("c", "practical advice for someone starting out"),
]

PROMPT = """Write ONE Spanish paragraph — {angle} — about: {topic}.

{conventions}

Family requirement (this is the point of the text): {instruction}
The first requirement item must NOT appear within the first 8 words of the paragraph.
Otherwise write plain, natural Spanish; do not add unusual typography or numbers that are not required.
Return only the paragraph."""


def load_key():
    for line in KEY_PATH.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if "OPENROUTER" in line.upper() and "=" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
        if line and "=" not in line and not line.startswith("#"):
            return line
    raise RuntimeError(f"no key in {KEY_PATH}")


# ---- deterministic convention post-edit + audit ---------------------------------------------
_CARD = {"dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
         "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
         "dieciséis": 16, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20}
_ORD = [("primer", 1), ("primero", 1), ("primera", 1), ("segundo", 2), ("segunda", 2), ("tercer", 3),
        ("tercero", 3), ("tercera", 3), ("cuarto", 4), ("cuarta", 4), ("quinto", 5), ("quinta", 5),
        ("sexto", 6), ("sexta", 6), ("séptimo", 7), ("séptima", 7), ("octavo", 8), ("octava", 8),
        ("noveno", 9), ("novena", 9), ("décimo", 10), ("décima", 10)]
_CARD_RX = re.compile(r"\b(" + "|".join(_CARD) + r")\b", re.I)
_ORD_RX = re.compile(r"\b(" + "|".join(w for w, _ in _ORD) + r")\b", re.I)
_ORD_N = dict(_ORD)
# plural ordinal words -> RAE digit plurals (1.os / 1.as). "segundos" is excluded: in running
# prose it almost always means seconds of time, not "second ones".
_ORD_PL = {w + "s": n for w, n in _ORD if w.endswith(("o", "a")) and w != "segundo"}
_ORD_PL_RX = re.compile(r"\b(" + "|".join(_ORD_PL) + r")\b", re.I)


def fixup(t):
    t = t.replace("\r", "").replace(" ", " ").strip()
    t = re.sub(r"^\s*(?:Párrafo|Paragraph)\s*:\s*", "", t)
    t = re.sub(r"</?\w+[^>]*>", "", t)              # stray HTML tags
    t = t.replace("**", "").replace("__", "")     # markdown bold
    t = re.sub(r"(?<!\w)\*(\S[^*]*?)\*(?!\w)", r"\1", t)   # markdown italics
    t = t.replace("“", "«").replace("”", "»").replace("’", "'").replace("‘", "'")
    # straight double quotes -> « » by pairing
    parts = t.split('"')
    if len(parts) > 1:
        t = "".join(p + ("«" if i % 2 == 0 else "»") for i, p in enumerate(parts[:-1])) + parts[-1]
    t = t.replace("«  ", "«").replace("« ", "«").replace(" »", "»").replace("  »", "»")
    t = t.replace("…", "...").replace(". . .", "...")
    t = re.sub(r"\.{4,}", "...", t)
    t = t.replace("–", "—")
    t = re.sub(r"\s—\s", " —", t)            # spaced raya after a word: attach to the aside
    # RAE: an aside that ends the sentence keeps its closing raya before the period (—inciso—.)
    t = re.sub(r"(?<=\s)—([^—.]{2,}?)\.(?=\s|$)", r"—\1—.", t)   # opening raya = preceded by a space
    # Spanish puts the period/comma OUTSIDE the closing quote: «...». not «....»
    t = re.sub(r"(?<!\.)([.,])»", r"»\1", t)   # but an ellipsis inside the quote stays («pero...»)
    t = re.sub(r"(\d)\.(?:er|o)\b", r"\1.º", t)     # one ordinal form: 3.er / 3.o -> 3.º
    t = re.sub(r"(\d)\s*%", r"\1 %", t)
    t = re.sub(r"(\d)\s+por ciento\b", r"\1 %", t, flags=re.I)
    t = _CARD_RX.sub(lambda m: str(_CARD[m.group(1).lower()]), t)
    t = _ORD_RX.sub(lambda m: f"{_ORD_N[m.group(1).lower()]}.{'ª' if m.group(1).lower().endswith('a') else 'º'}", t)
    t = _ORD_PL_RX.sub(lambda m: f"{_ORD_PL[m.group(1).lower()]}.{'as' if m.group(1).lower().endswith('as') else 'os'}", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\s*\n\s*", " ", t)          # one paragraph
    return t.strip()


_SENT = re.compile(r"[^.!?]+[.!?]+")


def violations(t):
    v = []
    if re.search(r'["“”]', t): v.append("straight/curly double quotes")
    if "…" in t: v.append("single-glyph ellipsis")
    if re.search(r"\s-\s|–", t): v.append("spaced hyphen/en dash")
    if re.search(r" {2,}", t): v.append("double space")
    if re.search(r"\d%", t): v.append("N% without space")
    if re.search(r"\bpor ciento\b", t, re.I): v.append("por ciento")
    if _CARD_RX.search(t): v.append("cardinal in words")
    if _ORD_RX.search(t): v.append("ordinal in words")
    if _ORD_PL_RX.search(t): v.append("plural ordinal in words")
    if re.search(r"\d\.(?:er|o|a)\b(?!s)", t): v.append("ordinal form not .º/.ª")
    for s in _SENT.findall(t):
        letters = [c for c in s if c.isalpha()]
        if len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.8:
            v.append("all-caps sentence"); break
    if "\n" in t: v.append("multiple paragraphs")
    if re.search(r"(?<!\.)[.,]»", t): v.append("punctuation inside closing quote")
    if re.search(r"(?<=\s)—[^—.]{2,}\.(?=\s|$)", t): v.append("sentence-final aside without closing raya")
    if re.search(r"</?\w+>|\*\*|(?<!\w)\*\S", t): v.append("markup")
    return v


def annotate(rec):
    fam = FAMILY[rec["family"]]
    t = rec["text_es"]
    rec["words"] = len(t.split())
    rec["sentences"] = len(_SENT.findall(t))
    rec["regex_k"] = fam.regex_k(t)
    rec["violations"] = violations(t)
    return rec


def gen_one(key, model, fam, topic, angle, doc_id):
    body = {"model": model, "temperature": 1.0, "max_tokens": 700,
            "messages": [{"role": "user", "content": PROMPT.format(
                angle=angle[1], topic=topic, conventions=CONVENTIONS, instruction=fam.instruction)}]}
    for attempt in range(5):
        try:
            r = requests.post(URL, json=body, timeout=120, headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            txt = fixup(r.json()["choices"][0]["message"]["content"])
            if len(txt.split()) < 60:
                raise ValueError(f"too short ({len(txt.split())} words)")
            return annotate({"doc_id": doc_id, "family": fam.name, "topic": topic, "angle": angle[0],
                             "text_es": txt, "verify": None, "pass": None})
        except Exception as e:
            if attempt == 4:
                print(f"{doc_id} FAILED: {e}", flush=True)
                return None
            time.sleep(3 * (attempt + 1))


def load_family(name):
    f = OUT_DIR / f"{name}.json"
    return json.load(open(f)) if f.exists() else []


def save_family(name, recs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(sorted(recs, key=lambda r: r["doc_id"]), open(OUT_DIR / f"{name}.json", "w"),
              indent=0, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--n_topics", type=int, default=80, help="topics per family (x3 angles)")
    ap.add_argument("--topic_offset", type=int, default=0, help="start topic index (regeneration rounds)")
    ap.add_argument("--angles", default="abc")
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--audit_only", action="store_true")
    args = ap.parse_args()

    data = {n: load_family(n) for n in args.families}
    if args.audit_only:
        for n, recs in data.items():
            for r in recs:
                r["text_es"] = fixup(r["text_es"]); annotate(r)
            save_family(n, recs)
        report(data); return

    key = load_key()
    todo = []
    for n in args.families:
        have = {r["doc_id"] for r in data[n]}
        for ti in range(args.topic_offset, args.topic_offset + args.n_topics):
            topic = TOPICS[ti % len(TOPICS)]
            for a in ANGLES:
                if a[0] not in args.angles:
                    continue
                doc_id = f"{n}__t{ti:03d}{a[0]}"
                if doc_id not in have:
                    todo.append((FAMILY[n], topic, a, doc_id))
    print(f"{len(todo)} texts to generate across {len(args.families)} families with {args.model}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, key, args.model, f, t, a, d): f.name for f, t, a, d in todo}
        for fu in as_completed(futs):
            rec = fu.result(); done += 1
            if rec:
                data[rec["family"]].append(rec)
            if done % 100 == 0:
                for n in args.families:
                    save_family(n, data[n])
                print(f"{done}/{len(todo)}", flush=True)
    for n in args.families:
        save_family(n, data[n])
    report(data)


def report(data):
    print(f"\n{'family':14s} {'n':>4s} {'regex_k>=5':>10s} {'clean':>6s} {'words med':>9s} {'sent med':>8s}")
    for n, recs in data.items():
        if not recs:
            print(f"{n:14s}    0"); continue
        ws = sorted(r["words"] for r in recs); ss = sorted(r["sentences"] for r in recs)
        print(f"{n:14s} {len(recs):4d} {sum(r['regex_k'] >= 5 for r in recs):10d} "
              f"{sum(not r['violations'] for r in recs):6d} {ws[len(ws)//2]:9d} {ss[len(ss)//2]:8d}")


if __name__ == "__main__":
    main()
