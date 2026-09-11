#!/usr/bin/env python
"""Step 2a — translate every Spanish source text into ONE house-style English paragraph
(Gemini 2.5 Flash via OpenRouter, all families in one parallel pool), then normalise it
deterministically to the house style and audit it.

HOUSE STYLE (user-approved plan 2026-09-07) = the nat pole of all 17 families = standard American
English: sentence-initial capitals, standard case, US spelling, -ize, -ed pasts, while/among/amid,
single space after periods, serial comma, straight quotes, attached em dash, three-dot ellipsis,
period/comma inside the closing quote, digits 2-20, N%, digit ordinals (3rd), contracted forms,
the word "and". Enforced with the registry renderers (properties.py) family by family, so the
nat twin of EVERY family is exactly this style; the alt twin (collect_english.py) flips one family.

Output: dataset_files/style_translation/english/<family>.json
  [{doc_id, family, text_es, anchors_es, text_en_raw, text_nat, k_en, audit_nat:[...],
    rounds, verify_en: null|{...}, pass: null|bool}]
Resumable per doc_id; `--retry_failed` re-translates records with pass == False (feedback appended).
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
from src.sandbox.ext_styleprops.properties import PROPS, render
from src.sandbox.style_translation.families import FAMILIES, FAMILY
from src.sandbox.style_translation.gen_spanish import load_key, URL

FINAL_ES = STYLE_TRANSLATION_DATA / "final"
OUT_DIR = STYLE_TRANSLATION_DATA / "english"

# order matters: quotes before quote punctuation; typography before lexical; numbers last
NORMALISE_ORDER = ["curly_quotes", "quote_punct", "em_dash", "ellipsis", "double_space",
                   "sentence_caps", "all_caps", "us_uk", "ise_ize", "brit_t_past", "whilst",
                   "contractions", "ampersand", "oxford_comma", "num_words", "percent_sign",
                   "ordinal_words",
                   # lexically diverse families (2026-09-11): compounds before vocabulary, units after numbers
                   "hyphen_compound", "uk_vocab", "register", "diacritics", "latin_abbr", "flat_adverb",
                   "irreg_past", "latin_plural", "title_abbr", "unit_abbr"]
assert set(NORMALISE_ORDER) == set(PROPS)

HOUSE_STYLE = """House style — standard American English. Apply ALL of these, every time:
- Standard sentence capitalization and standard case (never all caps).
- American spelling (color, center, favorite, neighbor, traveled, gray, catalog, defense, theater, liter, meter, behavior, flavor, honor, humor, labor, harbor, jewelry, aluminum, pajamas, artifact, labeled, canceled).
- -ize/-yze spellings (organize, realize, recognize, apologize, criticize, emphasize, summarize, minimize, prioritize, utilize, analyze).
- -ed past forms: learned, spelled, burned, dreamed, leaped, leaned, spilled, spoiled (never learnt, spelt...).
- while, among, amid (never whilst, amongst, amidst).
- Contracted forms where a contraction is possible: don't, isn't, it's, there's, I'm, they're, can't, won't.
- One space after every period. Serial (Oxford) comma in lists: "bread, cheese, and wine".
- Straight double quotes "..." for quotations, with the period or comma INSIDE the closing quote: "like this," she said. Render the Spanish « » as straight double quotes.
- Asides with an em dash attached on both sides: word—aside—word (never a spaced hyphen " - ").
- Ellipsis as three dots (...).
- Numbers 2-20 as digits (3 days, 12 people); percentages as digits with the sign and no space (15%); ordinals as digits (1st, 2nd, 3rd, 10th); the word "and" (never &).
- American vocabulary: truck, vacation, trash, gasoline, flashlight, cell phone, sidewalk, cookie, sweater, pants, diaper, closet, eggplant, zucchini, cilantro, math, highway, parking lot, gas station, shopping cart, stroller, crib, windshield, wrench, fender, counterclockwise, oatmeal, takeout, zip code, railroad, license plate, mailbox, mailman, vest, faucet, drugstore, apartment, elevator (never lorry, holiday, rubbish, petrol, torch, mobile phone, pavement, biscuit, jumper, trousers, nappy, wardrobe, aubergine, courgette, coriander, maths, motorway, car park, trolley, pushchair, cot, spanner, flat, lift).
- Plain everyday words: begin, buy, try, live, check, fix, kids, show, later, also, maybe, big, find out, look into, throw away, figure out, take part, get rid of, look for (never commence, purchase, attempt, reside, verify, repair, children, demonstrate, subsequently, additionally, perhaps, large, discover, investigate, discard, determine, participate, eliminate, seek).
- Metric units as symbols after a space: 5 km, 2 kg, 30 ml, 20 cm, 3 mm, 250 g, 15 °C (never kilometers, kilograms...).
- Loanwords without accents: cafe, naive, cliche, fiance, facade, jalapeno, decor, saute, puree, creme, entree, matinee, debut, elite, senor, pinata, protege, deja vu, a la carte, soiree, canape, pate.
- English phrases, not Latin abbreviations: for example, that is, and so on, versus, approximately (never e.g., i.e., etc., vs., approx.).
- Closed compounds: email, online, website, wellbeing, cooperate, reuse, homemade, nonstop, lifestyle, workout, checkup, makeup, setup, backup, startup, coordinate, preheat, multitask, smartphone, teamwork, daytime, nighttime, weekend, sunscreen, toothbrush, handmade, overnight, timeline, username, login, offline, microwave, rainwater, firewood, greenhouse, lawnmower, wheelbarrow, handlebars, kickstand, headlight, taillight, seatpost, backpack (never e-mail, on-line, well-being, co-operate...).
- -ly adverbs after verbs: drive slowly, hold tightly, brake gently, breathe deeply, speak quietly (never drive slow, hold tight).
- Irregular American past forms: dove, snuck, lit, pled, sped, wove, shone, strove, knelt (never dived, sneaked, lighted, pleaded, speeded, weaved, shined, strived, kneeled).
- Anglicised plurals: indexes, formulas, cactuses, appendixes, curriculums, stadiums, forums, antennas, syllabuses, octopuses, radiuses, memorandums, referendums, millenniums, aquariums, terrariums, vertexes, matrixes, nebulas (never indices, formulae, cacti, appendices...).
- Titles and street words in full: Doctor Smith, Professor Lee, Mister Brown, Mount Fuji, Saint Paul, Main Street, Fifth Avenue, Oak Road, Sunset Boulevard (never Dr., Prof., Mr., Mt., St., Ave., Rd., Blvd.)."""

PROMPT = """Translate the following Spanish paragraph into natural, fluent English.

Rules:
- Translate faithfully, sentence by sentence, in the same order; do not merge, split, add, or drop sentences or content.
- Keep every quotation, aside, list, number, percentage, and ordinal of the source.
- Use the natural everyday English equivalent of each word (color for color, neighbor for vecino, organize for organizar, learned for aprendió, while for mientras, and so on) rather than a rarer synonym.
{house_style}
{feedback}
Spanish paragraph:
{text}

Return only the English paragraph."""


def load_docs(fam):
    return json.load(open(FINAL_ES / f"{fam}.json"))


def load_out(fam):
    f = OUT_DIR / f"{fam}.json"
    return json.load(open(f)) if f.exists() else []


def save_out(fam, recs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(sorted(recs, key=lambda r: r["doc_id"]), open(OUT_DIR / f"{fam}.json", "w"),
              indent=0, ensure_ascii=False)


# ---- deterministic house-style normalisation --------------------------------------------------
def pre_fix(t):
    t = t.replace("\r", "").replace(" ", " ").strip()
    t = re.sub(r"</?\w+[^>]*>", "", t)
    t = t.replace("**", "").replace("__", "")
    t = re.sub(r"^\s*(?:English|Translation)\s*:\s*", "", t, flags=re.I)
    t = t.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')
    t = t.replace("’", "'").replace("‘", "'")
    t = t.replace("…", "...").replace(". . .", "...")
    t = re.sub(r"\.{4,}", "...", t)
    t = re.sub(r"\s*[–]\s*", "—", t)
    t = re.sub(r"(\d)\s+%", r"\1%", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\s*\n\s*", " ", t)
    return t.strip()


def normalise_nat(t):
    """Force every detected manifestation of every family to its nat rendering."""
    t = pre_fix(t)
    for fam in NORMALISE_ORDER:
        opps = PROPS[fam].find_opps(t)
        if opps:
            t, _ = render(t, opps, "nat")
    return t


def audit_nat(t):
    """Manifestations still not in nat form (any family) + residual typography problems."""
    bad = []
    for fam in PROPS:
        for o in PROPS[fam].find_opps(t):
            if t[o.start:o.end] != o.nat:
                bad.append(f"{fam}:{t[o.start:o.end]!r}")
    if re.search(r"[“”«»…–]", t): bad.append("typography:curly/angular/ellipsis/en-dash")
    if re.search(r" {2,}", t): bad.append("typography:double space")
    if re.search(r"</?\w+>|\*\*", t): bad.append("markup")
    if "\n" in t: bad.append("multiple paragraphs")
    return bad


def annotate(rec):
    # Gemini renders the connective "es decir" as "that's," — a plain error; house style is "that is,"
    rec["text_en_raw"] = re.sub(r"(?<=[,;(] )that[’']s,", "that is,", rec["text_en_raw"])
    rec["text_en_raw"] = re.sub(r"(?:(?<=[.!?] )|^)That[’']s,", "That is,", rec["text_en_raw"])
    rec["text_nat"] = normalise_nat(rec["text_en_raw"])
    rec["k_en"] = len(PROPS[rec["family"]].find_opps(rec["text_nat"]))
    rec["audit_nat"] = audit_nat(rec["text_nat"])
    rec["words_en"] = len(rec["text_nat"].split())
    return rec


def translate_one(key, model, fam, doc, feedback=""):
    body = {"model": model, "temperature": 0.3, "max_tokens": 900,
            "messages": [{"role": "user", "content": PROMPT.format(
                house_style=HOUSE_STYLE, feedback=feedback, text=doc["text_es"])}]}
    for attempt in range(5):
        try:
            r = requests.post(URL, json=body, timeout=120, headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
            if len(raw.split()) < 50:
                raise ValueError(f"too short ({len(raw.split())} words)")
            return raw
        except Exception as e:
            if attempt == 4:
                print(f"{doc['doc_id']} FAILED: {e}", flush=True)
                return None
            time.sleep(3 * (attempt + 1))


def feedback_for(rec):
    fam = FAMILY[rec["family"]]
    notes = []
    if rec.get("k_en", 0) < 5:
        notes.append(f"A previous translation kept only {rec.get('k_en', 0)} of the source's "
                     f"{len(rec['anchors_es'])} decision points for the style family "
                     f"'{fam.name}' ({fam.nat} vs {fam.alt}). These Spanish anchors MUST each be "
                     f"rendered with their standard English equivalent so the choice remains: "
                     + "; ".join(rec["anchors_es"][:9]) + ".")
    HINTS = {
        "oxford_comma": "Write every list of three or more items so that the last two items are SINGLE words, e.g. 'large, flat, and resistant' or 'bread, cheese, and wine' (move modifiers elsewhere if needed), always with the serial comma.",
        "double_space": "Keep every sentence of the source as a separate English sentence (at least 6 sentences), each ending with a period and the next one starting with a capitalized word (not a digit or a quotation mark).",
        "quote_punct": "Every quotation of the source must be kept in double quotes and be immediately followed by a comma or a period placed INSIDE the closing quote, e.g. \"like this,\" or \"like this.\"",
        "ellipsis": "Keep every three-dot ellipsis (...) of the source at the same place, with text continuing after it.",
        "brit_t_past": "Use exactly these verbs for the corresponding Spanish ones: aprendió -> learned, deletreó -> spelled, quemó -> burned, soñó -> dreamed, saltó -> leaped, se apoyó/se inclinó -> leaned, derramó -> spilled, se estropeó/se echó a perder -> spoiled.",
        "us_uk": "Use exactly the American spelling of the standard equivalent (color, favorite, neighbor, center, theater, liter, meter, gray, catalog, defense, behavior, flavor, honor, humor, labor, harbor, traveled, labeled, canceled, jewelry, aluminum, pajamas, artifact).",
        "ise_ize": "Use the -ize verb for each Spanish one (organizar -> organize, reconocer -> recognize, darse cuenta -> realize, disculparse -> apologize, criticar -> criticize, enfatizar -> emphasize, resumir -> summarize, minimizar -> minimize, priorizar -> prioritize, utilizar -> utilize, analizar -> analyze).",
        "whilst": "Translate mientras as 'while', entre (among) as 'among', en medio de as 'amid'.",
        "ampersand": "Keep every two-noun pair joined with the word 'and'.",
        "contractions": "Use a contraction in every clause where English allows one (isn't, don't, can't, it's, there's, I'm, you're, won't).",
        # lexically diverse families (2026-09-11)
        "uk_vocab": "Name each object with its everyday American word (camión -> truck, vacaciones -> vacation, basura -> trash, gasolina -> gasoline, linterna -> flashlight, teléfono móvil -> cell phone, acera -> sidewalk, galleta -> cookie, jersey -> sweater, pantalones -> pants, pañal -> diaper, armario -> closet, berenjena -> eggplant, calabacín -> zucchini, matemáticas -> math, autopista -> highway, aparcamiento -> parking lot, gasolinera -> gas station, carrito -> shopping cart, cochecito -> stroller, cuna -> crib, parabrisas -> windshield, llave inglesa -> wrench, guardabarros -> fender, gachas -> oatmeal, comida para llevar -> takeout, código postal -> zip code, ferrocarril -> railroad, matrícula -> license plate, buzón -> mailbox, cartero -> mailman, chaleco -> vest, grifo -> faucet, farmacia -> drugstore, piso -> apartment, ascensor -> elevator).",
        "register": "Use the plain everyday word for each of these (empezar -> begin, comprar -> buy, intentar -> try, vivir -> live, revisar -> check, arreglar -> fix, niños -> kids, mostrar -> show, más tarde -> later, también -> also, quizá -> maybe, grande -> big, averiguar -> find out, investigar -> look into, tirar -> throw away, resolver -> figure out, participar -> take part, deshacerse de -> get rid of, buscar -> look for).",
        "unit_abbr": "Keep every measurement as digits plus the unit SYMBOL with a space: 5 km, 2 kg, 30 ml, 20 cm, 3 mm, 250 g, 15 °C.",
        "diacritics": "Translate each loanword source word with the English loanword WITHOUT accents (café -> cafe, ingenuo -> naive, cliché -> cliche, prometido -> fiance, fachada -> facade, jalapeño -> jalapeno, decoración -> decor, saltear -> saute, puré -> puree, plato principal -> entree, matiné -> matinee, debut -> debut, élite -> elite, señor -> senor, piñata -> pinata, pupilo -> protege, velada -> soiree, canapé -> canape, paté -> pate, macramé -> macrame, guardería -> creche, papel maché -> papier-mache).",
        "latin_abbr": "Translate EVERY one of these literally, keeping it as a connective: por ejemplo -> 'for example,' ; es decir -> ', that is,' (with commas, never 'that\'s' or 'meaning'); etcétera / y así sucesivamente -> 'and so on' ; frente a / contra (comparing options) -> 'versus' ; aproximadamente / unos N / unas N -> 'approximately N' (never 'about', 'around', 'some'). Never use e.g., i.e., etc., vs., approx.",
        "hyphen_compound": "Use the closed compound for each of these (correo electrónico -> email, en línea -> online, sitio web -> website, bienestar -> wellbeing, cooperar -> cooperate, reutilizar -> reuse, casero -> homemade, sin parar -> nonstop, estilo de vida -> lifestyle, entrenamiento -> workout, chequeo -> checkup, maquillaje -> makeup, configuración -> setup, copia de seguridad -> backup, empresa emergente -> startup, coordinar -> coordinate, precalentar -> preheat, multitarea -> multitask, teléfono inteligente -> smartphone, trabajo en equipo -> teamwork, fin de semana -> weekend, protector solar -> sunscreen, cepillo de dientes -> toothbrush, hecho a mano -> handmade, cronología -> timeline, nombre de usuario -> username, inicio de sesión -> login, sin conexión -> offline, microondas -> microwave, agua de lluvia -> rainwater, leña -> firewood, invernadero -> greenhouse, cortacésped -> lawnmower, carretilla -> wheelbarrow, manillar -> handlebars, pata de cabra -> kickstand, faro delantero -> headlight, luz trasera -> taillight, tija -> seatpost, mochila -> backpack).",
        "flat_adverb": "Put every manner adverb AFTER its verb (and after a short object if there is one), never before it: 'pedal slowly', 'hold it tightly', 'breathe deeply', 'speak softly', 'go directly to', 'inflate them carefully' (NOT 'carefully inflate', NOT 'clearly explaining', NOT 'was easily obtained'). Use these adverbs: despacio -> slowly, rápido -> quickly, suavemente -> gently, con fuerza -> tightly, firmemente -> firmly, en voz alta -> loudly, con seguridad -> safely, hondo -> deeply, barato -> cheaply, directamente -> directly, mal -> wrongly, bajito -> softly, uniformemente -> evenly, con cuidado -> carefully, en silencio -> quietly, de cerca -> closely, claramente -> clearly, fácilmente -> easily, fino -> thinly, recién -> freshly, correctamente -> properly.",
        "irreg_past": "Use exactly these simple-past forms, affirmative (never a negation or a synonym): se zambulló/se lanzó al agua/buceó -> dove, se coló/se escabulló/entró a hurtadillas -> snuck, encendió/prendió -> lit (never 'turned on', 'switched on'), suplicó/rogó/imploró -> pled (never 'begged'), aceleró/salió a toda velocidad -> sped (or sped up/off), tejió -> wove, brilló/resplandeció -> shone (never 'glowed', 'sparkled'), se esforzó -> strove, se arrodilló -> knelt.",
        "latin_plural": "Use the anglicised plural: índices -> indexes, fórmulas -> formulas, cactus -> cactuses, apéndices -> appendixes, planes de estudio -> curriculums, estadios -> stadiums, foros -> forums, antenas -> antennas, temarios -> syllabuses, pulpos -> octopuses, radios -> radiuses, memorandos -> memorandums, referéndums -> referendums, milenios -> millenniums, acuarios -> aquariums, terrarios -> terrariums, vértices -> vertexes, matrices -> matrixes, nebulosas -> nebulas, tesauros -> thesauruses, gimnasios -> gymnasiums, simposios -> symposiums, hipopótamos -> hippopotamuses.",
        "title_abbr": "Write every title and street word in full before/after the name: doctor X -> Doctor X, profesor X -> Professor X, señor X -> Mister X, monte X -> Mount X, san/santa X -> Saint X, calle X -> X Street, avenida X -> X Avenue, carretera X -> X Road, bulevar X -> X Boulevard (never Dr., Prof., Mr., Mt., St., Ave., Rd., Blvd.).",
    }
    if rec.get("k_en", 0) < 5 and fam.name in HINTS:
        notes.append(HINTS[fam.name])
    v = rec.get("verify_en") or {}
    if v and not v.get("faithful", True):
        notes.append("A previous translation was judged unfaithful: " + str(v.get("notes", ""))[:200])
    if v and not (v.get("coherent", True) and v.get("fluent", True)):
        notes.append("A previous translation was judged not fluent/coherent: " + str(v.get("notes", ""))[:200])
    if rec.get("audit_nat"):
        notes.append("Do not use these forms: " + ", ".join(rec["audit_nat"][:6]))
    return ("\nFeedback on the previous attempt (fix it):\n- " + "\n- ".join(notes) + "\n") if notes else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="docs per family (trial runs)")
    ap.add_argument("--retry_failed", action="store_true", help="re-translate records with pass == False")
    ap.add_argument("--audit_only", action="store_true")
    ap.add_argument("--fill", type=int, default=0,
                    help="add up to N spare Spanish texts (pass=True, not yet translated) per family that has < 200 usable translations")
    args = ap.parse_args()

    out = {n: load_out(n) for n in args.families}
    if args.audit_only:
        for n, recs in out.items():
            for r in recs:
                annotate(r)
            save_out(n, recs)
        report(out); return

    key = load_key()
    jobs = []
    for n in args.families:
        have = {r["doc_id"]: r for r in out[n]}
        for d in load_docs(n)[: args.limit]:
            rec = have.get(d["doc_id"])
            if rec is None:
                rec = {"doc_id": d["doc_id"], "family": n, "text_es": d["text_es"],
                       "anchors_es": d["verify"]["anchors"], "rounds": 0, "verify_en": None, "pass": None}
                out[n].append(rec); have[d["doc_id"]] = rec
                jobs.append((n, rec, ""))
            elif args.retry_failed and rec["rounds"] < 6 and (rec.get("pass") is False or rec.get("k_en", 99) < 5 or rec.get("audit_nat")):
                jobs.append((n, rec, feedback_for(rec)))
    if args.fill:
        SP_DIR = STYLE_TRANSLATION_DATA / "spanish"
        for n in args.families:
            have = {r["doc_id"] for r in out[n]}
            usable = sum(1 for r in out[n] if r.get("k_en", 0) >= 5 and not r.get("audit_nat") and r.get("pass") is not False)
            need = max(0, 200 - usable)
            if need == 0:
                continue
            spares = [d for d in json.load(open(SP_DIR / f"{n}.json")) if d.get("pass") and d["doc_id"] not in have]
            take = spares[: min(len(spares), need + args.fill)]
            for d in take:
                rec = {"doc_id": d["doc_id"], "family": n, "text_es": d["text_es"],
                       "anchors_es": d["verify"]["anchors"], "rounds": 0, "verify_en": None, "pass": None,
                       "spare": True}
                out[n].append(rec); jobs.append((n, rec, ""))
            print(f"{n}: usable={usable} need={need} -> adding {len(take)} spare Spanish texts (of {len(spares)} available)", flush=True)
    print(f"{len(jobs)} translations to run with {args.model}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(translate_one, key, args.model, n, rec, fb): (n, rec) for n, rec, fb in jobs}
        for fu in as_completed(futs):
            n, rec = futs[fu]; raw = fu.result(); done += 1
            if raw:
                rec.update(text_en_raw=raw, rounds=rec["rounds"] + 1, verify_en=None, **{"pass": None})
                annotate(rec)
            if done % 500 == 0:
                for m in args.families:
                    save_out(m, out[m])
                print(f"{done}/{len(jobs)}", flush=True)
    for n in args.families:
        save_out(n, out[n])
    report(out)


def report(out):
    print(f"\n{'family':14s} {'n':>4s} {'k_en>=5':>7s} {'nat-clean':>9s} {'k_en med':>8s} {'words med':>9s}")
    for n, recs in out.items():
        recs = [r for r in recs if "text_nat" in r]
        if not recs:
            print(f"{n:14s}    0"); continue
        ks = sorted(r["k_en"] for r in recs); ws = sorted(r["words_en"] for r in recs)
        print(f"{n:14s} {len(recs):4d} {sum(r['k_en'] >= 5 for r in recs):7d} "
              f"{sum(not r['audit_nat'] for r in recs):9d} {ks[len(ks)//2]:8d} {ws[len(ws)//2]:9d}")


if __name__ == "__main__":
    main()
