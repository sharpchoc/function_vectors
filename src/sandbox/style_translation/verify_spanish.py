#!/usr/bin/env python
"""Verify every Spanish candidate with Claude Haiku 4.5 (OpenRouter), ONE text per call.

Why per-text calls: sub-agents handed 276 texts fell back to writing heuristic scripts instead
of reading (empty notes, an "English-word detector" flagging 87 % of oxford_comma texts as
non-fluent, hallucinated number-convention violations). One text per call with a strict JSON
schema makes every verdict a genuine reading. Conventions and length are NOT judged here — the
deterministic audit in gen_spanish.py is the authority for those.

Per text the judge returns: coherent (bool), fluent (bool), anchors (list of the exact Spanish
words/constructions where the English translation must choose between the family's two styles),
notes. k_found = len(anchors). Written into spanish/<family>.json as `verify` (+ `pass` recomputed
by collect_verdicts.py). Resumable: texts with a verdict are skipped unless --redo.
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
from src.sandbox.style_translation.families import FAMILIES, FAMILY
from src.sandbox.style_translation.gen_spanish import load_key, URL, load_family, save_family

COUNTING_RULES = {
    "contractions": ("Count EVERY finite clause whose most natural English rendering contains a contractible "
                     "auxiliary/copula/modal: negations with no/tampoco/nunca + verb (no es -> isn't, no está -> isn't, "
                     "no puede -> can't, no hay -> there isn't, no tiene -> doesn't have, no funcionan -> don't work, "
                     "no dejes -> don't, no será -> won't), «es» with a pronoun/demonstrative subject or «la verdad es que» "
                     "(it's / that's), and first/second person estoy/estás/soy/eres (I'm / you're). Each clause counts once."),
    "num_words": ("Count every cardinal number from 2 to 20 written as digits and used as a plain count or quantity "
                  "(3 días, 12 personas). Do NOT count 1, numbers above 20, percentages, ordinals (2.ª), times of day, "
                  "decimals, or the second number of a range (cada 2 o 3 meses counts as 2 anchors: 2 and 3)."),
    "ordinal_words": "Count every ordinal from 1.º to 10.º written as digits with a marker (1.º, 2.ª, 3.er, 4.º ...). Each occurrence counts.",
    "percent_sign": "Count every percentage written as digits followed by %. Each occurrence counts.",
    "sentence_caps": "Anchors are the first word of each sentence; k_found = number of sentences (6-8 expected).",
    "all_caps": "Anchors are the sentences themselves; k_found = number of sentences.",
    "double_space": "Anchors are the sentence boundaries (a sentence-final period followed by another sentence); k_found = number of such boundaries.",
    "us_uk": "Count each Spanish word whose standard English equivalent has an American/British spelling difference (color->color/colour, centro->center/centre, vecino->neighbor/neighbour, favorito->favorite/favourite, teatro, litro, metro, gris->gray/grey, catálogo, defensa, comportamiento->behavior/behaviour, sabor->flavor/flavour, honor, humor, labor, viajó->traveled/travelled, etiquetado, cancelado->canceled/cancelled, joyería->jewelry/jewellery, aluminio, pijama->pajamas/pyjamas, artefacto, puerto->harbor/harbour ...). Each occurrence counts.",
    "ise_ize": "Count each Spanish verb or noun whose standard English equivalent ends in -ize/-ise, -yze/-yse or -ization/-isation (organizar->organize/organise, reconocer->recognize, darse cuenta->realize, disculparse->apologize, criticar, enfatizar, resumir->summarize, minimizar, maximizar, priorizar, especializarse, caracterizar, categorizar, estandarizar, utilizar, memorizar, finalizar, generalizar, analizar->analyze/analyse, paralizar). Each occurrence counts.",
    "brit_t_past": "Count each past-tense verb whose English past tense has an -ed/-t alternative: aprender (learned/learnt), deletrear (spelled/spelt), quemar (burned/burnt), soñar (dreamed/dreamt), saltar (leaped/leapt), apoyarse/inclinarse (leaned/leant), derramar (spilled/spilt), estropearse/echarse a perder (spoiled/spoilt). Only PAST forms count (aprendió, aprendí, quemó, se derramó ...), not infinitives or present tense.",
    "whilst": "Count each «mientras» used as a conjunction (while/whilst), each «entre» meaning among (among/amongst), and each «en medio de» (amid/amidst). Do not count «entre» meaning between two things.",
    "ampersand": "Count each pair of two nouns joined by «y» that English would render as 'X and Y' (pan y queso, padres y profesores). Do not count «y» joining clauses or adjectives, and do not count the final «y» of a longer list.",
    "oxford_comma": "Count each list of THREE OR MORE items whose last two are joined by y/o/e/u (A, B y C). Two-item pairs do not count.",
    "curly_quotes": "Count each quoted span in « ». Each opening-closing pair counts once.",
    "quote_punct": "Count each quotation in « » that is immediately followed by a period or comma (so English must place the punctuation inside or outside the closing quote).",
    "em_dash": "Count each aside set off with rayas (—inciso—), including an aside closed by the sentence-final period.",
    "ellipsis": "Count each three-dot ellipsis (...) that is followed by more text. A final ellipsis that ends the paragraph does not count.",
    # lexically diverse families (2026-09-11)
    "uk_vocab": "Count each Spanish word for an object whose standard English name differs between American and British English (camión->truck/lorry, vacaciones->vacation/holiday, basura->trash/rubbish, gasolina->gasoline/petrol, linterna->flashlight/torch, teléfono móvil->cell phone/mobile phone, acera->sidewalk/pavement, galleta->cookie/biscuit, jersey->sweater/jumper, pantalones->pants/trousers, pañal->diaper/nappy, armario->closet/wardrobe, berenjena->eggplant/aubergine, calabacín->zucchini/courgette, cilantro->cilantro/coriander, matemáticas->math/maths, autopista->highway/motorway, aparcamiento->parking lot/car park, gasolinera->gas station/petrol station, carrito->shopping cart/trolley, cochecito->stroller/pushchair, cuna->crib/cot, parabrisas->windshield/windscreen, llave inglesa->wrench/spanner, guardabarros->fender/mudguard, sentido antihorario->counterclockwise/anticlockwise, gachas->oatmeal/porridge, comida para llevar->takeout/takeaway, código postal->zip code/postcode, ferrocarril->railroad/railway, matrícula->license plate/number plate, buzón->mailbox/postbox, cartero->mailman/postman, chaleco->vest/waistcoat, grifo->faucet/tap, farmacia->drugstore/chemist's, piso->apartment/flat, ascensor->elevator/lift). Each occurrence counts.",
    "register": "Count each Spanish word whose standard English equivalent has a plain and a formal variant (empezar/comenzar->begin/commence, comprar->buy/purchase, intentar->try/attempt, vivir->live/reside, revisar/comprobar->check/verify, arreglar/reparar->fix/repair, niños->kids/children, mostrar/enseñar->show/demonstrate, más tarde/luego->later/subsequently, también->also/additionally, quizá/tal vez->maybe/perhaps, grande->big/large, averiguar/descubrir->find out/discover, investigar->look into/investigate, tirar/desechar->throw away/discard, resolver/determinar->figure out/determine, participar->take part/participate, deshacerse de/eliminar->get rid of/eliminate, buscar->look for/seek). Each occurrence counts.",
    "unit_abbr": "Count each number followed by a metric unit symbol (5 km, 2 kg, 30 ml, 20 cm, 3 mm, 250 g, 15 °C). Each occurrence counts; a unit written as a word (kilómetros) does not count.",
    "diacritics": "Count each Spanish word whose standard English equivalent is a loanword that can be written with or without its accent (café->cafe/café, ingenuo->naive/naïve, cliché, prometido->fiance/fiancé, fachada->facade/façade, jalapeño, decoración->decor/décor, saltear->saute/sauté, puré->puree/purée, crema->creme/crème, plato principal->entree/entrée, matiné->matinee/matinée, debut/estreno->debut/début, élite, señor/señora kept in English, piñata, pupilo/protegido->protege/protégé, déjà vu, a la carta->a la carte/à la carte, velada->soiree/soirée, canapé, paté->pate/pâté, macramé, aplique->applique/appliqué, guardería->creche/crèche, papel maché, ingenuidad->naivete/naïveté). Each occurrence counts.",
    "latin_abbr": "Count each connective that English can write as a phrase or a Latin abbreviation: «por ejemplo» (for example/e.g.), «es decir» (that is/i.e.), «etcétera»/«y así sucesivamente» (and so on/etc.), «frente a»/«contra»/«versus» comparing two options (versus/vs.), «aproximadamente»/«unos»/«unas» before a number (approximately/approx.). Each occurrence counts.",
    "hyphen_compound": "Count each Spanish term whose standard English equivalent is a compound word that can be written closed or hyphenated (correo electrónico->email/e-mail, en línea->online/on-line, sitio web->website, bienestar->wellbeing, cooperar->cooperate, reutilizar->reuse, casero->homemade, sin parar->nonstop, estilo de vida->lifestyle, entrenamiento->workout, chequeo->checkup, maquillaje->makeup, configuración->setup, copia de seguridad->backup, empresa emergente->startup, coordinar->coordinate, precalentar->preheat, multitarea->multitask, teléfono inteligente->smartphone, trabajo en equipo->teamwork, durante el día->daytime, durante la noche->nighttime, fin de semana->weekend, protector solar->sunscreen, cepillo de dientes->toothbrush, hecho a mano->handmade, durante la noche->overnight, cronología->timeline, nombre de usuario->username, inicio de sesión->login, sin conexión->offline, microondas->microwave, agua de lluvia->rainwater, leña->firewood, invernadero->greenhouse, cortacésped->lawnmower, carretilla->wheelbarrow, manillar->handlebars, pata de cabra->kickstand, faro delantero->headlight, luz trasera->taillight, tija->seatpost, mochila->backpack). Each occurrence counts.",
    "flat_adverb": "Count each manner adverb that directly follows a verb (or a verb plus a short object) and that English renders with an -ly adverb which also has a colloquial flat form: despacio/lentamente (slowly/slow), rápido/rápidamente (quickly/quick), suavemente/con suavidad (gently/gentle), con fuerza/firmemente (tightly, firmly / tight, firm), en voz alta (loudly/loud), con seguridad (safely/safe), hondo/profundamente (deeply/deep), barato (cheaply/cheap), directamente (directly/direct), de otra manera (differently/different), mal (wrongly/wrong), bajito/en voz baja (softly/soft), uniformemente (evenly/even), con cuidado (carefully/careful), en silencio (quietly/quiet), de cerca (closely/close), con constancia (steadily/steady), claramente (clearly/clear), fácilmente (easily/easy), brillante (brightly/bright), ligeramente (lightly/light), fino/en rodajas finas (thinly/thin), finamente (finely/fine), recién (freshly/fresh), correctamente (properly/proper). Each occurrence counts.",
    "irreg_past": "Count each preterite verb whose English past tense has an irregular and a regular form: zambullirse/lanzarse al agua/bucear (dove/dived), colarse/escabullirse/entrar a hurtadillas (snuck/sneaked), encender (lit/lighted), suplicar/rogar (pled/pleaded), acelerar/ir a toda velocidad (sped/speeded), tejer (wove/weaved), brillar (shone/shined), esforzarse (strove/strived), arrodillarse (knelt/kneeled). Only past-tense uses count.",
    "latin_plural": "Count each PLURAL noun whose English plural has an anglicised and a classical form (índices->indexes/indices, fórmulas->formulas/formulae, cactus->cactuses/cacti, apéndices->appendixes/appendices, planes de estudio/currículos->curriculums/curricula, estadios->stadiums/stadia, foros->forums/fora, antenas->antennas/antennae, hongos->funguses/fungi, temarios->syllabuses/syllabi, pulpos->octopuses/octopi, radios->radiuses/radii, núcleos->nucleuses/nuclei, memorandos->memorandums/memoranda, referéndums->referendums/referenda, milenios->millenniums/millennia, acuarios->aquariums/aquaria, terrarios->terrariums/terraria, vértices->vertexes/vertices, matrices->matrixes/matrices, larvas->larvas/larvae, nebulosas->nebulas/nebulae, vértebras->vertebras/vertebrae, tesauros->thesauruses/thesauri, gimnasios->gymnasiums/gymnasia, simposios->symposiums/symposia, hipopótamos->hippopotamuses/hippopotami, antiguos alumnos->alumnuses/alumni). Singular uses do not count.",
    "title_abbr": "Count each title or street word followed by a proper name: doctor/doctora X (Doctor/Dr.), profesor/profesora X (Professor/Prof.), señor X (Mister/Mr.), monte X (Mount/Mt.), san/santa X (Saint/St.), calle X (Street/St.), avenida X (Avenue/Ave.), carretera/camino X (Road/Rd.), bulevar X (Boulevard/Blvd.). Each occurrence counts; the words without a name do not count.",
}

PROMPT = """You are verifying ONE Spanish source text for a style-translation study. Family: {name}.
When this Spanish is translated into English, the translator must choose between two styles:
  NAT = {nat}
  ALT = {alt}
Decision point = {opportunity}.

Judge exactly three things and answer in JSON only:
1. "coherent": true if the paragraph flows and makes sense as a single piece of prose (no
   contradictions, no truncation, not a list of disconnected sentences). IGNORE length, sentence
   count, and typography — they are checked elsewhere.
2. "fluent": true if it is natural, grammatical Spanish. Minor stiffness is fine. False only for
   real errors: broken syntax, English or machine-like wording, wrong agreement, nonsense words.
3. "anchors": the exact Spanish words/constructions where the English translation faces the
   NAT/ALT choice. Counting rule for this family: {rule}
   List each occurrence separately, in text order. Do not invent anchors that are not in the text.
4. "notes": one short sentence, or "".

TEXT:
{text}

Reply with a single JSON object with keys "coherent", "fluent", "anchors", "notes". No markdown."""


def judge_one(key, model, fam, rec):
    body = {"model": model, "temperature": 0.0, "max_tokens": 600,
            "messages": [{"role": "user", "content": PROMPT.format(
                name=fam.name, nat=fam.nat, alt=fam.alt, opportunity=fam.opportunity,
                rule=COUNTING_RULES[fam.name], text=rec["text_es"])}]}
    for attempt in range(5):
        try:
            r = requests.post(URL, json=body, timeout=90, headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            start = raw.index("{")
            v, _ = json.JSONDecoder(strict=False).raw_decode(raw[start:])   # tolerate trailing text / raw newlines
            anchors = [str(a) for a in v.get("anchors", [])]
            return rec["doc_id"], {"coherent": bool(v.get("coherent")), "fluent": bool(v.get("fluent")),
                                   "k_found": len(anchors), "anchors": anchors,
                                   "notes": str(v.get("notes", ""))[:300], "judge": model}
        except Exception as e:
            if attempt == 4:
                print(f"{rec['doc_id']} JUDGE FAILED: {e}", flush=True)
                return rec["doc_id"], None
            time.sleep(2 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=[f.name for f in FAMILIES])
    ap.add_argument("--model", default="anthropic/claude-haiku-4.5")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--redo", action="store_true", help="re-judge texts that already have a verdict")
    ap.add_argument("--limit", type=int, default=None, help="max texts per family (calibration runs)")
    args = ap.parse_args()
    key = load_key()
    data = {n: load_family(n) for n in args.families}
    jobs = []
    for n, recs in data.items():
        todo = [r for r in recs if args.redo or not r.get("verify")]
        if args.limit:
            todo = todo[:args.limit]
        jobs += [(FAMILY[n], r) for r in todo]
    print(f"{len(jobs)} texts to judge with {args.model}", flush=True)
    idx = {(r["family"], r["doc_id"]): r for recs in data.values() for r in recs}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, key, args.model, f, r): (f.name, r["doc_id"]) for f, r in jobs}
        for fu in as_completed(futs):
            doc_id, v = fu.result(); fam = futs[fu][0]
            if v is not None:
                idx[(fam, doc_id)]["verify"] = v
            done += 1
            if done % 500 == 0:
                for n in args.families:
                    save_family(n, data[n])
                print(f"{done}/{len(jobs)}", flush=True)
    for n in args.families:
        save_family(n, data[n])
    print(f"\n{'family':14s} {'judged':>6s} {'coh&flu':>7s} {'k>=5':>5s} {'coh&flu&k>=5':>12s}")
    for n, recs in data.items():
        ver = [r for r in recs if r.get("verify")]
        cf = [r for r in ver if r["verify"]["coherent"] and r["verify"]["fluent"]]
        k5 = [r for r in ver if r["verify"]["k_found"] >= 5]
        print(f"{n:14s} {len(ver):6d} {len(cf):7d} {len(k5):5d} {sum(1 for r in cf if r['verify']['k_found'] >= 5):12d}")


if __name__ == "__main__":
    main()
