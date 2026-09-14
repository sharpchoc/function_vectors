"""Multilingual convention families (user decision 2026-09-13): the TARGET text is not English.
Source = English (generated as a translation of the target text); target = Portuguese / Spanish / French / German
twins that differ only in an orthographic or regional convention. Natural pole = house style = the currently
standard / most common form; alternative = the other variety or the pre-/post-reform form.

Each family gets a registry Property (lexicon toggle, nat = first form) registered into
ext_styleprops.properties.PROPS so scoring.decide / cue_tokens / build_prompts / rollout / analyze work unchanged.
Prompt header for these families: "English:\\n{source}\\n\\n{Target language}:\\n{twin}".
"""
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_BOOT = Path(__file__).resolve().parents[3]
for p in (_BOOT, _BOOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from src.sandbox.ext_styleprops.properties import _lexicon_property, PROPS, Opp, _dedup


@dataclass
class MLFamily:
    name: str
    tgt_code: str
    tgt_lang: str            # header word, e.g. "Portuguese"
    variety_nat: str         # for the generator: which orthography/variety to write in
    nat: str                 # legend label
    alt: str
    pairs: list
    instruction: str         # family requirement for the generator (English)
    verify_extra: str = ""   # extra check for the verifier
    judge_ignore: str = ""
    prop: object = None
    src_lang: str = "English"      # source language of the prompt (Spanish for English-target families)
    domain: str = "text"           # "text" (writing conventions) or "code" (code_families.py)

    @property
    def nat_label(self): return self.nat
    @property
    def alt_label(self): return self.alt

    def regex_k(self, text):
        return len(self.prop.find_opps(text))

    def header(self, source):
        return f"{self.src_lang}:\n{source}\n\n{self.tgt_lang}:\n"


def _mk(fam):
    L = _lexicon_property(f"ML_{fam.name}", fam.name, fam.pairs, fam.nat, fam.alt)
    fam.prop = L()
    PROPS[fam.name] = fam.prop
    return fam


# ---------------------------------------------------------------------------------------------------------------
# Portuguese: Brazilian (nat) vs European (alt) — lexical + orthographic (circumflex/acute, mute c)
PT_BR_EU = [
    ("ônibus", "autocarro"), ("trem", "comboio"), ("celular", "telemóvel"), ("sorvete", "gelado"), ("banheiro", "casa de banho"),
    ("geladeira", "frigorífico"), ("café da manhã", "pequeno-almoço"), ("esporte", "desporto"), ("esportes", "desportos"),
    ("calçada", "passeio"), ("xícara", "chávena"), ("suco", "sumo"), ("abacaxi", "ananás"), ("tela", "ecrã"), ("arquivo", "ficheiro"),
    ("arquivos", "ficheiros"), ("caminhão", "camião"), ("açougue", "talho"), ("bala", "rebuçado"), ("ponto de ônibus", "paragem"),
    ("faixa de pedestres", "passadeira"), ("estacionamento", "parque de estacionamento"), ("carteira de motorista", "carta de condução"),
    ("garçom", "empregado de mesa"), ("cardápio", "ementa"), ("aluguel", "aluguer"), ("tênis", "ténis"), ("econômico", "económico"),
    ("econômica", "económica"), ("fenômeno", "fenómeno"), ("gênero", "género"), ("bônus", "bónus"), ("cômodo", "cómodo"),
    ("acadêmico", "académico"), ("acadêmica", "académica"), ("quilômetro", "quilómetro"), ("quilômetros", "quilómetros"),
    ("ingênuo", "ingénuo"), ("fato", "facto"), ("fatos", "factos"), ("contato", "contacto"), ("registro", "registo"), ("equipe", "equipa"),
    ("umidade", "humidade"), ("planejar", "planear"), ("planejamento", "planeamento"), ("gerenciar", "gerir"), ("gerente", "gerente"),
    ("bilhete", "bilhete"), ("ônibus", "autocarro"), ("time", "equipa"), ("torcedor", "adepto"), ("goleiro", "guarda-redes"),
    ("grama", "relva"), ("sanduíche", "sandes"), ("presunto", "fiambre"), ("terno", "fato"), ("meia", "peúga"), ("controle", "controlo"),
    ("gerência", "gerência"), ("telefone celular", "telemóvel"), ("fila", "bicha"), ("caneta", "esferográfica"), ("secretária eletrônica", "atendedor de chamadas"),
    ("eletrônico", "eletrónico"), ("eletrônica", "eletrónica"), ("polêmica", "polémica"), ("Antônio", "António"), ("cênico", "cénico"),
    ("premiê", "primeiro-ministro"), ("apartamento", "apartamento"), ("perua", "carrinha"), ("conserto", "reparação"), ("mamadeira", "biberão"),
    ("chiclete", "pastilha elástica"), ("pedágio", "portagem"), ("esparadrapo", "adesivo"), ("privada", "sanita"), ("cadarço", "atacador"),
    ("ímã", "íman"), ("quadrinhos", "banda desenhada"), ("aeromoça", "hospedeira"), ("açúcar", "açúcar"),
]
_PT_DROP = {"bala", "meia", "fila", "grama", "tela", "time", "privada", "perua", "gerente", "bilhete", "apartamento", "açúcar", "gerência", "caneta",
            "banheiro", "geladeira", "ponto de ônibus", "calçada", "cardápio", "terno", "sanduíche", "conserto", "mamadeira", "chiclete", "pedágio",
            "quadrinhos", "secretária eletrônica", "premiê", "telefone celular", "esparadrapo"}   # gender-changing swaps / collisions removed 2026-09-13
PT_BR_EU = [(a, b) for a, b in PT_BR_EU if a != b and a not in _PT_DROP]

# European Portuguese: post-1990 Acordo (nat) vs pre-1990 (alt) — mute consonants dropped
PT_ACORDO_EU = [
    ("ação", "acção"), ("ações", "acções"), ("ato", "acto"), ("atos", "actos"), ("ator", "actor"), ("atores", "actores"), ("atriz", "actriz"),
    ("atual", "actual"), ("atuais", "actuais"), ("atualmente", "actualmente"), ("atualidade", "actualidade"), ("atualizar", "actualizar"),
    ("atividade", "actividade"), ("atividades", "actividades"), ("ativo", "activo"), ("ativa", "activa"), ("ativos", "activos"),
    ("adoção", "adopção"), ("adotar", "adoptar"), ("adotado", "adoptado"), ("afeto", "afecto"), ("batismo", "baptismo"),
    ("coleção", "colecção"), ("coleções", "colecções"), ("coletivo", "colectivo"), ("coletiva", "colectiva"), ("correto", "correcto"),
    ("correta", "correcta"), ("corretamente", "correctamente"), ("correção", "correcção"), ("direção", "direcção"), ("direto", "directo"),
    ("direta", "directa"), ("diretamente", "directamente"), ("diretor", "director"), ("diretora", "directora"), ("elétrico", "eléctrico"),
    ("elétrica", "eléctrica"), ("elétricos", "eléctricos"), ("exato", "exacto"), ("exata", "exacta"), ("exatamente", "exactamente"),
    ("exceção", "excepção"), ("exceções", "excepções"), ("fator", "factor"), ("fatores", "factores"), ("ótimo", "óptimo"), ("ótima", "óptima"),
    ("ótimos", "óptimos"), ("ótimas", "óptimas"), ("objetivo", "objectivo"), ("objetivos", "objectivos"), ("objeto", "objecto"),
    ("objetos", "objectos"), ("projeto", "projecto"), ("projetos", "projectos"), ("projetar", "projectar"), ("proteção", "protecção"),
    ("proteger", "proteger"), ("redação", "redacção"), ("reação", "reacção"), ("seleção", "selecção"), ("selecionar", "seleccionar"),
    ("setor", "sector"), ("setores", "sectores"), ("teto", "tecto"), ("efetivo", "efectivo"), ("efetivamente", "efectivamente"),
    ("efetuar", "efectuar"), ("espetáculo", "espectáculo"), ("respetivo", "respectivo"), ("respetiva", "respectiva"), ("aspeto", "aspecto"),
    ("aspetos", "aspectos"), ("Egito", "Egipto"), ("ótica", "óptica"), ("característica", "característica"), ("perfeito", "perfeito"),
    ("inspeção", "inspecção"), ("infeção", "infecção"), ("receção", "recepção"), ("perspetiva", "perspectiva"), ("eletricidade", "electricidade"),
    ("eletrónico", "electrónico"), ("eletrónica", "electrónica"), ("tática", "táctica"), ("didático", "didáctico"), ("prático", "prático"),
    ("lecionar", "leccionar"), ("dialeto", "dialecto"), ("arquiteto", "arquitecto"), ("arquitetura", "arquitectura"), ("detetar", "detectar"),
    ("detetive", "detective"), ("adjetivo", "adjectivo"), ("subtil", "subtil"), ("otimizar", "optimizar"), ("ótimas", "óptimas"),
]
PT_ACORDO_EU = [(a, b) for a, b in PT_ACORDO_EU if a != b]

# Brazilian Portuguese: post-1990 (nat) vs pre-1990 (alt) — trema, hiatus accents, double-vowel circumflex, hyphens
PT_ACORDO_BR = [
    ("ideia", "idéia"), ("ideias", "idéias"), ("assembleia", "assembléia"), ("europeia", "européia"), ("europeias", "européias"),
    ("plateia", "platéia"), ("joia", "jóia"), ("joias", "jóias"), ("heroico", "heróico"), ("heroica", "heróica"), ("paranoico", "paranóico"),
    ("geleia", "geléia"), ("colmeia", "colméia"), ("estreia", "estréia"), ("estreias", "estréias"), ("epopeia", "epopéia"), ("boia", "bóia"),
    ("jiboia", "jibóia"), ("coreia", "coréia"), ("Coreia", "Coréia"), ("voo", "vôo"), ("voos", "vôos"), ("enjoo", "enjôo"), ("zoo", "zôo"),
    ("perdoo", "perdôo"), ("abençoo", "abençôo"), ("veem", "vêem"), ("leem", "lêem"), ("creem", "crêem"), ("deem", "dêem"),
    ("frequente", "freqüente"), ("frequentes", "freqüentes"), ("frequência", "freqüência"), ("frequentemente", "freqüentemente"),
    ("frequentar", "freqüentar"), ("consequência", "conseqüência"), ("consequências", "conseqüências"), ("consequentemente", "conseqüentemente"),
    ("sequência", "seqüência"), ("sequências", "seqüências"), ("tranquilo", "tranqüilo"), ("tranquila", "tranqüila"), ("tranquilos", "tranqüilos"),
    ("tranquilamente", "tranqüilamente"), ("tranquilidade", "tranqüilidade"), ("cinquenta", "cinqüenta"), ("linguiça", "lingüiça"),
    ("aguentar", "agüentar"), ("aguenta", "agüenta"), ("eloquente", "eloqüente"), ("delinquente", "delinqüente"), ("pinguim", "pingüim"),
    ("pinguins", "pingüins"), ("bilíngue", "bilíngüe"), ("questão", "questão"), ("antissocial", "anti-social"), ("autoestima", "auto-estima"),
    ("infraestrutura", "infra-estrutura"), ("ultrassom", "ultra-som"), ("autorretrato", "auto-retrato"), ("semiárido", "semi-árido"),
    ("extraoficial", "extra-oficial"), ("minissaia", "mini-saia"), ("micro-ondas", "microondas"), ("antirrugas", "anti-rugas"),
    ("autoescola", "auto-escola"), ("autoajuda", "auto-ajuda"), ("contrarregra", "contra-regra"), ("suprassumo", "supra-sumo"),
    ("ultrapassar", "ultrapassar"), ("socioeconômico", "sócio-econômico"), ("agroindústria", "agro-indústria"), ("infravermelho", "infra-vermelho"),
    ("para-brisa", "pára-brisa"), ("para-choque", "pára-choque"), ("para-raios", "pára-raios"), ("para-quedas", "pára-quedas"),
    ("apoio", "apóio"), ("apoia", "apóia"), ("averigue", "averigúe"), ("argui", "argúi"), ("feiura", "feiúra"), ("baiuca", "baiúca"),
]
PT_ACORDO_BR = [(a, b) for a, b in PT_ACORDO_BR if a != b]

# Spanish: post-2010 RAE (nat) vs pre-2010 (alt) — accents on solo, demonstrative pronouns, guion, truhan, monosyllables
ES_RAE2010 = [
    ("solo", "sólo"), ("este", "éste"), ("esta", "ésta"), ("estos", "éstos"), ("estas", "éstas"), ("ese", "ése"), ("esa", "ésa"),
    ("esos", "ésos"), ("esas", "ésas"), ("aquel", "aquél"), ("aquella", "aquélla"), ("aquellos", "aquéllos"), ("aquellas", "aquéllas"),
    ("guion", "guión"), ("guiones", "guiones"), ("truhan", "truhán"), ("fie", "fié"), ("fio", "fió"), ("hui", "huí"), ("huis", "huís"),
    ("riais", "riáis"), ("crie", "crié"), ("crio", "crió"), ("lie", "lié"), ("lio", "lió"), ("Sion", "Sión"), ("ion", "ión"), ("prion", "prión"),
]
ES_RAE2010 = [(a, b) for a, b in ES_RAE2010 if a != b]
_ES_DEM = {"este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "aquel", "aquella", "aquellos", "aquellas"}
_ES_PRON_NEXT = re.compile(r"\s*(?:[,.;:!?)»]|$|\b(?:es|era|fue|será|sería|son|eran|fueron|serán|que|de|del|no|se|me|te|le|lo|la|los|las|nos|"
                           r"sí|también|tampoco|ya|y|o|pero|con|sin|para|por|en|a|al|como|más|menos|siempre|nunca|puede|pueden|tiene|tienen|"
                           r"tenía|resulta|parece|funciona|suele|suelen|debe|deben|hay|último|última|últimos|últimas|sigue|va|van|viene|vienen|"
                           r"fue|ha|han|había|está|están|estaba|estaban|permite|ofrece|requiere|exige|ayuda|hace|hacen|da|dan|sirve|resultó|"
                           r"salió|quedó|quedaba|resultaba|era)\b)", re.I)
_ES_SOLO_ADJ_NEXT = re.compile(r"\s+(?:[,.;:!?)»]|en\b|con\b|sin\b|a\b|de\b|para\b|por\b|y\b|o\b|pero\b|que\b|hasta\b|desde\b|durante\b|el\b|la\b|los\b|las\b|un\b|una\b|unos\b|unas\b|al\b|del\b|hay\b|se\b|me\b|te\b|le\b|lo\b|nos\b|si\b|cuando\b|porque\b|después\b|antes\b|así\b|entonces\b|tras\b|\d)", re.I)


def _mk_es(fam):
    L = _lexicon_property("ML_es_rae2010_base", fam.name, fam.pairs, fam.nat, fam.alt)

    class EsRae(L):
        def find_opps(self, text):
            out = []
            for o in L.find_opps(self, text):
                w = o.nat.lower()
                after = text[o.end:]
                if w in _ES_DEM and not _ES_PRON_NEXT.match(after):
                    continue          # demonstrative adjective (este libro): no accent in either era
                if w == "solo":
                    before = text[max(0, o.start - 12):o.start]
                    if re.search(r"\b(?:el|un|mi|tu|su|estoy|está|estaba|estar|sentirse|siento|sentí|quedó|quedé|vive|vivía|come|comía|trabaja|trabajaba)\s*$", before, re.I) and not _ES_SOLO_ADJ_NEXT.match(after):
                        continue      # likely adjective (un hombre solo / está solo)
                out.append(o)
            return _dedup(out)

    EsRae.__name__ = "ML_es_rae2010"
    fam.prop = EsRae()
    PROPS[fam.name] = fam.prop
    return fam


# French: traditional (nat) vs 1990 rectifications (alt) — circumflex on i/u, ognon, évènement, compound numbers
FR_1990 = [
    ("paraître", "paraitre"), ("paraît", "parait"), ("apparaître", "apparaitre"), ("apparaît", "apparait"), ("disparaître", "disparaitre"),
    ("disparaît", "disparait"), ("connaître", "connaitre"), ("connaît", "connait"), ("reconnaître", "reconnaitre"), ("reconnaît", "reconnait"),
    ("naître", "naitre"), ("naît", "nait"), ("maître", "maitre"), ("maîtres", "maitres"), ("maîtriser", "maitriser"), ("maîtrise", "maitrise"),
    ("plaît", "plait"), ("coût", "cout"), ("coûts", "couts"), ("coûter", "couter"), ("coûte", "coute"), ("coûtent", "coutent"), ("coûteux", "couteux"),
    ("coûteuse", "couteuse"), ("goût", "gout"), ("goûts", "gouts"), ("goûter", "gouter"), ("goûte", "goute"), ("août", "aout"), ("sûrement", "surement"),
    ("brûler", "bruler"), ("brûle", "brule"), ("brûlé", "brulé"), ("brûlure", "brulure"), ("flûte", "flute"), ("piqûre", "piqure"), ("entraîner", "entrainer"),
    ("entraîne", "entraine"), ("entraînement", "entrainement"), ("traîner", "trainer"), ("traîne", "traine"), ("fraîche", "fraiche"), ("fraîches", "fraiches"),
    ("fraîcheur", "fraicheur"), ("dîner", "diner"), ("dîne", "dine"), ("île", "ile"), ("îles", "iles"), ("boîte", "boite"), ("boîtes", "boites"),
    ("croûte", "croute"), ("dégoûtant", "dégoutant"), ("abîmer", "abimer"), ("abîmé", "abimé"), ("huître", "huitre"), ("huîtres", "huitres"),
    ("chaîne", "chaine"), ("chaînes", "chaines"), ("s'il vous plaît", "s'il vous plait"), ("oignon", "ognon"), ("oignons", "ognons"), ("nénuphar", "nénufar"),
    ("événement", "évènement"), ("événements", "évènements"), ("réglementaire", "règlementaire"), ("réglementation", "règlementation"), ("week-end", "weekend"),
    ("week-ends", "weekends"), ("porte-monnaie", "portemonnaie"), ("mille-pattes", "millepatte"), ("vingt et un", "vingt-et-un"), ("trente et un", "trente-et-un"),
    ("cent vingt", "cent-vingt"), ("deux cents", "deux-cents"), ("trois cents", "trois-cents"), ("deux mille", "deux-mille"), ("bûche", "buche"),
    ("bûcheron", "bucheron"), ("voûte", "voute"), ("croûton", "crouton"), ("soûl", "soul"), ("dîners", "diners"), ("gîte", "gite"), ("presqu'île", "presqu'ile"),
    ("s'entraîner", "s'entrainer"), ("accroître", "accroitre"), ("croître", "croitre"), ("décroître", "décroitre"), ("cloître", "cloitre"), ("épître", "épitre"),
    ("traître", "traitre"), ("aîné", "ainé"), ("aînée", "ainée"), ("chaînette", "chainette"), ("déchaîner", "déchainer"), ("enchaîner", "enchainer"),
    ("affût", "affut"), ("ragoût", "ragout"), ("assidûment", "assidument"), ("continûment", "continument"), ("mûrir", "murir"), ("mûre", "mure"), ("mûres", "mures"),
]
FR_1990 = [(a, b) for a, b in FR_1990 if a != b]

# German: 1996 reform (nat) vs pre-reform (alt) — ß/ss after short vowels, daß, triple consonants, misc
DE_1996 = [
    ("dass", "daß"), ("muss", "muß"), ("musst", "mußt"), ("müsst", "müßt"), ("lässt", "läßt"), ("bisschen", "bißchen"), ("Fluss", "Fluß"),
    ("Schluss", "Schluß"), ("Prozess", "Prozeß"), ("Kongress", "Kongreß"), ("Kompromiss", "Kompromiß"), ("Kuss", "Kuß"), ("Schuss", "Schuß"),
    ("Anschluss", "Anschluß"), ("Genuss", "Genuß"), ("Einfluss", "Einfluß"), ("Abschluss", "Abschluß"), ("Verschluss", "Verschluß"),
    ("Ausschuss", "Ausschuß"), ("Fass", "Faß"), ("Pass", "Paß"), ("nass", "naß"), ("gewusst", "gewußt"), ("wusste", "wußte"), ("wussten", "wußten"),
    ("passt", "paßt"), ("isst", "ißt"), ("vergisst", "vergißt"), ("Riss", "Riß"), ("Biss", "Biß"), ("Imbiss", "Imbiß"), ("Missverständnis", "Mißverständnis"),
    ("misst", "mißt"), ("Stängel", "Stengel"), ("rau", "rauh"), ("raue", "rauhe"), ("rauen", "rauhen"), ("Schifffahrt", "Schiffahrt"), ("nummerieren", "numerieren"),
    ("nummeriert", "numeriert"), ("platzieren", "plazieren"), ("platziert", "plaziert"), ("Tipp", "Tip"), ("Tipps", "Tips"), ("Stopp", "Stop"),
    ("Känguru", "Känguruh"), ("Delfin", "Delphin"), ("Delfine", "Delphine"), ("Rad fahren", "radfahren"), ("im Allgemeinen", "im allgemeinen"),
    ("heute Abend", "heute abend"), ("heute Morgen", "heute morgen"), ("gestern Abend", "gestern abend"), ("morgen Abend", "morgen abend"),
    ("Leid tun", "leid tun"), ("Recht haben", "recht haben"), ("Schnee fahren", "schneefahren"), ("Zuhause", "zu Hause"), ("aufwändig", "aufwendig"),
    ("Albtraum", "Alptraum"), ("Gämse", "Gemse"), ("Quäntchen", "Quentchen"), ("belämmert", "belemmert"), ("Tollpatsch", "Tolpatsch"),
    ("Rohheit", "Roheit"), ("Zähheit", "Zäheit"), ("Stuckateur", "Stukkateur"), ("Ass", "As"), ("Karamell", "Karamel"), ("Mopp", "Mop"),
    ("Messestand", "Meßstand"), ("Fitness", "Fitneß"), ("Stress", "Streß"), ("gestresst", "gestreßt"), ("Boss", "Boß"), ("Erdgeschoss", "Erdgeschoß"),
    ("Geschoss", "Geschoß"), ("Schloss", "Schloß"), ("Schlosses", "Schlosses"), ("essbar", "eßbar"), ("Essstäbchen", "Eßstäbchen"), ("Nussschale", "Nußschale"),
    ("Nuss", "Nuß"), ("Nüsse", "Nüsse"), ("Kuss", "Kuß"), ("küsst", "küßt"), ("Verdruss", "Verdruß"), ("Überdruss", "Überdruß"), ("Erlass", "Erlaß"),
    ("Anlass", "Anlaß"), ("Zusammenhang", "Zusammenhang"), ("bewusst", "bewußt"), ("Bewusstsein", "Bewußtsein"), ("selbstbewusst", "selbstbewußt"),
    ("Schlussfolgerung", "Schlußfolgerung"), ("Beschluss", "Beschluß"), ("Entschluss", "Entschluß"), ("Ausschluss", "Ausschluß"), ("Zuschuss", "Zuschuß"),
    ("Überschuss", "Überschuß"), ("Vorschuss", "Vorschuß"), ("Grussformel", "Grußformel"), ("Gruss", "Gruß"),
]
DE_1996 = [(a, b) for a, b in DE_1996 if a != b and a not in ("Zuhause", "Gruss", "Grussformel", "Schnee fahren", "Nüsse", "Schlosses", "Messestand", "Zusammenhang")]


import difflib
from src.sandbox.ext_styleprops.properties import Property


def _substr_property(cls_name, prop_name, pairs, nat_lab, alt_lab):
    """Lexicon toggle WITHOUT word boundaries (Japanese / Chinese have no spaces). Longest form first."""
    table = {}
    for a, b in pairs:
        table[a] = (a, b); table[b] = (a, b)
    rx = re.compile("(" + "|".join(sorted(map(re.escape, table), key=len, reverse=True)) + ")")

    class S(Property):
        name = prop_name
        family = "spelling"

        def find_opps(self, text):
            return _dedup([Opp(m.start(1), m.end(1), *table[m.group(1)]) for m in rx.finditer(text)])

        def classify(self, tail):
            m = rx.search(tail)
            if m is None:
                return None
            nat, alt = table[m.group(1)]
            return "nat" if m.group(1) == nat else "alt"

    S.__name__ = cls_name; S.nat_label, S.alt_label, S.confound = nat_lab, alt_lab, "low"
    return S


def _rule_property(cls_name, prop_name, nat_char, alt_str, known_words, nat_lab, alt_lab):
    """Every word containing `nat_char` is an opportunity; alt = the word with nat_char -> alt_str (ё→е, ß→ss).
    classify: nat_char present -> nat; a known word in its alt form -> alt."""
    word_rx = re.compile(r"\w*" + nat_char + r"\w*", re.UNICODE)
    alt_forms = {w.replace(nat_char, alt_str) for w in known_words} | {w.replace(nat_char, alt_str).capitalize() for w in known_words}
    alt_rx = re.compile(r"\b(" + "|".join(sorted(map(re.escape, alt_forms), key=len, reverse=True)) + r")\b") if alt_forms else None

    class R(Property):
        name = prop_name
        family = "spelling"

        def find_opps(self, text):
            return _dedup([Opp(m.start(), m.end(), m.group(0), m.group(0).replace(nat_char, alt_str)) for m in word_rx.finditer(text)])

        def classify(self, tail):
            m_nat = re.search(nat_char, tail)
            m_alt = alt_rx.search(tail) if alt_rx else None
            if m_nat and (m_alt is None or m_nat.start() <= m_alt.start()):
                return "nat"
            if m_alt:
                return "alt"
            return None

    R.__name__ = cls_name; R.nat_label, R.alt_label, R.confound = nat_lab, alt_lab, "low"
    return R


def _zh_property(cls_name, prop_name, nat_locale, alt_locale, nat_lab, alt_lab):
    """Chinese twins by full script/regional conversion (zhconv). Opportunities = character runs that differ between
    the text in the nat locale and its conversion to the alt locale. classify: any alt-locale-only form -> alt, any
    nat-locale-only form -> nat (a segment that converts unchanged in both directions decides nothing)."""
    import zhconv

    class Z(Property):
        name = prop_name
        family = "spelling"

        @staticmethod
        def to_nat(text): return zhconv.convert(text, nat_locale)
        @staticmethod
        def to_alt(text): return zhconv.convert(text, alt_locale)

        def find_opps(self, text):
            alt = zhconv.convert(text, alt_locale)
            sm = difflib.SequenceMatcher(None, text, alt, autojunk=False)
            opps = []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal" and i2 > i1 and j2 > j1:
                    opps.append(Opp(i1, i2, text[i1:i2], alt[j1:j2]))
            return opps

        def classify(self, tail):
            if zhconv.convert(tail, nat_locale) != tail:       # contains forms specific to the alt locale
                return "alt"
            if zhconv.convert(tail, alt_locale) != tail:       # contains forms specific to the nat locale
                return "nat"
            return None

    Z.__name__ = cls_name; Z.nat_label, Z.alt_label, Z.confound = nat_lab, alt_lab, "low"
    return Z


def _mk_prop(fam, cls):
    fam.prop = cls(); PROPS[fam.name] = fam.prop; return fam


# ---- Ukrainian 2019 (nat = new forms incl. permitted variants) vs pre-2019 -----------------------------------------
UK_2019 = [
    ("проєкт", "проект"), ("проєкти", "проекти"), ("проєкту", "проекту"), ("проєкті", "проекті"), ("проєктом", "проектом"), ("проєктів", "проектів"),
    ("проєктування", "проектування"), ("авдиторія", "аудиторія"), ("авдиторії", "аудиторії"), ("авдиторію", "аудиторію"), ("авдит", "аудит"),
    ("етер", "ефір"), ("етері", "ефірі"), ("етеру", "ефіру"), ("катедра", "кафедра"), ("катедри", "кафедри"), ("катедрі", "кафедрі"), ("міт", "міф"),
    ("міти", "міфи"), ("мітологія", "міфологія"), ("Атени", "Афіни"), ("Атенах", "Афінах"), ("павза", "пауза"), ("павзи", "паузи"), ("павзу", "паузу"),
    ("лавреат", "лауреат"), ("лавреати", "лауреати"), ("фавна", "фауна"), ("фавни", "фауни"), ("індик", "індик"), ("пів години", "півгодини"),
    ("пів року", "півроку"), ("пів яблука", "пів'яблука"), ("пів аркуша", "піваркуша"), ("пів міста", "півміста"), ("пів дня", "півдня"),
    ("священник", "священик"), ("священники", "священики"), ("Дікенс", "Діккенс"), ("анатема", "анафема"), ("Марта", "Марфа"), ("дитирамб", "дифірамб"),
    ("ортопед", "ортопед"), ("Гете", "Гете"), ("архимандрит", "архімандрит"), ("проєкція", "проекція"), ("проєкції", "проекції"), ("проєктор", "проектор"),
    ("Атена", "Афіна"), ("міту", "міфу"), ("етерний", "ефірний"), ("авдієнція", "аудієнція"), ("Фавст", "Фауст"), ("пів хвилини", "півхвилини"),
    ("пів кілограма", "півкілограма"), ("пів літра", "півлітра"), ("пів метра", "півметра"), ("пів сторінки", "півсторінки"), ("пів склянки", "півсклянки"),
]
UK_2019 = [(a, b) for a, b in UK_2019 if a != b]

# ---- Japanese trailing long vowel: with ー (nat) vs without (alt) ---------------------------------------------------
JA_LONG = [("コンピューター", "コンピュータ"), ("プリンター", "プリンタ"), ("ユーザー", "ユーザ"), ("サーバー", "サーバ"), ("ブラウザー", "ブラウザ"),
           ("フォルダー", "フォルダ"), ("スピーカー", "スピーカ"), ("モニター", "モニタ"), ("プレーヤー", "プレーヤ"), ("センター", "センタ"),
           ("エレベーター", "エレベータ"), ("エスカレーター", "エスカレータ"), ("カレンダー", "カレンダ"), ("ドライバー", "ドライバ"), ("カウンター", "カウンタ"),
           ("メーカー", "メーカ"), ("リーダー", "リーダ"), ("トレーナー", "トレーナ"), ("スニーカー", "スニーカ"), ("マネージャー", "マネージャ"),
           ("プログラマー", "プログラマ"), ("デザイナー", "デザイナ"), ("エディター", "エディタ"), ("プロセッサー", "プロセッサ"), ("コントローラー", "コントローラ"),
           ("コンピューター", "コンピュータ"), ("スキャナー", "スキャナ"), ("ルーター", "ルータ"), ("アダプター", "アダプタ"), ("コネクター", "コネクタ"),
           ("パラメーター", "パラメータ"), ("インジケーター", "インジケータ"), ("セパレーター", "セパレータ"), ("フィルター", "フィルタ"), ("バッファー", "バッファ"),
           ("レジスター", "レジスタ"), ("キャラクター", "キャラクタ"), ("インストーラー", "インストーラ"), ("ヘッダー", "ヘッダ"), ("フッター", "フッタ"),
           ("シャッター", "シャッタ"), ("モーター", "モータ"), ("ジェネレーター", "ジェネレータ"), ("オペレーター", "オペレータ"), ("キャリアー", "キャリア"),
           ("ボイラー", "ボイラ"), ("ヒーター", "ヒータ"), ("クーラー", "クーラ"), ("ミキサー", "ミキサ"), ("トースター", "トースタ"), ("ハンバーガー", "ハンバーガ"),
           ("シャワー", "シャワ"), ("タワー", "タワ"), ("パワー", "パワ"), ("メンバー", "メンバ"), ("ナンバー", "ナンバ"), ("レター", "レタ"), ("バター", "バタ"),
           ("ライター", "ライタ"), ("セーター", "セータ"), ("ハンガー", "ハンガ"), ("スクーター", "スクータ"), ("トラクター", "トラクタ"), ("ヘリコプター", "ヘリコプタ")]
JA_LONG = list(dict.fromkeys(JA_LONG))

# ---- Russian ё: written (nat) vs replaced by е (alt); known words for classifying the е-form ------------------------
RU_YO_WORDS = """ещё её всё идёт даёт чёрный жёлтый зелёный ёлка вперёд приём объём учёба ребёнок весёлый тяжёлый серьёзный четвёртый пошёл нашёл
ушёл пришёл живёт берёт растёт поёт несёт ведёт зовёт чёткий щётка лёд мёд счёт полёт самолёт костёр ковёр шофёр актёр партнёр режиссёр
съёмка плёнка тётя сёстры звёзды гнёзда колёса озёра сёла слёзы зёрна чёрт вдвоём втроём ёж ёмкость надёжный дешёвый жёсткий чёрточка
ещё-бы мёртвый твёрдый лёгкий лёгкое подъём приёмник вёсла свёкла копчёный печёный варёный тушёный солёный сгущённый учёный отчёт расчёт
зачёт учёт надёжно определённый определённо совершённый решённый увлечённый заведёт заведённый заём наём наёмный поём проём приём
вернёмся вернётся начнём начнёт придём придёт пойдём пойдёт найдём найдёт возьмём возьмёт поймём поймёт живём даём несём ведём растём
зовём берём поёт поём чётко ёлочный ёрш ёкнуть щёлкать щёлкнуть шёпот шёлк шёлковый желёз черёмуха черёд очередь чёлка щёки щёлочь
уголёк пенёк денёк огонёк стёкла стёклышко тёплый тёплая тёплое тёмный тёмно жёлудь жёрнов ковёр манёвр манёвры плёс серёжка серёжки
тренажёр стажёр монтёр боксёр дирижёр гравёр жонглёр шёл шёлк пошёл ушёл нашёл пришёл вошёл вышёл зашёл обошёл перешёл подошёл прошёл
берёза берёзовый ёлочка ёлки лёжа стоём? вёз вёл нёс мёл плёл тёр пёк жёг лёг сёл ёкает ёрзать ёрничать ёженедельный
ежеднёвно зёв клёв клён клёны сёмга ёрш котёл козёл осёл орёл щегол чёрствый жёваный жёлоб чёткость ещёб посёлок новосёл сёмка
разъём приём подъёмник вертолёт самолёты полёты налёт взлёт залёт перелёт улёт лётчик лётный полётный пилотёр""".split()
RU_YO_WORDS = [w for w in RU_YO_WORDS if "ё" in w and w.isalpha()]

# ---- German Swiss: ß (nat, standard German) vs ss (alt, Swiss) --------------------------------------------------------
DE_SS_WORDS = """Straße Straßen groß große großen großer großes größer größte größten heißt heißen heiß heiße außerdem außen außer draußen
Fuß Füße Fußball Fußgänger weiß weiße weißen Spaß Maß Maßnahme Maßnahmen Grüße Gruß schließlich schließen schließt fließen fließt genießen
genießt Größe Größen bloß Soße süß süße süßen Fleiß fleißig Straßenbahn Stoß Anstoß Schoß mäßig regelmäßig regelmäßige regelmäßigen gemäß Buße
reißen reißt beißen gießen Gießkanne ließ ließen verließ saß aß fraß stieß floß Kloß Floß Grieß grüßen grüßt Fußweg Fußnote Maßstab
einigermaßen gewissermaßen dermaßen Muße Ruß ruß rußig Straßenrand Außenseite Außenbereich äußerst äußere äußeren Äußerung Schließfach
Verschleiß Verschleißteile Preußen preußisch Blöße Größenordnung großartig großzügig Großstadt Großeltern Großmutter Großvater Großteil
Weißwein Weißbrot Weißheit heißen Heißluft Fußboden Fußbremse Reißverschluss Reißzwecke Bußgeld Strauß Sträuße süßlich Süßigkeiten
Süßigkeit maßgeblich maßlos ebenmäßig gleichmäßig verhältnismäßig planmäßig zweckmäßig übermäßig unmäßig ausschließlich einschließlich
Schließung Anschließend anschließend abschließend ausgeschlossen? Genießer Geschoß Geschoss? Fließband fließend stoßen stößt Zusammenstoß
Fußspitze Fußgelenk Fußsohle Straßenverkehr Straßenlaterne Straßenname""".split()
DE_SS_WORDS = [w for w in DE_SS_WORDS if "ß" in w and w.replace("ß", "").isalpha()]


def _pairs_text(pairs, n=None):
    ps = pairs if n is None else pairs[:n]
    return ", ".join(f"{a} (not {b})" for a, b in ps)


ML_FAMILIES = [
    _mk(MLFamily("pt_br_eu", "pt", "Portuguese", "Brazilian Portuguese (Brazilian vocabulary and spelling, e.g. ônibus, econômico, fato, celular)",
                 "Brazilian Portuguese (ônibus, econômico, celular)", "European Portuguese (autocarro, económico, telemóvel)", PT_BR_EU,
                 "Write in BRAZILIAN Portuguese. Use at least 6 DIFFERENT words from this list, in exactly the Brazilian form given first: "
                 + _pairs_text(PT_BR_EU) + ". Never use the European forms in parentheses.",
                 judge_ignore="Brazilian vs European Portuguese vocabulary or spelling for the same thing")),
    _mk(MLFamily("pt_acordo_eu", "pt", "Portuguese", "European Portuguese in the post-1990 Acordo Ortográfico spelling (ação, ótimo, atual, direção — no mute consonants)",
                 "post-1990 European spelling (ação, ótimo, atual)", "pre-1990 European spelling (acção, óptimo, actual)", PT_ACORDO_EU,
                 "Write in EUROPEAN Portuguese with the post-1990 Acordo Ortográfico spelling. Use at least 6 DIFFERENT words from this list, in exactly the "
                 "reformed form given first: " + _pairs_text(PT_ACORDO_EU) + ". Never write the pre-reform forms in parentheses.",
                 judge_ignore="pre- vs post-1990 Portuguese spelling (mute consonants: acção/ação, óptimo/ótimo)")),
    _mk(MLFamily("pt_acordo_br", "pt", "Portuguese", "Brazilian Portuguese in the post-1990 Acordo Ortográfico spelling (ideia, voo, frequente, tranquilo, antissocial — no trema, no accent on ei/oi diphthongs)",
                 "post-1990 Brazilian spelling (ideia, voo, frequente)", "pre-1990 Brazilian spelling (idéia, vôo, freqüente)", PT_ACORDO_BR,
                 "Write in BRAZILIAN Portuguese with the post-1990 Acordo Ortográfico spelling. Use at least 6 DIFFERENT words from this list, in exactly the "
                 "reformed form given first: " + _pairs_text(PT_ACORDO_BR) + ". Never write the pre-reform forms in parentheses.",
                 judge_ignore="pre- vs post-1990 Brazilian spelling (trema, accents on ei/oi/oo, hyphens in compounds)")),
    _mk_es(MLFamily("es_rae2010", "es", "Spanish", "Spanish following the 2010 RAE orthography (solo and demonstrative pronouns without accent: solo, este, esta, guion)",
                 "post-2010 RAE (solo, este, guion)", "pre-2010 accents (sólo, éste, guión)", ES_RAE2010,
                 "Write in Spanish with the 2010 RAE orthography. The text must contain at least 6 occurrences in total of: the adverb «solo» meaning «solamente» "
                 "(at least 3 times, e.g. «solo hace falta», «solo dos veces») and demonstratives used as PRONOUNS standing alone, not before a noun "
                 "(at least 3 times, e.g. «este es el más fácil», «prefiero esa», «aquellos que empiezan»), and optionally «guion». Write them WITHOUT accent "
                 "(solo, este, esta, esos, aquel, guion), never sólo/éste/ésa/aquél/guión. Do NOT use «solo» as an adjective meaning alone, and do NOT place "
                 "este/esta/ese/esa/aquel before a noun anywhere in the text (write «el proceso», «dicho paso» instead).",
                 verify_extra="Also check: every «solo» is the adverb meaning «solamente» (never an adjective meaning alone), and every este/esta/estos/estas/ese/esa/esos/esas/aquel/aquella/aquellos/aquellas is a PRONOUN standing alone (never followed by a noun). Set anchors_ok=false if any occurrence violates this.",
                 judge_ignore="accents on solo / demonstrative pronouns / guion (sólo vs solo, éste vs este)")),
    _mk(MLFamily("fr_1990", "fr", "French", "French in the traditional orthography (coût, connaître, oignon, événement, vingt et un)",
                 "traditional spelling (coût, connaître, oignon)", "1990 rectifications (cout, connaitre, ognon)", FR_1990,
                 "Write in French with the TRADITIONAL orthography (circumflex on i and u kept). Use at least 6 DIFFERENT words from this list, in exactly the "
                 "traditional form given first: " + _pairs_text(FR_1990) + ". Never write the rectified forms in parentheses.",
                 judge_ignore="traditional vs 1990-rectified French spelling (circumflex on i/u, ognon/oignon, évènement/événement, hyphens in numbers)")),
    _mk(MLFamily("de_1996", "de", "German", "German in the reformed (1996/2006) orthography (dass, muss, Fluss, bisschen, Tipp)",
                 "reformed spelling (dass, muss, Fluss)", "pre-1996 spelling (daß, muß, Fluß)", DE_1996,
                 "Write in German with the REFORMED (1996) orthography. Use at least 6 DIFFERENT items from this list, in exactly the reformed form given first "
                 "(«dass» may count once even if used several times): " + _pairs_text(DE_1996) + ". Never write the pre-reform forms in parentheses.",
                 judge_ignore="pre- vs post-1996 German spelling (ß vs ss, daß/dass, Tip/Tipp, capitalisation of fixed phrases)")),
]
ML_FAMILIES += [
    _mk_prop(MLFamily("zh_simp_trad", "zh", "Chinese", "Simplified Chinese (mainland standard)",
                      "Simplified Chinese (这个, 软件, 说)", "Traditional Chinese (這個, 軟體, 說)", [],
                      "Translate into SIMPLIFIED Chinese (mainland standard), one paragraph of natural written Chinese.",
                      judge_ignore="Simplified vs Traditional Chinese characters and regional word forms"),
             _zh_property("ML_zh_simp_trad", "zh_simp_trad", "zh-cn", "zh-tw", "Simplified Chinese", "Traditional Chinese")),
    _mk_prop(MLFamily("zh_tw_hk", "zh", "Chinese", "Traditional Chinese as written in Taiwan (裡, 著, 為, 軟體, 網路, 計程車)",
                      "Taiwan Traditional (裡, 著, 軟體)", "Hong Kong Traditional (裏, 着, 軟件)", [],
                      "Translate into TRADITIONAL Chinese as written in TAIWAN (裡 not 裏, 著 not 着, 軟體, 網路, 計程車, 公車), one paragraph of natural written Chinese. "
                      "Include several of the words 裡, 著 (aspect marker), 為, 這裡, 那裡, 軟體, 網路, 資料, 影片, 公車, 計程車 where natural.",
                      judge_ignore="Taiwan vs Hong Kong character variants and regional vocabulary"),
             _zh_property("ML_zh_tw_hk", "zh_tw_hk", "zh-tw", "zh-hk", "Taiwan Traditional", "Hong Kong Traditional")),
    _mk_prop(MLFamily("ja_long_vowel", "ja", "Japanese", "Japanese, writing loanwords WITH the trailing long-vowel mark (コンピューター, ユーザー, サーバー)",
                      "with trailing ー (コンピューター)", "without trailing ー (コンピュータ)", JA_LONG,
                      "Translate into natural Japanese. Use at least 6 DIFFERENT loanwords from this list, written exactly WITH the trailing ー: "
                      + ", ".join(a for a, _ in JA_LONG) + ". Never drop the trailing ー on these words.",
                      judge_ignore="presence or absence of the trailing long-vowel mark ー on loanwords"),
             _substr_property("ML_ja_long_vowel", "ja_long_vowel", JA_LONG, "with trailing ー", "without trailing ー")),
    _mk_prop(MLFamily("ru_yo", "ru", "Russian", "Russian, always writing the letter ё where it is pronounced (ещё, её, всё, идёт, ребёнок)",
                      "ё written (ещё, всё, идёт)", "ё replaced by е (еще, все, идет)", [],
                      "Translate into natural Russian and ALWAYS write the letter ё where it is pronounced (ещё, её, всё, идёт, даёт, ребёнок, зелёный, четвёртый); "
                      "never replace ё by е. The text must contain at least 6 words with ё.",
                      judge_ignore="the letter ё written as е"),
             _rule_property("ML_ru_yo", "ru_yo", "ё", "е", RU_YO_WORDS, "ё written", "ё as е")),
    _mk_prop(MLFamily("de_swiss", "de", "German", "standard German with ß (Straße, groß, heißt, außerdem)",
                      "ß (Straße, groß, heißt)", "Swiss ss (Strasse, gross, heisst)", [],
                      "Translate into standard German (reformed orthography) and use at least 6 DIFFERENT words containing ß, e.g. " + ", ".join(DE_SS_WORDS[:40]) + ".",
                      judge_ignore="ß vs ss (Swiss spelling)"),
             _rule_property("ML_de_swiss", "de_swiss", "ß", "ss", DE_SS_WORDS, "ß", "ss")),
    _mk(MLFamily("uk_2019", "uk", "Ukrainian", "Ukrainian in the 2019 orthography (проєкт, авдиторія, етер, катедра, пів години)",
                 "2019 orthography (проєкт, етер, пів години)", "pre-2019 orthography (проект, ефір, півгодини)", UK_2019,
                 "Translate into Ukrainian following the 2019 orthography. Use at least 6 DIFFERENT words from this list, in exactly the 2019 form given first: "
                 + _pairs_text(UK_2019) + ". Never write the pre-2019 forms in parentheses.",
                 judge_ignore="pre- vs post-2019 Ukrainian spelling (проект/проєкт, аудиторія/авдиторія, ефір/етер, пів години)")),
]

# =============================== round 3 (2026-09-14): previously pruned ideas, now tested ===============================
import unicodedata
from src.sandbox.ext_styleprops.properties import _US_UK

EN_ARCHAIC = [("today", "to-day"), ("tomorrow", "to-morrow"), ("tonight", "to-night"), ("connection", "connexion"), ("connections", "connexions"),
    ("reflection", "reflexion"), ("inflection", "inflexion"), ("cooperate", "coöperate"), ("cooperation", "coöperation"), ("cooperative", "coöperative"),
    ("coordinate", "coördinate"), ("coordination", "coördination"), ("reenter", "reënter"), ("preeminent", "preëminent"), ("show", "shew"), ("shows", "shews"),
    ("showed", "shewed"), ("shown", "shewn"), ("jail", "gaol"), ("jailer", "gaoler"), ("wagon", "waggon"), ("wagons", "waggons"), ("anyone", "any one"),
    ("everyone", "every one"), ("someone", "some one"), ("cannot", "can not"), ("percent", "per cent"), ("Halloween", "Hallowe'en"), ("encyclopedia", "encyclopædia"),
    ("medieval", "mediæval"), ("aesthetic", "æsthetic"), ("fantasy", "phantasy"), ("demon", "dæmon"), ("role", "rôle"), ("hotel", "hôtel"), ("depot", "dépôt"),
    ("develop", "develope"), ("surprise", "surprize"), ("surprised", "surprized"), ("music", "musick"), ("public", "publick"), ("magic", "magick"),
    ("choose", "chuse"), ("clothes", "cloathes"), ("although", "altho'"), ("though", "tho'"), ("through", "thro'"), ("its", "it's")]
EN_ARCHAIC = [(a, b) for a, b in EN_ARCHAIC if a not in ("its", "public", "music", "magic", "choose", "clothes", "develop", "surprise", "surprised")]

EN_CANADIAN = [(a, b) for a, b in _US_UK]                       # -our/-re/-ll/-ogue/-ence pairs; -ize words stay -ize in both poles (Canadian/Oxford)

_COLL = ["team", "family", "staff", "government", "committee", "crew", "class", "group", "band", "club", "company", "audience", "jury", "council", "couple", "public"]
EN_COLLECTIVE = []
for n in _COLL:
    for art in ("the", "our", "my", "a", "this"):
        for sg, pl in (("is", "are"), ("has", "have"), ("was", "were"), ("wants", "want"), ("needs", "need"), ("does", "do"), ("plays", "play"), ("meets", "meet"),
                       ("decides", "decide"), ("thinks", "think"), ("says", "say"), ("works", "work"), ("hopes", "hope"), ("plans", "plan"), ("agrees", "agree"), ("isn't", "aren't"), ("doesn't", "don't")):
            EN_COLLECTIVE.append((f"{art} {n} {sg}", f"{art} {n} {pl}"))

EN_EXONYMS = [("Kyiv", "Kiev"), ("Kharkiv", "Kharkov"), ("Lviv", "Lvov"), ("Odesa", "Odessa"), ("Mumbai", "Bombay"), ("Chennai", "Madras"), ("Kolkata", "Calcutta"),
    ("Bengaluru", "Bangalore"), ("Pune", "Poona"), ("Kochi", "Cochin"), ("Thiruvananthapuram", "Trivandrum"), ("Yangon", "Rangoon"), ("Myanmar", "Burma"),
    ("Ho Chi Minh City", "Saigon"), ("Beijing", "Peking"), ("Nanjing", "Nanking"), ("Guangzhou", "Canton"), ("Tianjin", "Tientsin"), ("Chongqing", "Chungking"),
    ("Xi'an", "Sian"), ("Suzhou", "Soochow"), ("Hangzhou", "Hangchow"), ("Qingdao", "Tsingtao"), ("Shenyang", "Mukden"), ("Mao Zedong", "Mao Tse-tung"),
    ("Busan", "Pusan"), ("Daegu", "Taegu"), ("Incheon", "Inchon"), ("Gwangju", "Kwangju"), ("Jeju", "Cheju"), ("Almaty", "Alma-Ata"), ("Chișinău", "Kishinev"),
    ("Tbilisi", "Tiflis"), ("Thessaloniki", "Salonika"), ("Livorno", "Leghorn"), ("Eswatini", "Swaziland"), ("Czechia", "the Czech Republic"), ("Sri Lanka", "Ceylon"),
    ("Tokyo", "Tokio"), ("Zhejiang", "Chekiang"), ("Sichuan", "Szechuan"), ("Shandong", "Shantung"), ("Guangdong", "Kwangtung"), ("Fujian", "Fukien"), ("Xinjiang", "Sinkiang"),
    ("Vadodara", "Baroda"), ("Varanasi", "Benares"), ("Puducherry", "Pondicherry"), ("Mangaluru", "Mangalore"), ("Shimla", "Simla"), ("Kanpur", "Cawnpore")]

EN_INCLUSIVE = [("chairman", "chairperson"), ("chairmen", "chairpersons"), ("fireman", "firefighter"), ("firemen", "firefighters"), ("policeman", "police officer"),
    ("policemen", "police officers"), ("mankind", "humankind"), ("manpower", "workforce"), ("stewardess", "flight attendant"), ("stewardesses", "flight attendants"),
    ("waitress", "server"), ("waitresses", "servers"), ("mailman", "mail carrier"), ("mailmen", "mail carriers"), ("spokesman", "spokesperson"), ("spokesmen", "spokespersons"),
    ("businessman", "businessperson"), ("businessmen", "businesspeople"), ("freshman", "first-year student"), ("freshmen", "first-year students"), ("congressman", "member of Congress"),
    ("salesman", "salesperson"), ("salesmen", "salespeople"), ("foreman", "supervisor"), ("man-made", "artificial"), ("manned", "staffed"), ("cameraman", "camera operator"),
    ("fisherman", "fisher"), ("fishermen", "fishers"), ("layman", "layperson"), ("weatherman", "weather forecaster"), ("housewife", "homemaker"), ("housewives", "homemakers"),
    ("repairman", "repair technician"), ("middleman", "intermediary"), ("workmanship", "craftsmanship"), ("man hours", "person hours"), ("mankind's", "humankind's"),
    ("chairwoman", "chairperson"), ("anchorman", "anchor"), ("garbageman", "sanitation worker"), ("doorman", "door attendant"), ("handyman", "maintenance worker"),
    ("craftsman", "artisan"), ("craftsmen", "artisans"), ("draftsman", "drafter"), ("watchman", "security guard"), ("policewoman", "police officer"), ("man the", "staff the")]

EN_PREP = [("on the weekend", "at the weekend"), ("on weekends", "at weekends"), ("in the hospital", "in hospital"), ("Monday through Friday", "Monday to Friday"),
    ("write me", "write to me"), ("different from", "different to"), ("different than", "different to"), ("on the team", "in the team"), ("fill out", "fill in"),
    ("protest the", "protest against the"), ("appeal the", "appeal against the"), ("a quarter after", "a quarter past"), ("half after", "half past"), ("ten of", "ten to"),
    ("meet with", "meet"), ("on the street", "in the street"), ("in back of", "behind"), ("on a diet", "on a diet"), ("in line", "in the queue"), ("stand in line", "queue up"),
    ("on the phone", "on the phone"), ("at the university", "at university"), ("in the future", "in future"), ("check it out", "check it"), ("talk with", "talk to"),
    ("visit with", "visit"), ("get in touch with", "get in touch with"), ("outside of", "outside"), ("off of", "off"), ("named for", "named after"), ("cater to", "cater for"),
    ("bored of", "bored with"), ("on vacation", "on holiday"), ("at the front of", "in front of"), ("a week from Monday", "a week on Monday"), ("Monday through Wednesday", "Monday to Wednesday")]
EN_PREP = [(a, b) for a, b in EN_PREP if a != b]

BE_TARASK = [("снег", "сьнег"), ("свет", "сьвет"), ("з'ява", "зьява"), ("план", "плян"), ("клуб", "клюб"), ("Еўропа", "Эўропа"), ("лагічны", "лягічны"), ("метад", "мэтад"),
    ("філасофія", "філязофія"), ("сімвал", "сымбаль"), ("сістэма", "сыстэма"), ("лабараторыя", "лябараторыя"), ("класічны", "клясычны"), ("парламент", "парлямэнт"),
    ("сцяна", "сьцяна"), ("свята", "сьвята"), ("жыццё", "жыцьцё"), ("вяселле", "вясельле"), ("каханне", "каханьне"), ("пытанне", "пытаньне"), ("змена", "зьмена"),
    ("сняданак", "сьняданак"), ("слесар", "сьлесар"), ("спеў", "сьпеў"), ("снеданне", "сьнеданьне"), ("дзверы", "дзьверы"), ("здзейсніць", "зьдзейсьніць"),
    ("веданне", "веданьне"), ("чытанне", "чытаньне"), ("сумленне", "сумленьне"), ("здароўе", "здароўе"), ("клас", "кляса"), ("кампутар", "кампутар"),
    ("філолаг", "філёляг"), ("фізіка", "фізыка"), ("тэорыя", "тэорыя"), ("грамадства", "грамадзтва"), ("бліскучы", "бліскучы"), ("дысцыпліна", "дысцыпліна"),
    ("сталіца", "сталіца"), ("змяніць", "зьмяніць"), ("з'езд", "зьезд"), ("аб'ём", "абʼём"), ("сцвярджаць", "сьцьвярджаць"), ("светла", "сьветла"), ("спяваць", "сьпяваць"),
    ("снежань", "сьнежань"), ("мясцовы", "мясцовы"), ("пенсія", "пэнсія"), ("тэлефон", "тэлефон"), ("аўтамабіль", "аўтамабіль"), ("аэрапорт", "аэрапорт"),
    ("сыр", "сыр"), ("зверы", "зьверы"), ("звязаць", "зьвязаць"), ("склад", "склад"), ("гісторыя", "гісторыя"), ("сяброўства", "сяброўства"), ("спадчына", "спадчына"),
    ("сцежка", "сьцежка"), ("снедаць", "сьнедаць"), ("змяншаць", "зьмяншаць"), ("з'явіцца", "зьявіцца"), ("дзве", "дзьве"), ("дзверцы", "дзьверцы"), ("цвік", "цьвік"),
    ("цвісці", "цьвісьці"), ("свеціць", "сьвяціць"), ("сябры", "сябры"), ("плошча", "плошча"), ("нацыя", "нацыя"), ("тэхніка", "тэхніка"), ("матэматыка", "матэматыка")]
BE_TARASK = [(a, b) for a, b in BE_TARASK if a != b]

DE_INCLUSIVE = [("Studenten", "Student*innen"), ("Lehrer", "Lehrer*innen"), ("Mitarbeiter", "Mitarbeiter*innen"), ("Kunden", "Kund*innen"), ("Teilnehmer", "Teilnehmer*innen"),
    ("Nutzer", "Nutzer*innen"), ("Kollegen", "Kolleg*innen"), ("Bürger", "Bürger*innen"), ("Schüler", "Schüler*innen"), ("Leser", "Leser*innen"), ("Besucher", "Besucher*innen"),
    ("Fahrer", "Fahrer*innen"), ("Radfahrer", "Radfahrer*innen"), ("Käufer", "Käufer*innen"), ("Verkäufer", "Verkäufer*innen"), ("Ärzte", "Ärzt*innen"), ("Forscher", "Forscher*innen"),
    ("Autoren", "Autor*innen"), ("Experten", "Expert*innen"), ("Anfänger", "Anfänger*innen"), ("Freunde", "Freund*innen"), ("Nachbarn", "Nachbar*innen"), ("Gärtner", "Gärtner*innen"),
    ("Bäcker", "Bäcker*innen"), ("Läufer", "Läufer*innen"), ("Spieler", "Spieler*innen"), ("Zuschauer", "Zuschauer*innen"), ("Sportler", "Sportler*innen"), ("Helfer", "Helfer*innen"),
    ("Handwerker", "Handwerker*innen"), ("Mechaniker", "Mechaniker*innen"), ("Trainer", "Trainer*innen"), ("Wanderer", "Wander*innen"), ("Köche", "Köch*innen"), ("Lernende", "Lernende"),
    ("Berater", "Berater*innen"), ("Bewohner", "Bewohner*innen"), ("Gäste", "Gäst*innen"), ("Kollege", "Kolleg*in"), ("Mitarbeitern", "Mitarbeiter*innen"), ("Kunde", "Kund*in"),
    ("Nutzern", "Nutzer*innen"), ("Teilnehmern", "Teilnehmer*innen"), ("Lehrern", "Lehrer*innen"), ("Schülern", "Schüler*innen"), ("Freunden", "Freund*innen"), ("Nachbarinnen", "Nachbar*innen")]
DE_INCLUSIVE = [(a, b) for a, b in DE_INCLUSIVE if a != b]

FR_INCLUSIVE = [("étudiants", "étudiant·e·s"), ("participants", "participant·e·s"), ("utilisateurs", "utilisateur·rice·s"), ("employés", "employé·e·s"), ("enseignants", "enseignant·e·s"),
    ("visiteurs", "visiteur·euse·s"), ("clients", "client·e·s"), ("voisins", "voisin·e·s"), ("amis", "ami·e·s"), ("habitants", "habitant·e·s"), ("débutants", "débutant·e·s"),
    ("coureurs", "coureur·euse·s"), ("vendeurs", "vendeur·euse·s"), ("acheteurs", "acheteur·euse·s"), ("lecteurs", "lecteur·rice·s"), ("auteurs", "auteur·rice·s"), ("experts", "expert·e·s"),
    ("jardiniers", "jardinier·ère·s"), ("boulangers", "boulanger·ère·s"), ("conducteurs", "conducteur·rice·s"), ("passagers", "passager·ère·s"), ("citoyens", "citoyen·ne·s"),
    ("travailleurs", "travailleur·euse·s"), ("bénévoles", "bénévoles"), ("mécaniciens", "mécanicien·ne·s"), ("cuisiniers", "cuisinier·ère·s"), ("voyageurs", "voyageur·euse·s"),
    ("randonneurs", "randonneur·euse·s"), ("apprentis", "apprenti·e·s"), ("commerçants", "commerçant·e·s"), ("propriétaires", "propriétaires"), ("invités", "invité·e·s"),
    ("professeurs", "professeur·e·s"), ("directeurs", "directeur·rice·s"), ("chercheurs", "chercheur·euse·s"), ("infirmiers", "infirmier·ère·s"), ("agriculteurs", "agriculteur·rice·s"),
    ("artisans", "artisan·e·s"), ("musiciens", "musicien·ne·s"), ("sportifs", "sportif·ve·s"), ("nageurs", "nageur·euse·s"), ("joueurs", "joueur·euse·s"), ("spectateurs", "spectateur·rice·s")]
FR_INCLUSIVE = [(a, b) for a, b in FR_INCLUSIVE if a != b]

KO_NORTH = [("노동", "로동"), ("여자", "녀자"), ("이유", "리유"), ("역사", "력사"), ("요리", "료리"), ("연습", "련습"), ("이용", "리용"), ("유행", "류행"), ("이발", "리발"),
    ("낙원", "락원"), ("냉면", "랭면"), ("노인", "로인"), ("녹음", "록음"), ("내일", "래일"), ("노선", "로선"), ("이론", "리론"), ("예절", "례절"), ("유리", "류리"), ("양심", "량심"),
    ("연락", "련락"), ("이해", "리해"), ("열차", "렬차"), ("노력", "로력"), ("연도", "년도"), ("여행", "려행"), ("여름", "여름"), ("유학", "류학"), ("이익", "리익"), ("이념", "리념"),
    ("역할", "역할"), ("연구", "연구"), ("낙하", "락하"), ("노년", "로년"), ("노무", "로무"), ("양식", "량식"), ("이자", "리자"), ("이상", "리상"), ("예의", "례의"), ("누각", "루각"),
    ("여권", "려권"), ("염려", "념려"), ("유용", "류용"), ("이별", "리별"), ("영도", "령도"), ("녹색", "록색"), ("노출", "로출"), ("이웃", "이웃"), ("나열", "라렬"), ("낙관", "락관"),
    ("노선도", "로선도"), ("연령", "년령"), ("육지", "륙지"), ("육상", "륙상"), ("여객", "려객"), ("역", "력"), ("요금", "료금"), ("이성", "리성"), ("이륙", "리륙")]
KO_NORTH = [(a, b) for a, b in KO_NORTH if a != b and a not in ("역",)]

JA_HIST_KANA = [("思う", "思ふ"), ("言う", "言ふ"), ("使う", "使ふ"), ("買う", "買ふ"), ("会う", "会ふ"), ("行う", "行ふ"), ("笑う", "笑ふ"), ("願う", "願ふ"), ("習う", "習ふ"), ("洗う", "洗ふ"),
    ("拾う", "拾ふ"), ("縫う", "縫ふ"), ("払う", "払ふ"), ("誘う", "誘ふ"), ("手伝う", "手伝ふ"), ("従う", "従ふ"), ("歌う", "歌ふ"), ("疑う", "疑ふ"), ("扱う", "扱ふ"), ("違う", "違ふ"),
    ("合う", "合ふ"), ("追う", "追ふ"), ("救う", "救ふ"), ("吸う", "吸ふ"), ("通う", "通ふ"), ("味わう", "味はふ"), ("向かう", "向かふ"), ("整う", "整ふ"), ("戦う", "戦ふ"), ("失う", "失ふ"),
    ("思います", "思ひます"), ("言います", "言ひます"), ("使います", "使ひます"), ("買います", "買ひます"), ("会います", "会ひます"), ("行います", "行ひます"), ("笑います", "笑ひます"),
    ("願います", "願ひます"), ("習います", "習ひます"), ("洗います", "洗ひます"), ("払います", "払ひます"), ("手伝います", "手伝ひます"), ("従います", "従ひます"), ("違います", "違ひます"),
    ("合います", "合ひます"), ("扱います", "扱ひます"), ("思えば", "思へば"), ("言えば", "言へば"), ("使えば", "使へば"), ("買えば", "買へば"), ("会えば", "会へば"), ("教える", "教へる"),
    ("考える", "考へる"), ("答える", "答へる"), ("迎える", "迎へる"), ("与える", "与へる"), ("伝える", "伝へる"), ("整える", "整へる"), ("覚える", "覚える"), ("見える", "見える"),
    ("教えます", "教へます"), ("考えます", "考へます"), ("答えます", "答へます"), ("伝えます", "伝へます"), ("与えます", "与へます"), ("用いる", "用ゐる"), ("用います", "用ゐます"),
    ("ように", "やうに"), ("ような", "やうな"), ("そう", "さう"), ("こう", "かう"), ("どう", "だう"), ("今日", "けふ"), ("いう", "いふ"), ("いえば", "いへば"), ("おもう", "おもふ"),
    ("いよう", "いやう"), ("しよう", "せう"), ("しましょう", "しませう"), ("でしょう", "でせう"), ("ましょう", "ませう")]
JA_HIST_KANA = [(a, b) for a, b in JA_HIST_KANA if a != b]


def _nfd_property(cls_name, prop_name, nat_lab, alt_lab):
    """Unicode NFC (nat) vs NFD (alt): every word with a precomposed accented letter is an opportunity; alt = NFD of the word.
    classify: combining marks present -> alt; precomposed accented letters present -> nat."""
    comb = re.compile("[\u0300-\u036f]")
    def has_precomposed(w): return any(unicodedata.normalize("NFD", ch) != ch for ch in w)
    word_rx = re.compile(r"\w+", re.UNICODE)

    class N(Property):
        name = prop_name
        family = "spelling"

        def find_opps(self, text):
            return _dedup([Opp(m.start(), m.end(), m.group(0), unicodedata.normalize("NFD", m.group(0))) for m in word_rx.finditer(text) if has_precomposed(m.group(0))])

        def classify(self, tail):
            m_c = comb.search(tail)
            m_p = next((i for i, ch in enumerate(tail) if unicodedata.normalize("NFD", ch) != ch), None)
            if m_c and (m_p is None or m_c.start() <= m_p):
                return "alt"
            if m_p is not None:
                return "nat"
            return None

        @staticmethod
        def to_nat(text): return unicodedata.normalize("NFC", text)

    N.__name__ = cls_name; N.nat_label, N.alt_label, N.confound = nat_lab, alt_lab, "low"
    return N


_COLL_VERBS = [("is", "are"), ("has", "have"), ("was", "were"), ("wants", "want"), ("needs", "need"), ("does", "do"), ("plays", "play"), ("meets", "meet"), ("decides", "decide"),
               ("thinks", "think"), ("says", "say"), ("works", "work"), ("hopes", "hope"), ("plans", "plan"), ("agrees", "agree"), ("isn't", "aren't"), ("doesn't", "don't"),
               ("hasn't", "haven't"), ("wasn't", "weren't"), ("gathers", "gather"), ("prefers", "prefer"), ("takes", "take"), ("makes", "make"), ("gets", "get"), ("keeps", "keep"),
               ("comes", "come"), ("goes", "go"), ("uses", "use"), ("shares", "share"), ("enjoys", "enjoy"), ("holds", "hold"), ("organizes", "organize"), ("runs", "run"), ("starts", "start")]
_COLL_VERB_MAP = {a: b for a, b in _COLL_VERBS} | {b: a for a, b in _COLL_VERBS}
_COLL_RX = re.compile(r"\b((?:[Tt]he|[Oo]ur|[Mm]y|[Aa]|[Tt]his|[Yy]our|[Tt]heir|[Ee]ach)\s+(?:\w+\s+)?(?:" + "|".join(_COLL) + r"))\s+(" + "|".join(re.escape(v) for v in _COLL_VERB_MAP) + r")\b")


class _EnCollective(Property):
    """collective noun (optionally with one adjective) + verb: singular agreement (nat) vs plural (alt); the opportunity span is the verb."""
    name = "en_collective"; family = "spelling"; nat_label = "singular agreement"; alt_label = "plural agreement"; confound = "low"

    def find_opps(self, text):
        out = []
        for m in _COLL_RX.finditer(text):
            v = m.group(2); other = _COLL_VERB_MAP[v]
            sg = v if v in dict(_COLL_VERBS) else other
            out.append(Opp(m.start(2), m.end(2), sg, other if sg == v else v))
        return _dedup(out)

    def classify(self, tail):
        m = re.match(r"\s*(" + "|".join(re.escape(v) for v in _COLL_VERB_MAP) + r")\b", tail)
        if m is None:
            m = _COLL_RX.search(tail)
            if m is None:
                return None
            v = m.group(2)
        else:
            v = m.group(1)
        return "nat" if v in dict(_COLL_VERBS) else "alt"


ML_FAMILIES += [
    _mk(MLFamily("en_archaic", "en", "English", "standard modern American English",
                 "modern spelling (today, connection, show, jail)", "archaic house style (to-day, connexion, shew, gaol)", EN_ARCHAIC,
                 "Write in standard modern American English. Use at least 6 DIFFERENT words from this list, in the modern form given first: " + _pairs_text(EN_ARCHAIC)
                 + ". Never write the archaic forms in parentheses.", judge_ignore="archaic vs modern spellings of the same word (to-day/today, connexion, shew, gaol, per cent)", src_lang="Spanish")),
    _mk(MLFamily("en_canadian", "en", "English", "standard American English (color, center, traveler, organize)",
                 "American spelling (color, center, organize)", "Canadian/Oxford spelling (colour, centre, but organize)", EN_CANADIAN,
                 "Write in standard AMERICAN English. Use at least 6 DIFFERENT words from this list, in the American form given first: " + _pairs_text(EN_CANADIAN[:80])
                 + ". Also use at least 3 verbs ending in -ize (organize, realize, recognize, prioritize, minimize). Never write the British forms in parentheses.",
                 judge_ignore="American vs Canadian/British spelling (colour/color, centre/center)", src_lang="Spanish")),
    _mk_prop(MLFamily("en_collective", "en", "English", "standard American English with SINGULAR verb agreement for collective nouns (the team is, the family has)",
                 "singular agreement (the team is)", "plural agreement (the team are)", EN_COLLECTIVE,
                 "Write in standard American English. The paragraph MUST contain at least 7 sentences whose grammatical subject is one of these collective nouns: team, family, staff, government, committee, crew, class, group, band, club, company, audience, jury, council, couple — "
                 "written as «the/our/my/a/this + (optional adjective) + noun» and IMMEDIATELY followed by the verb, using one of: is, has, was, wants, needs, does, plays, meets, decides, thinks, says, works, hopes, plans, agrees, gathers, prefers, takes, makes, gets, keeps, comes, goes, uses, shares, enjoys, holds, organizes, runs, starts "
                 "(e.g. «the team is ready», «our family has decided», «the committee wants», «this club meets weekly», «the crew takes»). Use SINGULAR agreement throughout (is/has/was/wants). Vary the nouns.",
                 judge_ignore="singular vs plural verb agreement with collective nouns (the team is/are)", src_lang="Spanish"), _EnCollective),
    _mk(MLFamily("en_exonyms", "en", "English", "standard modern English using CURRENT place names and romanisations (Kyiv, Mumbai, Beijing, Busan)",
                 "current names (Kyiv, Mumbai, Beijing)", "older exonyms (Kiev, Bombay, Peking)", EN_EXONYMS,
                 "Write in standard modern English about travel, history or geography so that the paragraph mentions at least 6 DIFFERENT place names from this list, in the current form given first: "
                 + _pairs_text(EN_EXONYMS) + ". Never write the older names in parentheses.", judge_ignore="current vs older place names and romanisations (Kyiv/Kiev, Mumbai/Bombay, Beijing/Peking)", src_lang="Spanish")),
    _mk(MLFamily("en_inclusive", "en", "English", "standard American English using the traditional occupational terms (chairman, fireman, policeman, mankind)",
                 "traditional terms (chairman, fireman, mankind)", "gender-inclusive terms (chairperson, firefighter, humankind)", EN_INCLUSIVE,
                 "Write in standard American English. Use at least 6 DIFFERENT words from this list, in the traditional form given first: " + _pairs_text(EN_INCLUSIVE)
                 + ". Never write the inclusive forms in parentheses.", judge_ignore="traditional vs gender-inclusive occupational terms (chairman/chairperson, fireman/firefighter)", src_lang="Spanish")),
    _mk(MLFamily("en_prep_idioms", "en", "English", "standard AMERICAN English prepositional idioms (on the weekend, in the hospital, Monday through Friday, different from)",
                 "American idioms (on the weekend, in the hospital)", "British idioms (at the weekend, in hospital)", EN_PREP,
                 "Write in standard AMERICAN English. Use at least 6 DIFFERENT phrases from this list, exactly in the American form given first: " + _pairs_text(EN_PREP)
                 + ". Never write the British forms in parentheses.", judge_ignore="American vs British prepositional idioms (on/at the weekend, in (the) hospital, through/to)", src_lang="Spanish")),
    _mk(MLFamily("be_tarask", "be", "Belarusian", "Belarusian in the official (academic, narkamaŭka) orthography",
                 "official orthography (снег, план, Еўропа)", "Taraškievica (сьнег, плян, Эўропа)", BE_TARASK,
                 "Write in Belarusian using the OFFICIAL (academic) orthography. Use at least 6 DIFFERENT words from this list, in the official form given first: " + _pairs_text(BE_TARASK)
                 + ". Never write the Taraškievica forms in parentheses.", judge_ignore="official vs Taraškievica Belarusian orthography (soft signs, foreign-word spellings)")),
    _mk(MLFamily("de_inclusive", "de", "German", "German with the generic masculine plural (die Studenten, die Mitarbeiter, die Kunden)",
                 "generic masculine (Studenten, Kunden)", "gender star (Student*innen, Kund*innen)", DE_INCLUSIVE,
                 "Write in German using the GENERIC MASCULINE for groups of people. Use at least 6 DIFFERENT plural nouns from this list, in the form given first: " + _pairs_text(DE_INCLUSIVE)
                 + ". Never write the gender-star forms in parentheses.", judge_ignore="generic masculine vs gender-star forms (Studenten/Student*innen)")),
    _mk(MLFamily("fr_inclusive", "fr", "French", "French with the generic masculine plural (les étudiants, les participants, les utilisateurs)",
                 "generic masculine (étudiants, clients)", "écriture inclusive (étudiant·e·s, client·e·s)", FR_INCLUSIVE,
                 "Write in French using the GENERIC MASCULINE plural for groups of people. Use at least 6 DIFFERENT plural nouns from this list, in the form given first: " + _pairs_text(FR_INCLUSIVE)
                 + ". Never write the point-médian forms in parentheses.", judge_ignore="generic masculine vs écriture inclusive with point médian (étudiants/étudiant·e·s)")),
    _mk_prop(MLFamily("es_nfd", "es", "Spanish", "Spanish (standard, with accented letters as usual: más, también, está, año, día, también)",
                      "NFC (precomposed é, ñ)", "NFD (e + combining accent)", [],
                      "Write in natural Spanish; the paragraph must contain at least 8 words with accented letters or ñ (más, también, está, año, día, después, así, aquí, sólo?, país, número, fácil, difícil, camión, mañana).",
                      judge_ignore="Unicode normalisation of accented letters (precomposed vs decomposed)"),
             _nfd_property("ML_es_nfd", "es_nfd", "NFC", "NFD")),
    _mk_prop(MLFamily("ko_north", "ko", "Korean", "standard South Korean orthography (노동, 여자, 이유, 역사, 요리, 연습)",
                 "South Korean (노동, 여자, 역사)", "North Korean (로동, 녀자, 력사)", KO_NORTH,
                 "Write in standard SOUTH Korean. Use at least 6 DIFFERENT words from this list, in the South Korean form given first: " + _pairs_text(KO_NORTH)
                 + ". Never write the North Korean forms in parentheses.", judge_ignore="South vs North Korean spelling (initial ㄹ/ㄴ: 노동/로동, 여자/녀자)"),
             _substr_property("ML_ko_north", "ko_north", KO_NORTH, "South Korean", "North Korean")),
    _mk_prop(MLFamily("ja_hist_kana", "ja", "Japanese", "modern Japanese kana spelling (思う, 言う, 使う, 教える, ように, そう)",
                      "modern kana (思う, 教える, ように)", "historical kana (思ふ, 教へる, やうに)", JA_HIST_KANA,
                      "Write in natural modern Japanese. Use at least 6 DIFFERENT expressions from this list, exactly in the modern form given first (dictionary or ます form as listed): "
                      + ", ".join(a for a, _ in JA_HIST_KANA[:60]) + ". Never use the historical kana forms.", judge_ignore="modern vs historical kana spelling (思う/思ふ, 教える/教へる, ように/やうに)"),
             _substr_property("ML_ja_hist_kana", "ja_hist_kana", JA_HIST_KANA, "modern kana", "historical kana")),
]
ML_FAMILY = {f.name: f for f in ML_FAMILIES}
ML_NAMES = [f.name for f in ML_FAMILIES]


try:
    from src.sandbox.style_translation import code_families as _code_families  # noqa: F401  (appends the code families to ML_FAMILIES / ML_FAMILY / PROPS)
except Exception as _e:  # pragma: no cover
    print("code_families not loaded:", _e)


if __name__ == "__main__":
    for f in ML_FAMILIES:
        print(f"{f.name:14s} {f.tgt_lang:11s} pairs={len(f.pairs):3d}  nat={f.nat!r}  alt={f.alt!r}")
    # roundtrip smoke test
    tests = {"pt_br_eu": "Peguei o ônibus e depois o trem; o celular ficou no banheiro, um fato econômico.",
             "pt_acordo_eu": "A ação foi ótima e a direção atual manteve o objetivo do projeto exato.",
             "pt_acordo_br": "A ideia era um voo tranquilo com frequência, sem consequência para a assembleia.",
             "es_rae2010": "Solo hace falta paciencia. Este es el más fácil y aquellos que empiezan prefieren esa. Este libro no cuenta. Estaba solo en casa.",
             "fr_1990": "Le coût du dîner sur l'île paraît élevé, mais la chaîne connaît un événement en août.",
             "de_1996": "Ich weiß, dass er muss, ein bisschen am Fluss, zum Schluss ein Tipp im Prozess.",
             "zh_simp_trad": "这个软件的头发以后会更长，我们说的是网络和数据。他着急地看着计程车。",
             "zh_tw_hk": "這個軟體的頭髮以後會更長，我們說的是網路和資料。他著急地看著計程車。",
             "ja_long_vowel": "新しいコンピューターとプリンターをユーザーがサーバーに接続し、フォルダーをモニターで確認した。",
             "ru_yo": "Ещё вчера её ребёнок всё время идёт вперёд и даёт зелёный свет.",
             "de_swiss": "Die Straße ist groß und heißt außerdem Fußweg; schließlich weiß man, dass Spaß regelmäßig ist.",
             "uk_2019": "Цей проєкт зібрав авдиторію в етері, а катедра згадала міт про Атени за пів години.",
             "en_archaic": "Today the connection will show that anyone can cooperate; tomorrow the wagon reaches the jail, and everyone knows it cannot fail. Ten percent agreed.",
             "en_canadian": "The color of the center is a favor to the traveler who organizes the catalog.",
             "en_collective": "The team is ready and our whole family has decided, but the committee wants more; the staff was tired and the government says no; this club meets weekly.",
             "en_exonyms": "From Kyiv we flew to Mumbai, then Beijing, Busan, Yangon and Chennai.",
             "en_inclusive": "The chairman thanked the firemen, the policeman, the stewardess, the salesman and all of mankind.",
             "en_prep_idioms": "On the weekend she was in the hospital Monday through Friday; write me, it is different from the plan on the team, fill out the form, a quarter after nine.",
             "be_tarask": "Снег і свет: план, клуб, Еўропа, лагічны метад, сцяна, свята, жыццё.",
             "de_inclusive": "Die Studenten, Lehrer, Mitarbeiter, Kunden, Teilnehmer und Nutzer kamen mit den Kollegen.",
             "fr_inclusive": "Les étudiants, les participants, les utilisateurs, les employés, les enseignants et les visiteurs sont venus.",
             "es_nfd": "Mañana está más fácil: también el país y el número del camión, así que después vendrá el día.",
             "ko_north": "노동과 여자, 이유와 역사, 요리와 연습, 이용과 유행.",
             "ja_hist_kana": "私は思う。彼はそう言う。道具を使う。教えるように今日は会う。答えます。"}
    for name, t in tests.items():
        p = ML_FAMILY[name].prop; opps = p.find_opps(t)
        alt = "".join(t[(opps[i - 1].end if i else 0):o.start] + o.alt for i, o in enumerate(opps)) + t[opps[-1].end:]
        back = p.find_opps(alt)
        print(f"{name:14s} k={len(opps)} opps={[o.nat for o in opps]}\n   alt: {alt}\n   classify(alt)={p.classify(alt)} k_alt={len(back)} nat_restored={[o.nat for o in back]==[o.nat for o in opps]}")
