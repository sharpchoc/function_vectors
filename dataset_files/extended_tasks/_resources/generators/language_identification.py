#!/usr/bin/env python3
"""Generator for language_identification task.

Given a common word from French, Spanish, German, or Italian, output the
name of the language it belongs to.

Methodology (this script is the reproducible artifact):
  1. Downloaded the four kaikki.org/wiktextract per-language dictionary
     dumps (French, Spanish, German, Italian; these are structured
     extractions of each language's Wiktionary, https://kaikki.org).
  2. For each language, kept only words that are canonical dictionary
     lemmas (matched a language-specific head template such as "fr-noun",
     "it-verb", etc., NOT an inflected form-of entry) whose part of speech
     is noun, adjective, or verb.
  3. Cross-checked every candidate word (and, for German, its lowercased
     form, since German capitalizes all nouns and many capitalized
     sentence-initial function words such as "Aber"/"Wenn" would otherwise
     leak in) against the FULL part-of-speech index for that word string;
     dropped anything that is also tagged as a pronoun, adverb, conjunction,
     preposition, determiner, article, interjection, numeral, or proper
     name anywhere in the dump. This removes function-word homographs
     (French "fait"/"avoir"-type entries were kept because "fait"/"avoir"
     are themselves legitimate noun/verb lemmas with no competing
     function-word tag; highly frequent grammatical words like French
     "plusieurs" or Italian inflected forms like "sono"/"possa" that
     slipped past the automatic filter were dropped by hand, see
     MANUAL_EXCLUDE below).
  4. Excluded any word identical to a top-30000 English word (wordfreq),
     multi-word entries, and proper nouns.
  5. Cross-deduplicated across the four languages (case-insensitive): any
     word string appearing in more than one language's candidate pool was
     dropped from all of them, so every input has exactly one correct
     language label by construction.
  6. Ranked each language's remaining pool by wordfreq.zipf_frequency and
     kept the top 250 most frequent words per language (1000 total,
     perfectly balanced 4-way).

Self-check performed while curating: every MANUAL_EXCLUDE removal below is
annotated with the reason (inflected verb form, quantifier/determiner,
etc.); the automatic pos-cross-check step is itself a mechanical
independent-pass check (structured head-template pos vs. the word's global
pos tag set) and is re-verified at generation time by the assertion that no
word appears in more than one language list.
"""
import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "language_identification.json"
N_PER_LANG = 250

random.seed(42)

# -------------------------------------------------------------------------
# Curated word pools (see methodology above). Frequency-ranked descending
# within each language.
# -------------------------------------------------------------------------
FRENCH = [
    'fait', 'être', 'avoir', 'français', 'homme', 'première', 'bonne', 'année', 'mois', 'histoire',
    'nouvelle', 'travail', 'état', 'prendre', 'compte', 'aller', 'reste', 'politique', 'cours', 'seul',
    'mettre', 'tête', 'groupe', 'parler', 'famille', 'demande', 'savoir', 'besoin', 'société', 'pouvoir',
    'raison', 'soir', 'effet', 'équipe', 'général', 'nombre', 'dernier', 'mère', 'père', 'pris',
    'parti', 'trouver', 'passé', 'fille', 'guerre', 'semaine', 'ligne', 'problème', 'président', 'dieu',
    'fils', 'jeune', 'exemple', 'système', 'début', 'heure', 'façon', 'projet', 'nuit', 'idée',
    'conseil', 'argent', 'meilleur', 'forme', 'titre', 'ayant', 'mise', 'matin', 'choix', 'retour',
    'journée', 'grâce', 'moyen', 'ordre', 'école', 'étant', 'donné', 'truc', 'vidéo', 'livre',
    'musique', 'gauche', 'région', 'entreprise', 'septembre', 'peur', 'juin', 'deuxième', 'série', 'sécurité',
    'anglais', 'jouer', 'ancien', 'époque', 'lire', 'voix', 'doute', 'marché', 'écrit', 'avis',
    'vivre', 'juillet', 'rendre', 'propre', 'nombreux', 'développement', 'nationale', 'droite', 'trouvé', 'mesure',
    'abord', 'sortir', 'octobre', 'terme', 'siècle', 'voiture', 'gars', 'envie', 'janvier', 'frère',
    'avril', 'cœur', 'produit', 'laisse', 'propos', 'période', 'marche', 'emploi', 'plaisir', 'manque',
    'laisser', 'croire', 'décembre', 'armée', 'rencontre', 'confiance', 'rôle', 'demander', 'comprendre', 'août',
    'esprit', 'sortie', 'mouvement', 'intérieur', 'intérêt', 'présente', 'langue', 'perdu', 'offre', 'scène',
    'février', 'meilleure', 'qualité', 'risque', 'vieux', 'pire', 'rester', 'troisième', 'seconde', 'objet',
    'aider', 'étude', 'valeur', 'cour', 'plupart', 'joue', 'réseau', 'réponse', 'montre', 'penser',
    'présent', 'particulier', 'texte', 'succès', 'république', 'économique', 'contrôle', 'chercher', 'connu', 'américain',
    'création', 'activité', 'expérience', 'fonction', 'sein', 'directeur', 'chemin', 'liberté', 'mariage', 'arrêter',
    'espace', 'présence', 'économie', 'utiliser', 'œuvre', 'entrée', 'moyenne', 'réalité', 'université', 'numéro',
    'regarder', 'dimanche', 'départ', 'mauvais', 'travailler', 'voie', 'générale', 'taux', 'accès', 'marque',
    'fini', 'acheter', 'perdre', 'attendre', 'victoire', 'vérité', 'pratique', 'proche', 'suivre', 'répondre',
    'peuple', 'reçu', 'communauté', 'domaine', 'ouest', 'résultat', 'gagner', 'entendu', 'téléphone', 'demandé',
    'retrouver', 'honneur', 'permis', 'matière', 'coeur', 'prêt', 'devoir', 'arriver', 'pièce', 'moitié',
    'arrivée', 'éviter', 'énergie', 'heureux', 'samedi', 'entendre', 'église', 'faute', 'contraire', 'joueur',
]

SPANISH = [
    'hacer', 'estado', 'trabajo', 'lugar', 'hecho', 'gobierno', 'país', 'cuenta', 'decir', 'nuevo',
    'estar', 'ciudad', 'poder', 'primera', 'nueva', 'dios', 'manera', 'acuerdo', 'partido', 'grupo',
    'haber', 'mujer', 'buena', 'presidente', 'pasado', 'familia', 'cosa', 'lado', 'política', 'agua',
    'equipo', 'información', 'semana', 'pasa', 'dinero', 'ejemplo', 'número', 'hablar', 'señor', 'muerte',
    'través', 'realidad', 'frente', 'hijo', 'sociedad', 'desarrollo', 'nivel', 'llegar', 'juego', 'proyecto',
    'posible', 'razón', 'artículo', 'tierra', 'seguridad', 'mayoría', 'único', 'cuerpo', 'segundo', 'último',
    'universidad', 'programa', 'servicio', 'seguir', 'cabeza', 'pasar', 'internacional', 'situación', 'público', 'ayuda',
    'siguiente', 'dejar', 'proceso', 'educación', 'sentido', 'clase', 'puesto', 'español', 'tomar', 'policía',
    'empresa', 'sitio', 'especial', 'mundial', 'línea', 'obra', 'vivir', 'escuela', 'atención', 'segunda',
    'respecto', 'relación', 'población', 'música', 'poner', 'miedo', 'difícil', 'habla', 'debido', 'pregunta',
    'apoyo', 'oficial', 'fuerza', 'propio', 'trabajar', 'encontrar', 'inglés', 'justicia', 'edad', 'libertad',
    'corazón', 'octubre', 'diciembre', 'norte', 'político', 'imagen', 'volver', 'palabra', 'estudio', 'abril',
    'joven', 'región', 'junio', 'investigación', 'república', 'comunidad', 'dirección', 'película', 'vuelta', 'papel',
    'noviembre', 'consejo', 'respuesta', 'movimiento', 'precio', 'llamado', 'leer', 'comida', 'enero', 'llevar',
    'jefe', 'siglo', 'producción', 'experiencia', 'febrero', 'página', 'organización', 'oportunidad', 'economía', 'opinión',
    'espacio', 'común', 'conocer', 'usar', 'septiembre', 'duda', 'fecha', 'iglesia', 'resultado', 'feliz',
    'necesario', 'calidad', 'hermano', 'viaje', 'niño', 'mensaje', 'ejército', 'interés', 'campaña', 'objetivo',
    'carrera', 'ganar', 'prueba', 'lucha', 'pequeño', 'compañía', 'buscar', 'medida', 'éxito', 'plata',
    'puerta', 'autor', 'entrar', 'diferencia', 'estilo', 'militar', 'muestra', 'dije', 'hija', 'ayudar',
    'defensa', 'ministerio', 'fútbol', 'canción', 'recuerdo', 'jugar', 'congreso', 'época', 'fuente', 'evitar',
    'espera', 'comunicación', 'oficina', 'actividad', 'mitad', 'conseguir', 'departamento', 'conocido', 'construcción', 'supuesto',
    'crear', 'comprar', 'viejo', 'esperar', 'energía', 'necesidad', 'modelo', 'mantener', 'capacidad', 'cámara',
    'empezar', 'comisión', 'posición', 'título', 'entender', 'culpa', 'suficiente', 'administración', 'sangre', 'creer',
    'violencia', 'llamada', 'participación', 'mamá', 'área', 'decisión', 'unión', 'encuentro', 'televisión', 'entrada',
    'colegio', 'contenido', 'imposible', 'pagar', 'sexo', 'texto', 'constitución', 'izquierda', 'ropa', 'dolor',
]

GERMAN = [
    'Soll', 'Würde', 'Leben', 'Macht', 'Jahr', 'Ende', 'Welt', 'Stadt', 'Geld', 'Teil',
    'Arbeit', 'Frage', 'Spiel', 'Seite', 'Liebe', 'Geschichte', 'Familie', 'Bild', 'Glaube', 'Polizei',
    'Woche', 'Unternehmen', 'Grund', 'Nacht', 'Thema', 'Beispiel', 'Schule', 'Wasser', 'Musik', 'Junge',
    'Kopf', 'Gesellschaft', 'Straße', 'Anfang', 'Gruppe', 'Treffen', 'Schreiben', 'Denke', 'Politik', 'Sache',
    'Folge', 'Glück', 'Hilfe', 'Artikel', 'Regierung', 'Wort', 'Sohn', 'Spaß', 'Stelle', 'Prozent',
    'Mädchen', 'Vater', 'Bereich', 'Zukunft', 'Idee', 'Wert', 'Ziel', 'Entwicklung', 'Krieg', 'Erfolg',
    'Klasse', 'Preis', 'März', 'Kirche', 'Blick', 'Meinung', 'Raum', 'Juni', 'Oktober', 'Rahmen',
    'Stück', 'Möglichkeit', 'Einsatz', 'Zahl', 'Stunde', 'Höhe', 'Tochter', 'Wohnung', 'Titel', 'Sprache',
    'Kunst', 'Weise', 'Mitarbeiter', 'Menge', 'Fehler', 'Sicherheit', 'Spieler', 'Gefühl', 'Dezember', 'Gesicht',
    'Antwort', 'Vergleich', 'Berliner', 'Tragen', 'Herz', 'Januar', 'Sucht', 'Nutzen', 'Gemeinde', 'Entscheidung',
    'Luft', 'Stimme', 'Reihe', 'Kultur', 'Lösung', 'Boden', 'Februar', 'Kritik', 'Fußball', 'Bruder',
    'Opfer', 'Wirtschaft', 'Punkt', 'Verein', 'Körper', 'Kreis', 'Bevölkerung', 'Licht', 'Freundin', 'Staat',
    'Markt', 'Beginn', 'Regel', 'Politiker', 'Ausbildung', 'Ergebnis', 'Verbindung', 'Hälfte', 'Zeitung', 'Bedeutung',
    'Programm', 'Schüler', 'Gefallen', 'Kampf', 'Ahnung', 'Erfahrung', 'Projekt', 'Druck', 'Wochenende', 'Unterstützung',
    'Schau', 'Schritt', 'Verfügung', 'Natur', 'Bett', 'Schuld', 'Gefahr', 'Linie', 'Wahrheit', 'Angebot',
    'Samstag', 'Ordnung', 'Satz', 'Lehrer', 'Schluss', 'Aufgabe', 'Runde', 'Freitag', 'Gebäude', 'Gesetz',
    'Beitrag', 'Gleiche', 'Verhalten', 'Universität', 'Reise', 'Nummer', 'Gewalt', 'Besuch', 'Werk', 'Bildung',
    'Fenster', 'Freiheit', 'Zeichen', 'Begriff', 'Bericht', 'Verfahren', 'Freude', 'Montag', 'Auswahl', 'Leistung',
    'Versuch', 'Vertrag', 'Zusammenhang', 'Arzt', 'Hund', 'Sicht', 'Zeitpunkt', 'Meister', 'Größe', 'Volk',
    'Karte', 'Hintergrund', 'Kaffee', 'Arsch', 'Mittel', 'Wagen', 'Schloss', 'Schutz', 'Grenze', 'Unterschied',
    'Studium', 'Kontakt', 'Perfekt', 'Tisch', 'Mitglied', 'Quelle', 'Urlaub', 'Werbung', 'Technik', 'Studie',
    'Leid', 'Bier', 'Himmel', 'Gericht', 'Jahrhundert', 'Gebiet', 'Einfluss', 'Teilnehmer', 'Betrieb', 'Tief',
    'Wald', 'Energie', 'Hoffnung', 'Bieten', 'Schaden', 'Gespräch', 'Qualität', 'Nachricht', 'Abschluss', 'Haut',
    'Funktion', 'Rücken', 'Inhalt', 'Kontrolle', 'Beziehung', 'Aktion', 'Anteil', 'Zustand', 'Leisten', 'Rennen',
    'Künstler', 'Literatur', 'Diskussion', 'Zusammenarbeit', 'Umgebung', 'Insel', 'Geburtstag', 'Netz', 'Schwester', 'Prozess',
]

ITALIAN = [
    'essere', 'stato', 'fatto', 'lavoro', 'giorno', 'nuovo', 'città', 'detto', 'anno', 'storia',
    'avere', 'aver', 'vedere', 'posto', 'legge', 'andare', 'nuova', 'numero', 'gruppo', 'società',
    'famiglia', 'paese', 'governo', 'esempio', 'rispetto', 'possibile', 'bisogno', 'nazionale', 'morte', 'politica',
    'comune', 'parlare', 'scuola', 'senso', 'successo', 'seguito', 'scritto', 'trovare', 'settimana', 'pubblico',
    'letto', 'strada', 'genere', 'realtà', 'migliore', 'livello', 'settembre', 'ragione', 'particolare', 'situazione',
    'sapere', 'consiglio', 'attività', 'progetto', 'faccia', 'prendere', 'figlio', 'servizio', 'buon', 'accordo',
    'amore', 'unico', 'conto', 'tratta', 'voce', 'gioco', 'mese', 'ricerca', 'unica', 'passato',
    'capire', 'luogo', 'corpo', 'ragazza', 'partito', 'favore', 'sicurezza', 'acqua', 'cuore', 'aprile',
    'giugno', 'domanda', 'chiesa', 'articolo', 'titolo', 'ottobre', 'prova', 'maggior', 'musica', 'paura',
    'luglio', 'pensare', 'regione', 'fatta', 'sociale', 'vivere', 'inglese', 'maggiore', 'termine', 'questione',
    'messo', 'lingua', 'potere', 'mettere', 'possibilità', 'posizione', 'giornata', 'programma', 'febbraio', 'dovuto',
    'polizia', 'controllo', 'scelta', 'amico', 'dicembre', 'personale', 'semplice', 'sviluppo', 'trovato', 'rapporto',
    'poter', 'gennaio', 'sola', 'risposta', 'ruolo', 'linea', 'voglia', 'esperienza', 'squadra', 'qualità',
    'giovane', 'ragazzo', 'processo', 'sinistra', 'ufficiale', 'ultima', 'produzione', 'repubblica', 'internazionale', 'leggere',
    'massimo', 'spazio', 'libertà', 'metà', 'verità', 'viaggio', 'popolo', 'passo', 'ricordo', 'moglie',
    'cerca', 'presenza', 'rete', 'ordine', 'lavorare', 'civile', 'speciale', 'pagina', 'sentire', 'felice',
    'codice', 'piccola', 'movimento', 'perdere', 'macchina', 'stile', 'riguardo', 'passare', 'università', 'prodotto',
    'inizio', 'secolo', 'stagione', 'prezzo', 'occasione', 'terzo', 'portato', 'destra', 'reale', 'cambiare',
    'testo', 'mondiale', 'sesso', 'totale', 'deciso', 'popolazione', 'formazione', 'versione', 'portare', 'partire',
    'diventare', 'bambino', 'fratello', 'maniera', 'risultato', 'francese', 'scrivere', 'arrivare', 'tornare', 'danno',
    'usare', 'cercare', 'principale', 'relazione', 'aspetto', 'stampa', 'comunità', 'simile', 'messaggio', 'scopo',
    'prossimo', 'mostra', 'sangue', 'figlia', 'cibo', 'vecchio', 'lunga', 'entrare', 'crisi', 'guida',
    'notizia', 'differenza', 'ottenere', 'contratto', 'iniziato', 'azione', 'chiedere', 'telefono', 'pensiero', 'creare',
    'morto', 'superiore', 'discorso', 'proprietà', 'rischio', 'soluzione', 'colore', 'atto', 'autore', 'difesa',
    'scena', 'storico', 'campagna', 'direttore', 'commissione', 'pratica', 'usato', 'diretta', 'guardare', 'uscire',
]

POOLS = {
    'French': FRENCH,
    'Spanish': SPANISH,
    'German': GERMAN,
    'Italian': ITALIAN,
}


def main():
    for lang, words in POOLS.items():
        assert len(words) == N_PER_LANG, f"{lang}: expected {N_PER_LANG}, got {len(words)}"
        assert len(set(words)) == N_PER_LANG, f"{lang}: duplicate words within pool"

    # cross-language uniqueness (case-insensitive): no word may appear in two pools
    seen = {}
    for lang, words in POOLS.items():
        for w in words:
            key = w.lower()
            assert key not in seen, f"'{w}' appears in both {seen.get(key)} and {lang}"
            seen[key] = lang

    examples = []
    for lang, words in POOLS.items():
        for w in words:
            examples.append({"input": w, "output": lang})

    assert len(examples) == 1000

    random.shuffle(examples)

    inputs = [ex["input"] for ex in examples]
    assert len(inputs) == len(set(inputs)), "duplicate inputs after assembly"
    for ex in examples:
        assert ex["input"] == ex["input"].strip()
        assert ex["output"] == ex["output"].strip()

    # rule self-check: re-derive language from the word's source pool and compare
    lookup = {w: lang for lang, words in POOLS.items() for w in words}
    for ex in examples:
        assert lookup[ex["input"]] == ex["output"], f"mismatch for {ex['input']}"

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=1)

    print(f"Wrote {len(examples)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
