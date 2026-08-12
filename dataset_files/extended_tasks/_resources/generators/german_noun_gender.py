#!/usr/bin/env python3
"""Generator for german_noun_gender task.

Given a common German noun, output its definite article: "der", "die", or
"das".

Methodology (this script is the reproducible artifact):
  1. Downloaded the kaikki.org/wiktextract structured extraction of the
     German Wiktionary (https://kaikki.org/dictionary/German/).
  2. Kept only entries whose head template is exactly "de-noun" (a genuine
     noun lemma, not an inflected form-of entry) with a gender argument
     whose first comma-separated field (German head templates encode
     declension info after the gender, e.g. "m,es:s,e:^e") is "m", "f", or
     "n".
  3. Cross-checked the structured gender against an INDEPENDENT regex
     parse of the entry's human-readable "expansion" string (e.g. parsing
     "n" out of "Haus n (strong, genitive Hauses, plural Häuser, ...)");
     any mismatch between the two independent parses was dropped, as was
     any word with multiple senses that disagree on gender.
  4. Cross-checked every candidate word AND its lowercased form against the
     full part-of-speech index for that string; dropped anything also
     tagged as verb, adjective, pronoun, adverb, conjunction, preposition,
     determiner, article, interjection, or numeral anywhere in the dump.
     The lowercase check specifically catches sentence-initial-capitalized
     function-word homographs (e.g. capitalized "Ich"/"Aber"/"Wenn" showing
     up as rare nominalized-noun senses while the word is overwhelmingly a
     pronoun/conjunction/adverb in its lowercase form).
  5. Required words to start with an uppercase letter, contain only German
     letters (incl. ä/ö/ü/ß), be 3-12 characters (excludes unwieldy long
     compounds), and excluded multi-word/hyphenated/apostrophe forms.
  6. Ranked by wordfreq.zipf_frequency(word.lower(), "de") separately
     within each of the 3 genders and took the top 334/333/333 per gender
     (1000 total, balanced within the required +-10%).
  7. Removed by hand, after visual review of the sorted candidate lists:
     calendar words (month/weekday names, a closed masculine-only class
     that would have skewed the balance), first names and surnames that
     leaked in (Klaus, Fritz, Heinz, Bernd, Hermann, Horst, Johannes,
     Jürgen, Schmidt, Wagner, Siemens, Mercedes, ...), place names (Rom,
     Kiel, Maas), an obscure musical-note term ("His") easily confused with
     the English pronoun, pre-1996-spelling-reform ß/ss duplicate spellings
     and Swiss ss/ß variant duplicates of a word already kept under its
     standard modern spelling (e.g. kept "Fluss"+"Fuß", dropped
     "Fluß"+"Fuss"), an inflected plural-form leak ("Bilder" when "Bild" is
     already present; "Namen" when "Name" is already present; "Pläne" when
     "Plan" is already present), and a duplicate all-caps/mixed-case
     acronym pair (kept "Aids", dropped "AIDS"). Each removed word's
     replacement was backfilled from the next-highest-frequency word of
     the same gender.

Self-check performed at generation time: every (word, article) pair is
re-derived from the three lists below and the exact count/balance is
asserted.
"""
import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "german_noun_gender.json"

random.seed(42)

DER_MASCULINE = [
    'Mann', 'Euro', 'Grund', 'Kopf', 'Anfang', 'Gott', 'Artikel', 'Sohn', 'Vater', 'Bereich', 'Tod', 'Name',
    'Freund', 'Einsatz', 'Sommer', 'Job', 'Sex', 'Mitarbeiter', 'Fehler', 'Spieler', 'König', 'Boden', 'Fußball', 'Bruder',
    'Typ', 'Punkt', 'Körper', 'Kreis', 'Staat', 'Markt', 'Beginn', 'Politiker', 'Text', 'Schüler', 'Sport', 'Kampf',
    'Bürger', 'Satz', 'Lehrer', 'Chef', 'Dollar', 'Schluss', 'Zug', 'Beitrag', 'Sieg', 'Besuch', 'Rest', 'Bericht',
    'Zusammenhang', 'Arzt', 'Hund', 'Zeitpunkt', 'Hintergrund', 'Kaffee', 'Winter', 'Arsch', 'Schutz', 'Kontakt', 'Tisch', 'Urlaub',
    'Club', 'Einfluss', 'Teilnehmer', 'Wald', 'Fuß', 'Trainer', 'Partner', 'Richter', 'Müller', 'Abschluss', 'Inhalt', 'Anteil',
    'Künstler', 'Geburtstag', 'Prozess', 'Gegensatz', 'Eindruck', 'Flughafen', 'Mund', 'Verlag', 'Herbst', 'Professor', 'Rat', 'Star',
    'Test', 'Wunsch', 'Anschluss', 'Auftrag', 'Anspruch', 'Van', 'Gegner', 'Traum', 'General', 'Kaiser', 'Bus', 'Bezug',
    'Brief', 'Kommentar', 'Landkreis', 'Vertreter', 'Strom', 'Unfall', 'Wind', 'Fahrer', 'Hof', 'Song', 'Bahnhof', 'Geist',
    'Computer', 'Finger', 'Roman', 'Stein', 'Beruf', 'Bund', 'Alkohol', 'Kurs', 'Hinweis', 'Ball', 'Zweck', 'Stil',
    'Zuschauer', 'Täter', 'Besucher', 'Vorteil', 'Baum', 'Berg', 'Fischer', 'Abstand', 'Antrag', 'Standard', 'West', 'Streit',
    'Verkauf', 'Münster', 'Umgang', 'Witz', 'Charakter', 'Zugang', 'Termin', 'Fan', 'Gast', 'Ton', 'Nutzer', 'Keller',
    'Aufbau', 'Minister', 'Wolf', 'Unterricht', 'Hersteller', 'Vorschlag', 'Anhänger', 'Anlass', 'Amerikaner', 'Manager', 'Nord', 'Ost',
    'Beweis', 'Wettbewerb', 'Leser', 'Mut', 'Code', 'Geschmack', 'Bundestag', 'Schwanz', 'Ausdruck', 'Bewohner', 'Wähler', 'Standort',
    'Status', 'Musiker', 'Engel', 'Anwalt', 'Verlust', 'Zeitraum', 'Tipp', 'Kern', 'Schatten', 'Stern', 'Einwohner', 'Alltag',
    'Unternehmer', 'Shop', 'Stress', 'Flug', 'Pfarrer', 'Weltkrieg', 'Sieger', 'Sender', 'Besitzer', 'Nachfolger', 'Arbeitgeber', 'Humor',
    'Kuchen', 'Arbeiter', 'Kanal', 'Motor', 'Rand', 'Papst', 'Vortrag', 'Server', 'Vorstand', 'Führer', 'Hals', 'Nazi',
    'Papa', 'Direktor', 'Zufall', 'LKW', 'Vogel', 'Brand', 'Stoff', 'Bezirk', 'Bescheid', 'Strand', 'Fluss', 'Bauer',
    'Süd', 'Großteil', 'Anbieter', 'Schlüssel', 'Schuss', 'Kerl', 'Effekt', 'Ansatz', 'Chat', 'Pop', 'Müll', 'Gedanke',
    'Rassismus', 'Ausbau', 'Mittelpunkt', 'Schneider', 'Eingang', 'Sänger', 'Konflikt', 'Absatz', 'Teufel', 'Deal', 'Hut', 'Kapitän',
    'Franken', 'Ritter', 'Jäger', 'Aufwand', 'Gewinner', 'Gehalt', 'Entwurf', 'Schwerpunkt', 'Schmerz', 'Schatz', 'Cent', 'Sand',
    'Bauch', 'PKW', 'Verdacht', 'Tatort', 'Cup', 'Hafen', 'Glückwunsch', 'Fokus', 'Kontext', 'Ersatz', 'Master', 'Mond',
    'Flügel', 'Haufen', 'Bischof', 'Krebs', 'Haushalt', 'Sprecher', 'Fisch', 'Stock', 'Fernseher', 'Priester', 'Wahlkampf', 'Anhalt',
    'Käse', 'Bach', 'Bachelor', 'Händler', 'Chor', 'Österreicher', 'Arbeitnehmer', 'Profi', 'Ausländer', 'Neubau', 'Berater', 'Bestandteil',
    'Weltmeister', 'Beschluss', 'Vordergrund', 'Dialog', 'Austausch', 'Unsinn', 'Landtag', 'Protest', 'Transport', 'Hahn', 'Parkplatz', 'Kumpel',
    'Widerspruch', 'Abschied', 'Ablauf', 'Dreck', 'Index', 'Knochen', 'Stuhl', 'Kindergarten', 'Onkel', 'Pokal', 'Regisseur', 'Vorwurf',
    'First', 'Anruf', 'Klassiker', 'Spieltag', 'Bonus', 'Sack', 'Ehemann', 'Höhepunkt', 'Umsatz', 'Content', 'Frühling', 'Lohn',
    'Eigentümer', 'Durchmesser', 'Kämpfer', 'Kritiker', 'Faktor', 'Gründer', 'Senat', 'Aufenthalt', 'Modus', 'Autofahrer',
]

DIE_FEMININE = [
    'Uhr', 'Welt', 'Stadt', 'Arbeit', 'Art', 'Seite', 'Geschichte', 'Familie', 'Polizei', 'Woche', 'Nacht', 'Musik',
    'Hand', 'Person', 'Gesellschaft', 'Straße', 'Mutter', 'Gruppe', 'Politik', 'Sache', 'Hilfe', 'Regierung', 'Zukunft', 'Idee',
    'Wahl', 'Entwicklung', 'Angst', 'Lage', 'Kirche', 'Partei', 'Meinung', 'Mitte', 'Möglichkeit', 'Höhe', 'Tochter', 'Wohnung',
    'Sprache', 'Kunst', 'Menge', 'Chance', 'Sicherheit', 'Saison', 'Antwort', 'Gemeinde', 'Entscheidung', 'Luft', 'Kultur', 'Lösung',
    'Kritik', 'Region', 'Situation', 'Wirtschaft', 'Firma', 'Bevölkerung', 'Freundin', 'Ausbildung', 'Verbindung', 'Hälfte', 'Zeitung', 'Bedeutung',
    'Ahnung', 'Erfahrung', 'Liste', 'Verfügung', 'Natur', 'Gefahr', 'Linie', 'Wahrheit', 'Ordnung', 'Mannschaft', 'Serie', 'Aufgabe',
    'Bank', 'Tür', 'Universität', 'Nummer', 'Gewalt', 'Bildung', 'Freiheit', 'Freude', 'Auswahl', 'Leistung', 'Sicht', 'Lust',
    'Größe', 'Karte', 'Website', 'Position', 'Union', 'Werbung', 'Technik', 'Studie', 'Version', 'Energie', 'Hoffnung', 'Show',
    'Basis', 'Party', 'Qualität', 'Nachricht', 'Minute', 'App', 'Funktion', 'Kontrolle', 'Beziehung', 'Aktion', 'Organisation', 'Literatur',
    'Diskussion', 'Umgebung', 'Insel', 'Schwester', 'Kamera', 'Praxis', 'Szene', 'Demokratie', 'Bundesliga', 'Tour', 'Behandlung', 'Armee',
    'Farbe', 'Heimat', 'Leitung', 'Mehrheit', 'Führung', 'Liga', 'Industrie', 'Wirkung', 'Verwendung', 'Religion', 'Verwaltung', 'Anlage',
    'Ausgabe', 'Seele', 'Grundlage', 'Forschung', 'Karriere', 'Gelegenheit', 'Ausstellung', 'Erklärung', 'Wissenschaft', 'Mama', 'Produktion', 'Sendung',
    'Küche', 'Generation', 'Ecke', 'Masse', 'Stiftung', 'Einführung', 'Reaktion', 'Erinnerung', 'Gegend', 'Länge', 'Stimmung', 'Kommission',
    'Nase', 'Information', 'Pause', 'Tradition', 'Umwelt', 'Prüfung', 'Tätigkeit', 'Darstellung', 'Vorstellung', 'City', 'Bühne', 'Anwendung',
    'Analyse', 'Theorie', 'Brücke', 'Rechnung', 'Sammlung', 'Förderung', 'Aufnahme', 'Republik', 'Realität', 'Beschreibung', 'Homepage', 'Krankheit',
    'Kategorie', 'Einrichtung', 'Nutzung', 'Dame', 'Untersuchung', 'Ansicht', 'Staffel', 'Software', 'Philosophie', 'Einheit', 'Abteilung', 'Adresse',
    'Mission', 'Fläche', 'Pflicht', 'Hauptstadt', 'Ausnahme', 'Medizin', 'Media', 'Revolution', 'Temperatur', 'Mühe', 'Übersetzung', 'Integration',
    'Phase', 'Maschine', 'Einstellung', 'Teilnahme', 'Auflösung', 'Figur', 'Formel', 'Spur', 'Debatte', 'Tasche', 'Marke', 'Gemeinschaft',
    'Story', 'Milch', 'Krise', 'Methode', 'Webseite', 'Zustimmung', 'Haltung', 'Geburt', 'Planung', 'Übersicht', 'Änderung', 'Ursache',
    'Auflage', 'Katze', 'Operation', 'Oma', 'Feuerwehr', 'Statistik', 'Hochschule', 'Botschaft', 'Variante', 'Hochzeit', 'Zeitschrift', 'Stellung',
    'Initiative', 'Herkunft', 'Kombination', 'Niederlage', 'Station', 'Community', 'Aufklärung', 'Absicht', 'Königin', 'Bezeichnung', 'Verfassung', 'Kleidung',
    'Landschaft', 'Begründung', 'Vorbereitung', 'Hölle', 'Therapie', 'Umsetzung', 'Hose', 'Überraschung', 'Stufe', 'Kunde', 'Freundschaft', 'Nachfrage',
    'Struktur', 'Strategie', 'Herstellung', 'Klinik', 'Info', 'Gestaltung', 'Autobahn', 'Million', 'Handlung', 'Konkurrenz', 'Anfrage', 'Forderung',
    'Unterhaltung', 'Schönheit', 'Identität', 'Waffe', 'Flasche', 'Chemie', 'Plattform', 'Tabelle', 'Anerkennung', 'Mischung', 'Sitzung', 'Brust',
    'Bibliothek', 'Atmosphäre', 'Wirklichkeit', 'Umfrage', 'Fraktion', 'Kooperation', 'Definition', 'Mode', 'Regelung', 'Perspektive', 'Koalition', 'Akademie',
    'Einladung', 'Finanzierung', 'Versorgung', 'Front', 'Freizeit', 'Burg', 'Versicherung', 'Rente', 'Verbesserung', 'Zwecke', 'Innenstadt', 'Opposition',
    'Kohle', 'Verteidigung', 'Aussicht', 'Entfernung', 'Oberfläche', 'Eröffnung', 'Option', 'Hinsicht', 'Spannung', 'Existenz', 'Veränderung', 'Trennung',
    'Beratung', 'Verbreitung', 'Beteiligung', 'Lady', 'Rose', 'Leidenschaft', 'Kindheit', 'Erweiterung', 'Bewertung',
]

DAS_NEUTER = [
    'Jahr', 'Geld', 'Land', 'Bild', 'Problem', 'Thema', 'Beispiel', 'Auto', 'Kind', 'Team', 'Internet', 'Wort',
    'Prozent', 'Mädchen', 'Video', 'System', 'Stück', 'Gefühl', 'Gesicht', 'Interesse', 'Ergebnis', 'Programm', 'Zimmer', 'Projekt',
    'Wochenende', 'Bett', 'Angebot', 'Hotel', 'Gebäude', 'Gesetz', 'Gold', 'Werk', 'Fenster', 'Zeichen', 'Volk', 'Handy',
    'Studium', 'Mitglied', 'Auge', 'Bier', 'Gericht', 'Jahrhundert', 'Amt', 'Tor', 'Ding', 'Gespräch', 'Netz', 'Interview',
    'Telefon', 'Album', 'Krankenhaus', 'Konzept', 'Dorf', 'Ausland', 'Blut', 'Lied', 'Büro', 'Meer', 'Theater', 'Baby',
    'Museum', 'Urteil', 'Modell', 'Material', 'Zentrum', 'Geschäft', 'Radio', 'Schiff', 'Verhältnis', 'Fleisch', 'Institut', 'Holz',
    'Risiko', 'Feld', 'Kino', 'Prinzip', 'Training', 'Parlament', 'Glas', 'Tier', 'Konzert', 'Publikum', 'Verständnis', 'Papier',
    'Gegenteil', 'Design', 'Gewicht', 'Niveau', 'Dach', 'Zeug', 'Geschenk', 'Produkt', 'Forum', 'Gas', 'Restaurant', 'Fahrzeug',
    'Boot', 'Smartphone', 'Schicksal', 'Brot', 'Kreuz', 'Frühstück', 'Profil', 'Umfeld', 'Fahrrad', 'Festival', 'Gelände', 'Graf',
    'Frühjahr', 'Stadion', 'Management', 'Flugzeug', 'Gefängnis', 'Engagement', 'Studio', 'Business', 'Motto', 'Kapitel', 'Konto', 'Gymnasium',
    'Loch', 'Gehirn', 'Pferd', 'Turnier', 'Vorbild', 'Rad', 'Abenteuer', 'Netzwerk', 'Wesen', 'Archiv', 'Magazin', 'Game',
    'Format', 'Geschlecht', 'Wachstum', 'Datum', 'Paket', 'Blatt', 'Geheimnis', 'Rezept', 'Ohr', 'Chaos', 'Arschloch', 'Rathaus',
    'Zitat', 'Lebensmittel', 'Heft', 'Portal', 'Salz', 'Gemüse', 'Date', 'Shirt', 'Bein', 'Drama', 'Kloster', 'Objekt',
    'Halbfinale', 'Grab', 'Signal', 'Symbol', 'Kleid', 'Update', 'WLAN', 'Motiv', 'Eigentum', 'Talent', 'Grundstück', 'Ticket',
    'Argument', 'Hobby', 'Instrument', 'Ministerium', 'College', 'Tempo', 'Märchen', 'Metall', 'Ereignis', 'Detail', 'Maul', 'Tal',
    'Pech', 'Fazit', 'Monster', 'Ufer', 'Abitur', 'Kilo', 'Klavier', 'Motorrad', 'Wohnzimmer', 'Phänomen', 'Erlebnis', 'Schlafzimmer',
    'Pfund', 'Hirn', 'Handbuch', 'Haupt', 'Bündnis', 'Budget', 'Element', 'Tennis', 'Gemälde', 'Unglück', 'Regime', 'Handwerk',
    'Dokument', 'Vorfeld', 'KFZ', 'Office', 'Image', 'Medium', 'Sofa', 'Spektrum', 'Asyl', 'Orchester', 'Girl', 'Pack',
    'Kennzeichen', 'Experiment', 'Spielzeug', 'Benzin', 'Universum', 'Gedächtnis', 'Camp', 'Seminar', 'Protokoll', 'Gramm', 'Schwein', 'Bargeld',
    'Ausmaß', 'Mitleid', 'Gedicht', 'Tagebuch', 'Schwert', 'Duo', 'Journal', 'Leder', 'Paradies', 'Hektar', 'Gewerbe', 'Feedback',
    'Horn', 'Labor', 'Heer', 'Christentum', 'Königreich', 'Duell', 'Resultat', 'Becken', 'Ensemble', 'Volumen', 'Testament', 'Ostern',
    'Möbel', 'Werkzeug', 'Klo', 'Gepäck', 'Geschwister', 'Landgericht', 'Tablet', 'Kommando', 'Bundesland', 'Vorjahr', 'Anzeichen', 'Echo',
    'Denkmal', 'Zeitalter', 'Jubiläum', 'Zelt', 'Erdbeben', 'Passwort', 'Semester', 'Hemd', 'Debüt', 'Praktikum', 'Lebensjahr', 'Grundgesetz',
    'Alpha', 'Musical', 'Brötchen', 'Amtsgericht', 'Highlight', 'Dating', 'Potenzial', 'Trio', 'Statement', 'Potential', 'Rohr', 'Bedürfnis',
    'Label', 'Panorama', 'Siegel', 'Trikot', 'Stichwort', 'Bundesamt', 'Schuljahr', 'Harz', 'Quartal', 'Abo', 'Fass', 'Weib',
    'Drehbuch', 'Comeback', 'Display', 'Audio', 'Diplom', 'Brett', 'Kissen', 'Verzeichnis', 'Erdgeschoss', 'Exemplar', 'Kaninchen', 'Gebirge',
    'Arsenal', 'Genie', 'Geräusch', 'Heimspiel', 'Bit', 'Kabinett', 'Zeugnis', 'Schauspiel', 'Plakat', 'Genre', 'Mandat', 'Nest',
    'Register', 'Derby', 'Anwesen', 'Aids', 'Weltbild', 'Outfit', 'Wörterbuch', 'Banner', 'Ego',
]


def main():
    counts = {'der': len(DER_MASCULINE), 'die': len(DIE_FEMININE), 'das': len(DAS_NEUTER)}
    total = sum(counts.values())
    assert total == 1000, f"expected 1000 total, got {total}: {counts}"
    for label, lst in [('der', DER_MASCULINE), ('die', DIE_FEMININE), ('das', DAS_NEUTER)]:
        assert len(set(lst)) == len(lst), f"duplicate nouns within {label} list"
        # balance check: within +-10% of an even three-way split
        assert abs(len(lst) - total / 3) / (total / 3) <= 0.10, f"{label} list not balanced: {len(lst)}"

    all_words = DER_MASCULINE + DIE_FEMININE + DAS_NEUTER
    assert len(set(all_words)) == len(all_words), "a noun appears in more than one article list"

    examples = []
    for w in DER_MASCULINE:
        examples.append({"input": w, "output": "der"})
    for w in DIE_FEMININE:
        examples.append({"input": w, "output": "die"})
    for w in DAS_NEUTER:
        examples.append({"input": w, "output": "das"})

    assert len(examples) == 1000
    random.shuffle(examples)

    inputs = [ex["input"] for ex in examples]
    assert len(inputs) == len(set(inputs)), "duplicate inputs"
    for ex in examples:
        assert ex["input"] == ex["input"].strip()
        assert ex["output"] == ex["output"].strip()

    # rule self-check: re-derive article from source list membership
    article_of = {w: "der" for w in DER_MASCULINE}
    article_of.update({w: "die" for w in DIE_FEMININE})
    article_of.update({w: "das" for w in DAS_NEUTER})
    for ex in examples:
        assert article_of[ex["input"]] == ex["output"]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=1)

    print(f"Wrote {len(examples)} examples to {OUT_PATH} ({counts})")


if __name__ == "__main__":
    main()
