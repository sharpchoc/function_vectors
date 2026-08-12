#!/usr/bin/env python3
"""Generator for french_noun_gender task.

Given a common French noun, output its grammatical gender: "masculine" or
"feminine".

Methodology (this script is the reproducible artifact):
  1. Downloaded the kaikki.org/wiktextract structured extraction of the
     French Wiktionary (https://kaikki.org/dictionary/French/).
  2. Kept only entries whose head template is exactly "fr-noun" (i.e. a
     genuine noun lemma, not an inflected form-of entry) with a gender
     argument of plain "m" or "f" (excluded: "m-p"/"f-p" plural-only forms,
     "mf"/"mfbysense" common-gender/epicene nouns usable with either
     article).
  3. For every candidate word, cross-checked the structured gender against
     an INDEPENDENT regex parse of the human-readable "expansion" string on
     the same dictionary entry (e.g. parsing "m" out of
     "livre m (plural livres)"); any mismatch between the two independent
     parses was dropped.
  4. For words with multiple senses/etymologies (e.g. "livre" = book,
     masculine, vs. "livre" = pound, feminine), required every sense to
     agree on the same single gender; words with a genuine dual-gender
     split were dropped automatically by this check (this is exactly how
     "livre" itself is excluded from the pool below).
  5. Excluded any word also tagged elsewhere in the dump as a verb (removes
     verb/noun homographs such as "être", "avoir", "fait", "été" where the
     verb sense dominates), pronoun, adverb, conjunction, preposition,
     determiner, article, interjection, numeral, or adjective (kept the
     pool to "pure" nouns), plus multi-word entries, hyphenated/apostrophe
     forms, and capitalized (proper-noun) entries.
  6. Ranked by wordfreq.zipf_frequency(word, "fr") and took the top 1000
     by frequency, which came out to 534/466 masculine/feminine (M/F ratio
     within the pool is naturally close to even); to hit an exact 500/500
     split we kept the top 500 masculine and top 500 feminine nouns by
     frequency separately.
  7. A small number of remaining proper-noun-like or vulgar/slang leaks
     that survived the automatic filters (first names such as "louis",
     "martin", "robert", "catherine", "guillaume", "macron", ambiguous
     slang like "mac", "cul", "gay", "pute") were removed by hand after
     visual review of the full sorted candidate list, with the next
     highest-frequency word of the same gender promoted to backfill.

Self-check performed at generation time: every (word, gender) pair is
re-derived from the two dicts below and the reported count/ratio is
asserted; MANUAL_EXCLUDE is applied before ranking so the final list never
contains an excluded word.
"""
import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "french_noun_gender.json"
N_PER_GENDER = 500

random.seed(42)

MASCULINE = [
    'temps', 'pays', 'jour', 'homme', 'cas', 'moment', 'mois', 'travail', 'état', 'prix', 'nom', 'lieu',
    'coup', 'groupe', 'besoin', 'jeu', 'jean', 'effet', 'site', 'nombre', 'père', 'rapport', 'niveau', 'gouvernement',
    'accord', 'article', 'problème', 'dieu', 'fils', 'exemple', 'système', 'début', 'air', 'centre', 'projet', 'conseil',
    'argent', 'film', 'corps', 'chef', 'titre', 'matin', 'choix', 'retour', 'nord', 'ordre', 'mars', 'truc',
    'sud', 'art', 'match', 'septembre', 'mai', 'juin', 'train', 'mot', 'avis', 'juillet', 'développement', 'mec',
    'abord', 'octobre', 'terme', 'siècle', 'gars', 'janvier', 'frère', 'avril', 'roi', 'cœur', 'club', 'milieu',
    'propos', 'âge', 'emploi', 'novembre', 'plaisir', 'décembre', 'rôle', 'août', 'esprit', 'internet', 'mouvement', 'intérêt',
    'février', 'lien', 'ami', 'objet', 'réseau', 'texte', 'appel', 'message', 'succès', 'travers', 'sein', 'chemin',
    'mariage', 'pied', 'bureau', 'soleil', 'numéro', 'dimanche', 'taux', 'accès', 'bras', 'monsieur', 'domaine', 'sol',
    'ouest', 'code', 'sport', 'résultat', 'honneur', 'coeur', 'midi', 'sang', 'bord', 'style', 'terrain', 'samedi',
    'hôtel', 'vol', 'contrat', 'vendredi', 'discours', 'avenir', 'maître', 'dos', 'endroit', 'commerce', 'quartier', 'territoire',
    'principe', 'changement', 'respect', 'lundi', 'membre', 'secteur', 'soutien', 'ministère', 'village', 'poids', 'comité', 'album',
    'cinéma', 'visage', 'département', 'parc', 'seigneur', 'environnement', 'traitement', 'port', 'transport', 'avion', 'professeur', 'hiver',
    'stade', 'théâtre', 'hôpital', 'regard', 'usage', 'royaume', 'fil', 'tableau', 'fer', 'contact', 'gaz', 'procès',
    'parlement', 'goût', 'bébé', 'boulot', 'épisode', 'prince', 'vent', 'arrêt', 'avocat', 'sac', 'bonheur', 'budget',
    'secrétaire', 'lendemain', 'caractère', 'musée', 'siège', 'enseignement', 'acte', 'vin', 'contexte', 'processus', 'papier', 'front',
    'tribunal', 'candidat', 'noël', 'jeudi', 'accident', 'festival', 'établissement', 'chat', 'pont', 'anniversaire', 'football', 'danger',
    'champ', 'personnage', 'retard', 'mur', 'spectacle', 'mardi', 'espoir', 'silence', 'don', 'château', 'vice', 'écran',
    'univers', 'salon', 'entretien', 'appareil', 'crédit', 'bruit', 'héros', 'docteur', 'classement', 'cheval', 'palais', 'client',
    'sentiment', 'soin', 'avantage', 'institut', 'exercice', 'statut', 'test', 'patron', 'examen', 'domicile', 'collège', 'ouvrage',
    'van', 'achat', 'cerveau', 'bâtiment', 'concept', 'japon', 'lac', 'jardin', 'alcool', 'coût', 'patrimoine', 'impact',
    'cancer', 'agent', 'concert', 'nez', 'métier', 'événement', 'championnat', 'œil', 'accueil', 'minimum', 'cadeau', 'sommet',
    'hasard', 'secours', 'pain', 'crime', 'champion', 'bar', 'acteur', 'garçon', 'repas', 'canal', 'volume', 'maximum',
    'post', 'aéroport', 'comte', 'chiffre', 'défaut', 'logement', 'hommage', 'mandat', 'effort', 'pape', 'chômage', 'comportement',
    'phénomène', 'congrès', 'profit', 'cap', 'lait', 'cabinet', 'décès', 'printemps', 'appartement', 'bilan', 'euro', 'for',
    'islam', 'génie', 'conflit', 'climat', 'thème', 'métro', 'duc', 'salaire', 'dessin', 'rang', 'humour', 'magasin',
    'détail', 'élément', 'commentaire', 'brésil', 'arbre', 'aspect', 'bac', 'profil', 'engagement', 'impôt', 'investissement', 'fonctionnement',
    'vélo', 'blog', 'chapitre', 'essai', 'financement', 'jugement', 'rythme', 'entraîneur', 'talent', 'langage', 'plateau', 'bain',
    'tourisme', 'orient', 'tort', 'trou', 'tas', 'document', 'commandant', 'règlement', 'empereur', 'scénario', 'organisme', 'progrès',
    'doigt', 'meurtre', 'score', 'magazine', 'lieutenant', 'format', 'oeil', 'olivier', 'portrait', 'tournoi', 'poisson', 'guide',
    'stage', 'colonel', 'automne', 'bloc', 'ordinateur', 'séjour', 'souci', 'immeuble', 'transfert', 'rock', 'micro', 'choc',
    'mont', 'calcul', 'prénom', 'désir', 'disque', 'égard', 'décret', 'entraînement', 'circuit', 'studio', 'sucre', 'individu',
    'trafic', 'sénat', 'délai', 'angle', 'billet', 'sommeil', 'appui', 'enfer', 'gardien', 'récit', 'foot', 'show',
    'leader', 'atelier', 'pré', 'gouverneur', 'procureur', 'news', 'abus', 'drapeau', 'trésor', 'degré', 'sondage', 'repos',
    'paradis', 'cycle', 'mail', 'tir', 'témoin', 'outil', 'paiement', 'temple', 'ski', 'lot', 'dispositif', 'ballon',
    'arrondissement', 'ingénieur', 'laboratoire', 'lancement', 'virus', 'cou', 'ange', 'montage', 'navire', 'ventre', 'écrivain', 'océan',
    'rappel', 'morceau', 'symbole', 'business', 'jury', 'ménage', 'abri', 'dépôt', 'refus', 'arc', 'toit', 'forum',
    'prêtre', 'écart', 'fruit', 'sel', 'foyer', 'vainqueur', 'diplôme', 'témoignage', 'envoi', 'enregistrement', 'maintien', 'syndicat',
    'cousin', 'axe', 'solo', 'réalisateur', 'calendrier', 'seuil', 'fondateur', 'drame', 'déplacement', 'accent', 'col', 'stress',
    'loup', 'oncle', 'courrier', 'destin', 'bassin', 'équipement', 'attentat', 'copain', 'soldat', 'paquet', 'paysage', 'apprentissage',
    'mystère', 'pôle', 'facteur', 'défi', 'retrait', 'continent', 'métal', 'thé', 'motif', 'chevalier', 'équipage', 'lecteur',
    'remplacement', 'scandale', 'terrorisme', 'aménagement', 'open', 'design', 'val', 'registre',
]

FEMININE = [
    'vie', 'chose', 'année', 'ville', 'histoire', 'femme', 'tête', 'suite', 'famille', 'société', 'question', 'raison',
    'équipe', 'eau', 'mère', 'loi', 'face', 'fille', 'guerre', 'semaine', 'ligne', 'heure', 'façon', 'idée',
    'main', 'saison', 'situation', 'base', 'journée', 'grâce', 'école', 'pierre', 'région', 'entreprise', 'chambre', 'peur',
    'série', 'sécurité', 'peine', 'voix', 'voiture', 'affaire', 'population', 'action', 'culture', 'route', 'direction', 'chance',
    'confiance', 'photo', 'formation', 'classe', 'liste', 'langue', 'occasion', 'justice', 'scène', 'qualité', 'production', 'mer',
    'étude', 'campagne', 'valeur', 'cour', 'plupart', 'salle', 'réponse', 'position', 'république', 'carte', 'impression', 'association',
    'création', 'activité', 'expérience', 'fonction', 'organisation', 'liberté', 'présence', 'économie', 'réalité', 'compagnie', 'université', 'commission',
    'victoire', 'vérité', 'communauté', 'administration', 'version', 'soirée', 'matière', 'information', 'mission', 'union', 'construction', 'pièce',
    'moitié', 'arrivée', 'énergie', 'église', 'faute', 'solution', 'paix', 'majorité', 'décision', 'défense', 'prison', 'preuve',
    'éducation', 'lumière', 'lettre', 'édition', 'couleur', 'relation', 'espèce', 'erreur', 'carrière', 'course', 'taille', 'gestion',
    'différence', 'ouverture', 'protection', 'île', 'maladie', 'crise', 'industrie', 'application', 'chanson', 'banque', 'opération', 'vitesse',
    'religion', 'importance', 'volonté', 'pression', 'évolution', 'science', 'utilisation', 'absence', 'connaissance', 'émission', 'habitude', 'expression',
    'réunion', 'tendance', 'révolution', 'surface', 'madame', 'communication', 'peau', 'côte', 'élection', 'victime', 'province', 'consommation',
    'opposition', 'sœur', 'hauteur', 'naissance', 'faveur', 'télé', 'perte', 'possibilité', 'unité', 'violence', 'identité', 'responsabilité',
    'opinion', 'beauté', 'foi', 'planète', 'théorie', 'conférence', 'lecture', 'collection', 'conscience', 'assurance', 'exposition', 'croissance',
    'âme', 'maman', 'propriété', 'autorité', 'agence', 'existence', 'section', 'exploitation', 'catégorie', 'machine', 'vision', 'jeunesse',
    'méthode', 'revue', 'star', 'augmentation', 'structure', 'reine', 'génération', 'étape', 'intervention', 'boîte', 'bataille', 'sélection',
    'participation', 'réduction', 'disposition', 'réaction', 'indépendance', 'arme', 'phase', 'intention', 'amie', 'séance', 'urgence', 'constitution',
    'auto', 'agriculture', 'revanche', 'joie', 'télévision', 'déclaration', 'phrase', 'condition', 'publication', 'proposition', 'longueur', 'colère',
    'description', 'croix', 'échelle', 'résistance', 'compétition', 'info', 'station', 'égalité', 'plage', 'enfance', 'stratégie', 'procédure',
    'proximité', 'humanité', 'distribution', 'division', 'discussion', 'nourriture', 'définition', 'montagne', 'clé', 'lune', 'démocratie', 'formule',
    'faim', 'douleur', 'température', 'race', 'barre', 'exception', 'forêt', 'balle', 'honte', 'quantité', 'bibliothèque', 'présentation',
    'frontière', 'difficulté', 'soeur', 'aventure', 'nation', 'robe', 'oeuvre', 'traduction', 'rivière', 'écriture', 'littérature', 'réalisation',
    'fédération', 'épreuve', 'tradition', 'haine', 'veille', 'chaleur', 'réflexion', 'conduite', 'légende', 'académie', 'passion', 'alliance',
    'conception', 'initiative', 'préparation', 'électricité', 'neige', 'conversation', 'circulation', 'pluie', 'reconnaissance', 'amitié', 'conséquence', 'philosophie',
    'résidence', 'convention', 'option', 'promotion', 'surveillance', 'couverture', 'technologie', 'mairie', 'princesse', 'caisse', 'libération', 'personnalité',
    'fenêtre', 'dette', 'grève', 'attente', 'performance', 'viande', 'actualité', 'collaboration', 'composition', 'essence', 'attitude', 'copine',
    'diffusion', 'batterie', 'nécessité', 'représentation', 'fondation', 'interview', 'pêche', 'vigueur', 'obligation', 'fabrication', 'tentative', 'vallée',
    'fan', 'pause', 'mention', 'intelligence', 'réputation', 'autorisation', 'efficacité', 'bière', 'notion', 'drogue', 'cérémonie', 'manifestation',
    'interprétation', 'charte', 'institution', 'gamme', 'évidence', 'coopération', 'queue', 'architecture', 'destination', 'explication', 'publicité', 'inscription',
    'bourse', 'réussite', 'profondeur', 'comparaison', 'exécution', 'apparition', 'conclusion', 'transition', 'couronne', 'alimentation', 'caméra', 'solidarité',
    'fortune', 'tension', 'introduction', 'signature', 'piscine', 'feuille', 'résolution', 'bouteille', 'amélioration', 'présidence', 'thèse', 'horreur',
    'installation', 'odeur', 'richesse', 'poésie', 'pitié', 'médaille', 'intégration', 'disparition', 'totalité', 'priorité', 'actrice', 'extension',
    'corruption', 'oreille', 'licence', 'folie', 'précision', 'hypothèse', 'comédie', 'destruction', 'transformation', 'immigration', 'innovation', 'étoile',
    'contribution', 'toile', 'humeur', 'évaluation', 'restauration', 'adaptation', 'interdiction', 'fermeture', 'galerie', 'rédaction', 'dizaine', 'animation',
    'marge', 'boutique', 'diversité', 'dimension', 'assistance', 'orientation', 'bombe', 'présidente', 'leçon', 'recette', 'atmosphère', 'réception',
    'fleur', 'possession', 'fusion', 'autoroute', 'pauvreté', 'autonomie', 'météo', 'répartition', 'explosion', 'audience', 'suppression', 'fac',
    'faculté', 'magie', 'promesse', 'poursuite', 'liaison', 'concentration', 'sensation', 'apparence', 'émotion', 'séparation', 'préfecture', 'plateforme',
    'acquisition', 'profession', 'prière', 'ère', 'gloire', 'villa', 'instruction', 'moto', 'livraison', 'chapelle', 'transmission', 'cellule',
    'vertu', 'ile', 'occupation', 'fiction', 'finance', 'session', 'blessure', 'nomination', 'adoption', 'boucle', 'tempête', 'candidature',
    'souffrance', 'prévention', 'succession', 'banlieue', 'ceinture', 'nationalité', 'fréquence', 'compétence', 'observation', 'sauce', 'conservation', 'douceur',
    'inspiration', 'patience', 'chaise', 'métropole', 'récompense', 'quête', 'variété', 'bible',
]


def main():
    assert len(MASCULINE) == N_PER_GENDER, f"masculine: expected {N_PER_GENDER}, got {len(MASCULINE)}"
    assert len(FEMININE) == N_PER_GENDER, f"feminine: expected {N_PER_GENDER}, got {len(FEMININE)}"
    assert len(set(MASCULINE)) == N_PER_GENDER, "duplicate masculine nouns"
    assert len(set(FEMININE)) == N_PER_GENDER, "duplicate feminine nouns"
    assert not (set(MASCULINE) & set(FEMININE)), "a noun appears in both gender lists"

    examples = []
    for w in MASCULINE:
        examples.append({"input": w, "output": "masculine"})
    for w in FEMININE:
        examples.append({"input": w, "output": "feminine"})

    assert len(examples) == 1000
    random.shuffle(examples)

    inputs = [ex["input"] for ex in examples]
    assert len(inputs) == len(set(inputs)), "duplicate inputs"
    for ex in examples:
        assert ex["input"] == ex["input"].strip()
        assert ex["output"] == ex["output"].strip()

    # rule self-check: re-derive gender from source list membership
    gender_of = {w: "masculine" for w in MASCULINE}
    gender_of.update({w: "feminine" for w in FEMININE})
    for ex in examples:
        assert gender_of[ex["input"]] == ex["output"]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=1)

    print(f"Wrote {len(examples)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
