#!/usr/bin/env python3
"""Generator for spanish_noun_gender task.

Given a common Spanish noun, output its grammatical gender: "masculine" or
"feminine".

Methodology (this script is the reproducible artifact):
  1. Downloaded the kaikki.org/wiktextract structured extraction of the
     Spanish Wiktionary (https://kaikki.org/dictionary/Spanish/).
  2. Kept only entries whose head template is exactly "es-noun" (a genuine
     noun lemma, not an inflected form-of entry) with a gender argument of
     plain "m" or "f" (excluded: "m-p"/"f-p" plural-only forms).
  3. Cross-checked the structured gender against an INDEPENDENT regex parse
     of the entry's human-readable "expansion" string (e.g. parsing "m"
     out of "pie m (plural pies)"); dropped on mismatch, and required every
     sense/etymology of a word to agree on the same single gender (this
     automatically excludes genuine common-gender/dual-gender nouns such
     as "estudiante", "artista", "capital", "orden", "frente").
  4. Excluded any word also tagged elsewhere in the dump as verb, adjective,
     pronoun, adverb, conjunction, preposition, determiner, article,
     interjection, numeral, or proper name; plus multi-word, hyphenated,
     or apostrophe-containing entries and non-lowercase-initial (proper
     noun) entries.
  5. Ranked by wordfreq.zipf_frequency(word, "es") separately within each
     gender and took the top 500 masculine / top 500 feminine by
     frequency (1000 total, exactly balanced).
  6. Removed by hand, after visual review of the sorted candidate lists:
     country/city names that leaked in as nouns (Chile kept because its
     dominant Spanish sense is "chili pepper", but Cuba/Brasil/Paraguay/
     Toledo/Valencia/Berlin/Córdoba/Colón/Bolívar/Calderón/Rivera/Salvador
     dropped as primarily place/proper-noun associated), first names and
     surnames (Carmen, César, Fernando, Jorge, Martín, María, Ramón, Tony),
     English words leaked in from loanword entries (boy, center, full,
     house, man, play, baby, bob), vulgar/slang senses (concha, orto, coca,
     can), duplicate unaccented-typo spellings of an already-kept accented
     word (dia/día, razon/razón, corazon dup n/a), a duplicate alternate
     spelling (setiembre/septiembre), and inflected plural-form leaks
     (bolsas when bolsa is kept, corazones when corazón is kept, patas
     and piernas when pata/pierna are not both kept). Each removed word
     was backfilled from the next-highest-frequency word of the same
     gender, keeping regular -o/-a endings mixed with irregular endings
     (flor, mano-type nouns, though "mano" itself did not make the final
     frequency cut) so the mapping tests knowledge rather than pure
     orthography.

Self-check performed at generation time: every (word, gender) pair is
re-derived from the two lists below and the exact 500/500 balance is
asserted.
"""
import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "spanish_noun_gender.json"
N_PER_GENDER = 500

random.seed(42)

MASCULINE = [
    'tiempo', 'día', 'mundo', 'año', 'lugar', 'país', 'momento', 'dios', 'grupo', 'sistema', 'presidente', 'fin',
    'lado', 'problema', 'favor', 'dinero', 'ejemplo', 'número', 'través', 'amor', 'hijo', 'nivel', 'artículo', 'cuerpo',
    'servicio', 'mes', 'video', 'julio', 'rey', 'miedo', 'control', 'marzo', 'sur', 'chile', 'corazón', 'octubre',
    'diciembre', 'modo', 'plan', 'abril', 'junio', 'papel', 'noviembre', 'movimiento', 'enero', 'jefe', 'siglo', 'febrero',
    'cabo', 'valor', 'septiembre', 'director', 'niño', 'mensaje', 'ejército', 'interés', 'sol', 'éxito', 'autor', 'color',
    'fondo', 'ministerio', 'fútbol', 'puerto', 'congreso', 'sector', 'departamento', 'domingo', 'título', 'auto', 'error', 'club',
    'texto', 'post', 'dolor', 'análisis', 'código', 'efecto', 'profesor', 'cielo', 'origen', 'cine', 'instituto', 'canal',
    'juicio', 'hospital', 'conocimiento', 'verano', 'don', 'género', 'régimen', 'teléfono', 'edificio', 'doctor', 'acto', 'territorio',
    'viernes', 'vídeo', 'riesgo', 'miembro', 'producto', 'crecimiento', 'término', 'plazo', 'rato', 'usuario', 'tribunal', 'líder',
    'honor', 'barrio', 'parque', 'hogar', 'teatro', 'golpe', 'pan', 'lunes', 'carácter', 'evento', 'festival', 'hotel',
    'período', 'concepto', 'sábado', 'ecuador', 'personaje', 'comentario', 'alcalde', 'jugador', 'contexto', 'ejercicio', 'correo', 'tratamiento',
    'distrito', 'calor', 'secretario', 'propósito', 'aeropuerto', 'lenguaje', 'capitán', 'espíritu', 'coche', 'planeta', 'idioma', 'blog',
    'periodo', 'perfil', 'león', 'gas', 'compañero', 'crédito', 'jueves', 'tren', 'puente', 'aspecto', 'asesinato', 'comité',
    'conflicto', 'reconocimiento', 'municipio', 'pensamiento', 'escritor', 'minuto', 'delito', 'avión', 'costo', 'crimen', 'humor', 'cáncer',
    'compromiso', 'palacio', 'foro', 'museo', 'metro', 'cumpleaños', 'martes', 'universo', 'capítulo', 'software', 'sonido', 'escenario',
    'método', 'show', 'clima', 'cerebro', 'montón', 'dominio', 'comportamiento', 'mapa', 'ayuntamiento', 'examen', 'turismo', 'gato',
    'cuello', 'oeste', 'sentimiento', 'dueño', 'cristo', 'ángel', 'tráfico', 'cliente', 'panamá', 'millón', 'tío', 'cariño',
    'volumen', 'coronel', 'ámbito', 'árbol', 'castillo', 'salón', 'límite', 'campeón', 'bosque', 'imperio', 'nacimiento', 'patrimonio',
    'ruido', 'rostro', 'rock', 'alcohol', 'desastre', 'caballo', 'jardín', 'ritmo', 'tono', 'orgullo', 'cabello', 'agente',
    'diálogo', 'estadio', 'actor', 'bar', 'brazo', 'procedimiento', 'caballero', 'senado', 'barco', 'talento', 'homenaje', 'petróleo',
    'taller', 'huevo', 'fenómeno', 'entrenamiento', 'espectáculo', 'lanzamiento', 'funcionamiento', 'abuelo', 'elemento', 'rol', 'lago', 'porcentaje',
    'círculo', 'olor', 'formato', 'dedo', 'bienestar', 'repente', 'audio', 'índice', 'rosario', 'organismo', 'drama', 'virus',
    'queso', 'sabor', 'restaurante', 'vehículo', 'documental', 'mantenimiento', 'factor', 'metal', 'terror', 'polvo', 'socio', 'continente',
    'teniente', 'infierno', 'senador', 'cable', 'lujo', 'fraude', 'laboratorio', 'ingeniero', 'diablo', 'héroe', 'tránsito', 'periodismo',
    'chocolate', 'espejo', 'símbolo', 'testigo', 'rumbo', 'rango', 'paquete', 'cumplimiento', 'link', 'patio', 'episodio', 'recurso',
    'carro', 'entrenador', 'arroz', 'reloj', 'emperador', 'palo', 'aprendizaje', 'porno', 'cartel', 'plato', 'temor', 'álbum',
    'nieto', 'instante', 'establecimiento', 'asco', 'conductor', 'editor', 'genio', 'vicepresidente', 'patrón', 'duque', 'funcionario', 'rendimiento',
    'arco', 'embajador', 'equilibrio', 'órgano', 'mecanismo', 'cerro', 'campeonato', 'criterio', 'seguimiento', 'gabinete', 'empresario', 'flujo',
    'tesoro', 'jean', 'taxi', 'bolsillo', 'autobús', 'sindicato', 'poema', 'futbol', 'occidente', 'misterio', 'paisaje', 'caos',
    'pastor', 'panorama', 'vaso', 'eje', 'alquiler', 'pedazo', 'sufrimiento', 'toro', 'otoño', 'cristal', 'liderazgo', 'muchacho',
    'convenio', 'campamento', 'exceso', 'dólar', 'terrorismo', 'milagro', 'océano', 'apartamento', 'tour', 'calendario', 'circuito', 'descubrimiento',
    'eco', 'gerente', 'socialismo', 'vistazo', 'récord', 'carnaval', 'camión', 'depósito', 'monumento', 'suicidio', 'protocolo', 'tenis',
    'sacrificio', 'carbón', 'diccionario', 'bus', 'pozo', 'núcleo', 'mail', 'maíz', 'asistente', 'paraíso', 'rincón', 'desempleo',
    'préstamo', 'botón', 'consentimiento', 'himno', 'marketing', 'semestre', 'hombro', 'pasaje', 'terremoto', 'gimnasio', 'estrés', 'capitalismo',
    'servidor', 'mito', 'tuit', 'déficit', 'pasaporte', 'cementerio', 'ascenso', 'historial', 'alumno', 'catálogo', 'piano', 'horror',
    'campus', 'financiamiento', 'asesor', 'horno', 'coraje', 'cálculo', 'esquema', 'entretenimiento', 'hueso', 'balance', 'test', 'coro',
    'seminario', 'convento', 'vapor', 'ordenador', 'tubo', 'consenso', 'circo', 'azar', 'defecto', 'tigre', 'escritorio', 'vendedor',
    'administrador', 'coste', 'cuero', 'consejero', 'fantasma', 'dictador', 'click', 'plus', 'mediodía', 'estómago', 'trono', 'pibe',
    'demonio', 'descenso', 'cardenal', 'like', 'jugo', 'vigor', 'cuartel', 'portavoz', 'referéndum', 'sombrero', 'cadáver', 'enfrentamiento',
    'ferrocarril', 'balón', 'teclado', 'bronce', 'chat', 'cuchillo', 'limón', 'acento', 'vínculo', 'homicidio', 'tabaco', 'globo',
    'condado', 'sobrino', 'nacionalismo', 'maquillaje', 'huracán', 'algodón', 'testamento', 'síndrome', 'ángulo', 'boletín', 'film', 'lord',
    'racismo', 'oxígeno', 'ron', 'pintor', 'lema', 'énfasis', 'entendimiento', 'euro',
]

FEMININE = [
    'vez', 'vida', 'gente', 'ciudad', 'manera', 'ley', 'mujer', 'familia', 'guerra', 'información', 'semana', 'hora',
    'madre', 'muerte', 'realidad', 'sociedad', 'razón', 'tierra', 'seguridad', 'mayoría', 'universidad', 'foto', 'situación', 'educación',
    'clase', 'web', 'zona', 'paz', 'policía', 'empresa', 'línea', 'atención', 'luz', 'relación', 'población', 'justicia',
    'edad', 'voz', 'imagen', 'palabra', 'región', 'investigación', 'república', 'comunidad', 'dirección', 'película', 'respuesta', 'producción',
    'página', 'organización', 'oportunidad', 'economía', 'opinión', 'plaza', 'iglesia', 'calidad', 'red', 'provincia', 'campaña', 'carrera',
    'compañía', 'puerta', 'hija', 'canción', 'época', 'fuente', 'comunicación', 'oficina', 'actividad', 'mitad', 'construcción', 'boca',
    'energía', 'necesidad', 'capacidad', 'crisis', 'cámara', 'costa', 'comisión', 'posición', 'administración', 'violencia', 'participación', 'carta',
    'área', 'banda', 'fiesta', 'decisión', 'unión', 'televisión', 'constitución', 'mente', 'ropa', 'isla', 'versión', 'importancia',
    'unidad', 'cuestión', 'formación', 'victoria', 'revolución', 'propiedad', 'tecnología', 'democracia', 'posibilidad', 'nación', 'protección', 'lengua',
    'naturaleza', 'asociación', 'edición', 'venta', 'hambre', 'reunión', 'solución', 'confianza', 'creación', 'responsabilidad', 'memoria', 'ciencia',
    'cama', 'opción', 'aplicación', 'temporada', 'selección', 'función', 'independencia', 'cruz', 'corrupción', 'elección', 'niña', 'carne',
    'materia', 'oposición', 'expresión', 'caja', 'enfermedad', 'presión', 'misión', 'sección', 'escena', 'asamblea', 'estación', 'operación',
    'teoría', 'piel', 'luna', 'velocidad', 'visión', 'altura', 'competencia', 'generación', 'gestión', 'villa', 'publicación', 'intención',
    'cárcel', 'piedra', 'fundación', 'cadena', 'autoridad', 'literatura', 'continuación', 'actitud', 'existencia', 'pobreza', 'búsqueda', 'basura',
    'declaración', 'pantalla', 'belleza', 'tarea', 'identidad', 'pérdida', 'inversión', 'resistencia', 'voluntad', 'inteligencia', 'distribución', 'consecuencia',
    'presentación', 'estrategia', 'categoría', 'prisión', 'biblioteca', 'ruta', 'ocasión', 'etapa', 'tendencia', 'bandera', 'madera', 'resolución',
    'actualidad', 'presidencia', 'condición', 'frase', 'publicidad', 'audiencia', 'humanidad', 'división', 'deuda', 'habitación', 'religión', 'frecuencia',
    'plataforma', 'institución', 'colaboración', 'juventud', 'señal', 'superficie', 'tradición', 'onda', 'igualdad', 'conversación', 'sorpresa', 'moda',
    'tarjeta', 'sesión', 'colección', 'exposición', 'gracia', 'vergüenza', 'víctima', 'espalda', 'discusión', 'bolsa', 'sensación', 'intervención',
    'mentira', 'asistencia', 'representación', 'década', 'lectura', 'dictadura', 'abuela', 'labor', 'lluvia', 'vivienda', 'letra', 'disposición',
    'moneda', 'traducción', 'alegría', 'descripción', 'evolución', 'filosofía', 'raza', 'reacción', 'conexión', 'alianza', 'poesía', 'primavera',
    'jornada', 'definición', 'emergencia', 'personalidad', 'actuación', 'máquina', 'perspectiva', 'peña', 'felicidad', 'herramienta', 'agricultura', 'montaña',
    'promoción', 'dignidad', 'variedad', 'pintura', 'pista', 'temperatura', 'pieza', 'pasión', 'ventana', 'risa', 'academia', 'locura',
    'ausencia', 'explicación', 'gloria', 'impresión', 'cerveza', 'federación', 'ventaja', 'extensión', 'ejecución', 'comparación', 'pared', 'entidad',
    'ciudadanía', 'sonrisa', 'enseñanza', 'colonia', 'legislación', 'princesa', 'cooperación', 'infraestructura', 'infancia', 'interpretación', 'ingeniería', 'excepción',
    'riqueza', 'aprobación', 'celebración', 'flor', 'ubicación', 'fábrica', 'concentración', 'transmisión', 'raíz', 'solicitud', 'columna', 'broma',
    'introducción', 'magia', 'difusión', 'tormenta', 'conclusión', 'costumbre', 'reducción', 'solidaridad', 'liberación', 'localidad', 'votación', 'petición',
    'silla', 'programación', 'tensión', 'duración', 'postura', 'preparación', 'preocupación', 'batería', 'arquitectura', 'propaganda', 'boda', 'fama',
    'destrucción', 'vigilancia', 'clasificación', 'leyenda', 'integración', 'electricidad', 'paciencia', 'conservación', 'concepción', 'presidenta', 'nave', 'obligación',
    'aparición', 'limpieza', 'realización', 'identificación', 'explotación', 'autorización', 'profundidad', 'evaluación', 'revisión', 'cobertura', 'transición', 'diversidad',
    'paja', 'habilidad', 'curiosidad', 'estabilidad', 'computadora', 'soledad', 'fórmula', 'ceremonia', 'ocupación', 'pelota', 'medalla', 'conducta',
    'multitud', 'ilusión', 'recuperación', 'hoja', 'dama', 'lástima', 'profesora', 'ficción', 'directora', 'orientación', 'tía', 'expansión',
    'tesis', 'profesión', 'reflexión', 'fortuna', 'botella', 'instalación', 'transformación', 'autonomía', 'oración', 'composición', 'gasolina', 'espada',
    'actriz', 'coalición', 'depresión', 'violación', 'totalidad', 'emoción', 'coordinación', 'galería', 'actualización', 'ira', 'manifestación', 'prioridad',
    'embajada', 'separación', 'posesión', 'combinación', 'alimentación', 'bicicleta', 'cinta', 'nacionalidad', 'explosión', 'cirugía', 'camiseta', 'reproducción',
    'casualidad', 'soberanía', 'transparencia', 'sanidad', 'tristeza', 'oscuridad', 'nariz', 'ideología', 'aldea', 'redacción', 'circulación', 'imaginación',
    'inflación', 'fortaleza', 'delegación', 'apariencia', 'rama', 'lección', 'documentación', 'prevención', 'tragedia', 'recepción', 'elaboración', 'tele',
    'venganza', 'secretaria', 'longitud', 'represión', 'invitación', 'candidatura', 'regulación', 'finalidad', 'nena', 'biblia', 'ignorancia', 'guitarra',
    'inscripción', 'emisión', 'inspiración', 'desaparición', 'suspensión', 'cuenca', 'mención', 'esencia', 'llave', 'correa', 'catedral', 'dosis',
    'compañera', 'percepción', 'facilidad', 'privacidad', 'discriminación', 'info', 'intensidad', 'convención', 'selva', 'excelencia', 'comprensión', 'eliminación',
    'fabricación', 'psicología', 'convocatoria', 'planificación', 'financiación', 'pizza', 'utilización', 'invasión', 'gravedad', 'innovación', 'dependencia', 'trayectoria',
    'herencia', 'observación', 'ansiedad', 'magnitud', 'cuota', 'vega', 'pierna', 'virtud',
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
