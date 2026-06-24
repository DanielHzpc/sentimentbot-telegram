"""Servicio de análisis de sentimientos.

Modo optimizado:
- Detecta primero casos sensibles y emociones claras con reglas rápidas.
- Usa IA Transformers solo cuando el texto es ambiguo o necesita más análisis.
- Carga el modelo una sola vez y reutiliza la instancia para evitar demoras en cada mensaje.
- Tiene caché para textos repetidos.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, List, Tuple

DEFAULT_AI_MODEL = os.getenv("AI_MODEL_NAME", "finiteautomata/beto-sentiment-analysis")
USE_AI_SENTIMENT = os.getenv("USE_AI_SENTIMENT", "true").lower() in {"1", "true", "yes", "si", "sí"}
# hybrid_fast = reglas rápidas primero + IA solo cuando hace falta.
# ai_only = todo texto válido pasa por IA, pero será más lento.
# rules_only = sin IA, muy rápido.
ANALYSIS_MODE = os.getenv("ANALYSIS_MODE", "hybrid_fast").lower()
PRELOAD_AI_MODEL = os.getenv("PRELOAD_AI_MODEL", "true").lower() in {"1", "true", "yes", "si", "sí"}

LABEL_TRANSLATIONS = {
    "POS": "Positivo",
    "POSITIVE": "Positivo",
    "LABEL_2": "Positivo",
    "NEG": "Negativo",
    "NEGATIVE": "Negativo",
    "LABEL_0": "Negativo",
    "NEU": "Neutral",
    "NEUTRAL": "Neutral",
    "LABEL_1": "Neutral",
}

EXPLANATIONS = {
    "Positivo": "El mensaje expresa una emoción favorable, satisfacción, agrado o esperanza.",
    "Negativo": "El mensaje expresa una emoción desfavorable, malestar, inconformidad, tristeza o frustración.",
    "Tristeza": "El mensaje expresa tristeza, dolor emocional, vacío, desánimo o cansancio emocional.",
    "Enojo": "El mensaje expresa rabia, molestia, frustración o inconformidad fuerte.",
    "Miedo": "El mensaje expresa temor, ansiedad, inseguridad o preocupación.",
    "Neutral": "El mensaje no expresa una emoción suficientemente clara o mezcla varias señales emocionales.",
    "Alerta emocional": "El mensaje contiene señales de sufrimiento intenso o posible riesgo personal. Es recomendable buscar apoyo de una persona de confianza o ayuda profesional.",
}

RISK_PATTERNS = [
    r"\bme\s+quiero\s+matar\b",
    r"\bquiero\s+morir\b",
    r"\bno\s+quiero\s+vivir\b",
    r"\bme\s+voy\s+a\s+matar\b",
    r"\bme\s+quiero\s+morir\b",
    r"\bno\s+aguanto\s+m[aá]s\b",
    r"\bquiero\s+desaparecer\b",
    r"\bterminar\s+con\s+mi\s+vida\b",
]

POSITIVE_TERMS = {
    "feliz": 2, "alegre": 2, "contento": 2, "contenta": 2, "excelente": 3,
    "genial": 3, "perfecto": 3, "perfecta": 3, "maravilloso": 3,
    "maravillosa": 3, "increible": 3, "increíble": 3, "amo": 2,
    "encanta": 2, "encanto": 2, "encantó": 2, "gusta": 1, "gusto": 1, "gustó": 1,
    "bueno": 1, "buena": 1, "bien": 1, "gracias": 1, "recomiendo": 2,
    "satisfecho": 2, "satisfecha": 2, "tranquilo": 1, "tranquila": 1,
    "esperanza": 1, "logre": 2, "logré": 2, "mejor": 1, "bonito": 1,
}

NEGATIVE_TERMS = {
    "triste": 3, "deprimido": 3, "deprimida": 3, "llorar": 2, "llorando": 2,
    "solo": 1, "sola": 1, "vacio": 2, "vacío": 2, "dolor": 2,
    "mal": 2, "malo": 2, "mala": 2, "terrible": 3, "horrible": 3,
    "pesimo": 3, "pésimo": 3, "odio": 3, "rabia": 3, "ira": 3,
    "enojado": 2, "enojada": 2, "molesto": 2, "molesta": 2, "frustrado": 2,
    "frustrada": 2, "miedo": 2, "asustado": 2, "asustada": 2, "ansiedad": 2,
    "ansioso": 2, "ansiosa": 2, "problema": 1, "error": 1, "fallo": 1,
    "cansado": 1, "cansada": 1, "agotado": 2, "agotada": 2, "desesperado": 3,
    "desesperada": 3, "fracaso": 2, "perdido": 2, "perdida": 2, "peor": 2,
    "daño": 2, "dano": 2, "insuficiente": 1, "preocupado": 2, "preocupada": 2,
}

NEGATION_TERMS = {"no", "nunca", "jamas", "jamás", "tampoco", "ni"}
INTENSIFIERS = {"muy", "demasiado", "bastante", "re", "super", "súper", "tan", "totalmente"}
DIMINISHERS = {"poco", "algo", "medio", "casi"}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    text = re.sub(r"[^a-zñü\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_risk_message(text: str) -> bool:
    clean = normalize_text(text)
    return any(re.search(pattern, clean) for pattern in RISK_PATTERNS)


@lru_cache(maxsize=1)
def get_ai_pipeline():
    """Carga el modelo una sola vez. La primera carga puede tardar, luego queda en memoria."""
    if not USE_AI_SENTIMENT or ANALYSIS_MODE == "rules_only":
        return None

    try:
        # Torch es pesado; limitar hilos suele mejorar respuesta en equipos pequeños.
        try:
            import torch
            torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "2"))))
        except Exception:
            pass

        from transformers import pipeline

        return pipeline(
            task="sentiment-analysis",
            model=DEFAULT_AI_MODEL,
            tokenizer=DEFAULT_AI_MODEL,
            truncation=True,
            max_length=256,
            device=-1,
        )
    except Exception:
        return None


def warmup_ai_model() -> None:
    """Precarga la IA al iniciar para que el primer mensaje de Telegram no sea el lento."""
    if USE_AI_SENTIMENT and ANALYSIS_MODE != "rules_only" and PRELOAD_AI_MODEL:
        pipeline_model = get_ai_pipeline()
        if pipeline_model is not None:
            try:
                pipeline_model("mensaje de prueba")
            except Exception:
                pass


def analyze_with_ai(text: str) -> Dict[str, object] | None:
    ai_pipeline = get_ai_pipeline()
    if ai_pipeline is None:
        return None

    try:
        result = ai_pipeline(text[:700])[0]
        label = str(result.get("label", "NEU")).upper()
        score = float(result.get("score", 0.0))
        sentimiento = LABEL_TRANSLATIONS.get(label, "Neutral")
        confianza = int(round(score * 100))
        confianza = max(50, min(confianza, 99))

        return {
            "sentimiento": sentimiento,
            "confianza": confianza,
            "explicacion": EXPLANATIONS.get(sentimiento, EXPLANATIONS["Neutral"]),
            "modo_analisis": "IA Transformers",
        }
    except Exception:
        return None


def score_terms(words: List[str]) -> Tuple[float, float, List[str]]:
    positive_score = 0.0
    negative_score = 0.0
    detected_terms: List[str] = []

    for index, word in enumerate(words):
        previous_words = words[max(0, index - 3):index]
        has_negation = any(term in NEGATION_TERMS for term in previous_words)
        intensity = 1.0

        if any(term in INTENSIFIERS for term in previous_words):
            intensity += 0.45
        if any(term in DIMINISHERS for term in previous_words):
            intensity -= 0.25

        if word in POSITIVE_TERMS:
            value = POSITIVE_TERMS[word] * intensity
            detected_terms.append(word)
            if has_negation:
                negative_score += value
            else:
                positive_score += value

        if word in NEGATIVE_TERMS:
            value = NEGATIVE_TERMS[word] * intensity
            detected_terms.append(word)
            if has_negation:
                positive_score += value * 0.7
            else:
                negative_score += value

    return positive_score, negative_score, detected_terms


def analyze_with_rules(text: str) -> Dict[str, object]:
    clean = normalize_text(text)
    words = clean.split()
    positive_score, negative_score, detected_terms = score_terms(words)

    phrase_scores = {
        "no me gusta": (0, 2.5),
        "me siento mal": (0, 3.5),
        "estoy cansado de todo": (0, 4.0),
        "estoy cansada de todo": (0, 4.0),
        "todo esta bien": (2.5, 0),
        "me fue bien": (2.5, 0),
        "me siento feliz": (3.5, 0),
        "no estoy bien": (0, 3.0),
        "no puedo mas": (0, 4.0),
        "me siento vacio": (0, 3.5),
        "me siento vacia": (0, 3.5),
    }

    for phrase, (pos, neg) in phrase_scores.items():
        if phrase in clean:
            positive_score += pos
            negative_score += neg
            detected_terms.append(phrase)

    total = positive_score + negative_score
    if total == 0:
        return {
            "sentimiento": "Neutral",
            "confianza": 52,
            "explicacion": "No encontré señales emocionales fuertes; el mensaje parece informativo o ambiguo.",
            "modo_analisis": "Reglas rápidas",
        }

    difference = abs(positive_score - negative_score)
    dominance = difference / total
    base_confidence = 55 + int(dominance * 35) + min(len(detected_terms) * 2, 8)
    confianza = max(54, min(base_confidence, 96))

    if dominance < 0.18:
        return {
            "sentimiento": "Neutral",
            "confianza": max(55, confianza - 8),
            "explicacion": "El mensaje combina señales positivas y negativas o no tiene una emoción dominante.",
            "modo_analisis": "Reglas rápidas",
        }

    if negative_score > positive_score:
        sadness_terms = {"triste", "deprimido", "deprimida", "llorar", "llorando", "vacio", "vacío"}
        anger_terms = {"rabia", "ira", "odio", "enojado", "enojada"}
        fear_terms = {"miedo", "asustado", "asustada", "ansiedad", "ansioso", "ansiosa", "preocupado", "preocupada"}

        if any(term in detected_terms for term in sadness_terms):
            sentimiento = "Tristeza"
        elif any(term in detected_terms for term in anger_terms):
            sentimiento = "Enojo"
        elif any(term in detected_terms for term in fear_terms):
            sentimiento = "Miedo"
        else:
            sentimiento = "Negativo"
    else:
        sentimiento = "Positivo"

    return {
        "sentimiento": sentimiento,
        "confianza": confianza,
        "explicacion": EXPLANATIONS[sentimiento],
        "modo_analisis": "Reglas rápidas",
    }


def _should_use_ai_after_rules(text: str, rules_result: Dict[str, object]) -> bool:
    """Decide si vale la pena usar IA. Evita demoras en mensajes obvios."""
    if not USE_AI_SENTIMENT or ANALYSIS_MODE in {"rules_only"}:
        return False
    if ANALYSIS_MODE == "ai_only":
        return True

    confianza = int(rules_result.get("confianza", 0))
    sentimiento = str(rules_result.get("sentimiento", "Neutral"))
    word_count = len(normalize_text(text).split())

    # Si reglas rápidas están bastante seguras, responder rápido.
    if sentimiento != "Neutral" and confianza >= 78:
        return False

    # Mensajes largos/ambiguos sí pasan por IA.
    if word_count >= 9:
        return True

    # Neutral de baja confianza puede necesitar IA.
    return sentimiento == "Neutral" and confianza < 65


@lru_cache(maxsize=256)
def _analyze_cached(clean_key: str, original_text: str) -> Tuple[str, int, str, str]:
    if contains_risk_message(original_text):
        return (
            "Alerta emocional",
            98,
            EXPLANATIONS["Alerta emocional"],
            "Detección de riesgo + análisis emocional",
        )

    if ANALYSIS_MODE == "ai_only":
        ai_result = analyze_with_ai(original_text)
        if ai_result is not None:
            return (
                str(ai_result["sentimiento"]),
                int(ai_result["confianza"]),
                str(ai_result["explicacion"]),
                str(ai_result["modo_analisis"]),
            )

    rules_result = analyze_with_rules(original_text)

    if _should_use_ai_after_rules(original_text, rules_result):
        ai_result = analyze_with_ai(original_text)
        if ai_result is not None:
            return (
                str(ai_result["sentimiento"]),
                int(ai_result["confianza"]),
                str(ai_result["explicacion"]),
                "IA Transformers + reglas rápidas",
            )

    mode = str(rules_result.get("modo_analisis", "Reglas rápidas"))
    if ANALYSIS_MODE == "hybrid_fast" and USE_AI_SENTIMENT:
        mode = "Híbrido rápido"

    return (
        str(rules_result["sentimiento"]),
        int(rules_result["confianza"]),
        str(rules_result["explicacion"]),
        mode,
    )


def analyze_sentiment(text: str) -> Dict[str, object]:
    """Analiza un texto y devuelve sentimiento, confianza, explicación y modo usado."""
    clean_key = normalize_text(text)[:700]
    sentimiento, confianza, explicacion, modo = _analyze_cached(clean_key, text.strip()[:700])
    return {
        "sentimiento": sentimiento,
        "confianza": confianza,
        "explicacion": explicacion,
        "modo_analisis": modo,
    }
