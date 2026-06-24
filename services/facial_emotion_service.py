"""Servicio de detección de rostros y estimación de emociones faciales.

Versión académica ligera, sin DeepFace/TensorFlow.
Usa OpenCV para:
- detectar máximo 2 rostros,
- devolver coordenadas para recuadros en vivo,
- recortar la foto del rostro analizado,
- estimar un repertorio más amplio de emociones faciales.

Importante: esto es una estimación visual aproximada, no un diagnóstico psicológico.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

MAX_FACES = 2


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    def area(self) -> int:
        return self.w * self.h

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class FacialEmotionResult:
    persona: str
    emotion: str
    confidence: int
    explanation: str
    face_detected: bool
    mode: str
    box: FaceBox | None = None
    face_image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona,
            "sentimiento": self.emotion,
            "confianza": self.confidence,
            "explicacion": self.explanation,
            "rostro_detectado": self.face_detected,
            "modo_analisis": self.mode,
            "box": self.box.to_dict() if self.box else None,
            "face_image": self.face_image,
        }


def decode_base64_image(image_base64: str) -> np.ndarray | None:
    """Convierte una imagen base64 del navegador en un frame OpenCV."""
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _load_cascade(filename: str) -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(cv2.data.haarcascades + filename)


def _prepare_gray(frame: np.ndarray) -> np.ndarray:
    """Mejora contraste para facilitar detección de rostro y rasgos."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _remove_duplicate_faces(faces: list[FaceBox]) -> list[FaceBox]:
    cleaned: list[FaceBox] = []
    for face in sorted(faces, key=lambda item: item.area(), reverse=True):
        duplicate = False
        for existing in cleaned:
            cx1, cy1 = face.x + face.w / 2, face.y + face.h / 2
            cx2, cy2 = existing.x + existing.w / 2, existing.y + existing.h / 2
            dx = abs(cx1 - cx2)
            dy = abs(cy1 - cy2)
            size_diff = abs(face.area() - existing.area()) / max(face.area(), existing.area())
            if dx < max(face.w, existing.w) * 0.38 and dy < max(face.h, existing.h) * 0.38 and size_diff < 0.60:
                duplicate = True
                break
        if not duplicate:
            cleaned.append(face)
    return cleaned


def _filter_reasonable_faces(faces: list[FaceBox], frame: np.ndarray) -> list[FaceBox]:
    h_img, w_img = frame.shape[:2]
    filtered: list[FaceBox] = []
    for face in faces:
        if face.w <= 0 or face.h <= 0:
            continue
        ratio = face.w / max(1, face.h)
        area_ratio = face.area() / max(1, h_img * w_img)
        if 0.55 <= ratio <= 1.65 and area_ratio >= 0.012:
            filtered.append(face)
    return filtered


def detect_faces_opencv(frame: np.ndarray, max_faces: int = MAX_FACES) -> list[FaceBox]:
    """Detecta rostros con varias cascadas y devuelve máximo 2 rostros."""
    gray = _prepare_gray(frame)
    min_side = max(48, int(min(frame.shape[:2]) * 0.10))

    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_alt.xml",
    ]

    detected: list[FaceBox] = []
    for cascade_name in cascade_names:
        cascade = _load_cascade(cascade_name)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.06,
            minNeighbors=4,
            minSize=(min_side, min_side),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        detected.extend(FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in faces)

    # Detección lateral simple con imagen invertida. Ayuda cuando una persona no mira 100% al frente.
    profile = _load_cascade("haarcascade_profileface.xml")
    for source, flipped in [(gray, False), (cv2.flip(gray, 1), True)]:
        faces = profile.detectMultiScale(
            source,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(min_side, min_side),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for x, y, w, h in faces:
            if flipped:
                x = frame.shape[1] - int(x) - int(w)
            detected.append(FaceBox(int(x), int(y), int(w), int(h)))

    unique_faces = _remove_duplicate_faces(detected)
    unique_faces = _filter_reasonable_faces(unique_faces, frame)
    unique_faces = sorted(unique_faces, key=lambda item: item.area(), reverse=True)
    return unique_faces[:max_faces]




def _sanitize_client_faces(client_faces: list[dict[str, Any]], frame: np.ndarray, max_faces: int = MAX_FACES) -> list[FaceBox]:
    """Convierte recuadros del navegador en FaceBox seguros.

    El navegador Chrome/Edge puede usar la API nativa FaceDetector. Esa API suele detectar
    rostros mejor que Haar Cascade. Si el frontend envía esos recuadros, los usamos para el
    análisis y evitamos redetectar mal en el backend.
    """
    if not isinstance(client_faces, list):
        return []

    h_img, w_img = frame.shape[:2]
    faces: list[FaceBox] = []
    for item in client_faces:
        if not isinstance(item, dict):
            continue
        try:
            x = int(round(float(item.get("x", 0))))
            y = int(round(float(item.get("y", 0))))
            w = int(round(float(item.get("w", item.get("width", 0)))))
            h = int(round(float(item.get("h", item.get("height", 0)))))
        except (TypeError, ValueError):
            continue

        if w <= 0 or h <= 0:
            continue

        # Ligero padding para que el análisis incluya cejas, mentón y boca completa.
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.10)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(w_img - x, w + pad_x * 2)
        h = min(h_img - y, h + pad_y * 2)

        ratio = w / max(1, h)
        area_ratio = (w * h) / max(1, h_img * w_img)
        if 0.45 <= ratio <= 1.85 and area_ratio >= 0.006:
            faces.append(FaceBox(x, y, w, h))

    faces = _remove_duplicate_faces(faces)
    faces = _filter_reasonable_faces(faces, frame)
    return sorted(faces, key=lambda face: face.area(), reverse=True)[:max_faces]


def _safe_roi(gray: np.ndarray, box: FaceBox) -> np.ndarray:
    h_img, w_img = gray.shape[:2]
    x1 = max(0, box.x)
    y1 = max(0, box.y)
    x2 = min(w_img, box.x + box.w)
    y2 = min(h_img, box.y + box.h)
    return gray[y1:y2, x1:x2]


def _safe_face_crop(frame: np.ndarray, box: FaceBox, padding: float = 0.14) -> np.ndarray | None:
    h_img, w_img = frame.shape[:2]
    pad_x = int(box.w * padding)
    pad_y = int(box.h * padding)
    x1 = max(0, box.x - pad_x)
    y1 = max(0, box.y - pad_y)
    x2 = min(w_img, box.x + box.w + pad_x)
    y2 = min(h_img, box.y + box.h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _encode_image_to_data_url(image: np.ndarray | None) -> str | None:
    if image is None or image.size == 0:
        return None
    try:
        success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not success:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
    except Exception:
        return None


def _region_mean(region: np.ndarray, default: float = 0.0) -> float:
    return float(np.mean(region)) if region.size else default


def _region_std(region: np.ndarray, default: float = 0.0) -> float:
    return float(np.std(region)) if region.size else default


def _edge_density(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    edges = cv2.Canny(region, 55, 135)
    return float(np.count_nonzero(edges)) / max(1, edges.size)


def _dark_ratio(region: np.ndarray, threshold: int = 85) -> float:
    if region.size == 0:
        return 0.0
    return float(np.count_nonzero(region < threshold)) / max(1, region.size)


def _bright_ratio(region: np.ndarray, threshold: int = 170) -> float:
    if region.size == 0:
        return 0.0
    return float(np.count_nonzero(region > threshold)) / max(1, region.size)


def _mouth_shape_features(face_gray: np.ndarray) -> dict[str, float]:
    h, w = face_gray.shape[:2]
    mouth = face_gray[int(h * 0.60):int(h * 0.90), int(w * 0.13):int(w * 0.87)]
    if mouth.size == 0:
        return {"mouth_dark": 0, "mouth_edges": 0, "mouth_open": 0, "mouth_width_activity": 0, "mouth_symmetry": 0}

    mouth_dark = _dark_ratio(mouth, 88)
    mouth_edges = _edge_density(mouth)
    _, binary_dark = cv2.threshold(mouth, 82, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mouth_open = 0.0
    mouth_width_activity = 0.0
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < mouth.size * 0.015:
            continue
        mouth_open = max(mouth_open, bh / max(1, mouth.shape[0]))
        mouth_width_activity = max(mouth_width_activity, bw / max(1, mouth.shape[1]))

    left = mouth[:, :mouth.shape[1] // 2]
    right = mouth[:, mouth.shape[1] // 2:]
    symmetry = abs(_region_mean(left, 0) - _region_mean(right, 0))

    return {
        "mouth_dark": mouth_dark,
        "mouth_edges": mouth_edges,
        "mouth_open": mouth_open,
        "mouth_width_activity": mouth_width_activity,
        "mouth_symmetry": symmetry,
    }


def _detect_smiles(face_gray: np.ndarray, box: FaceBox) -> tuple[int, float]:
    """Detecta sonrisa con reglas más estrictas para evitar falsos positivos de felicidad."""
    smile_cascade = _load_cascade("haarcascade_smile.xml")
    lower_half = face_gray[int(face_gray.shape[0] * 0.48):, :]
    smiles = smile_cascade.detectMultiScale(
        lower_half,
        scaleFactor=1.55,
        minNeighbors=18,
        minSize=(max(28, box.w // 6), max(12, box.h // 18)),
    )
    if len(smiles) == 0:
        return 0, 0.0

    valid_widths: list[float] = []
    for _x, _y, w, h in smiles:
        width_ratio = int(w) / max(1, face_gray.shape[1])
        height_ratio = int(h) / max(1, face_gray.shape[0])
        if width_ratio >= 0.23 and height_ratio <= 0.28:
            valid_widths.append(width_ratio)

    if not valid_widths:
        return 0, 0.0
    return len(valid_widths), min(1.0, max(valid_widths))


def _detect_eyes(face_gray: np.ndarray, box: FaceBox) -> tuple[int, float, float]:
    eye_cascade = _load_cascade("haarcascade_eye.xml")
    eye_glasses_cascade = _load_cascade("haarcascade_eye_tree_eyeglasses.xml")
    upper_half = face_gray[:int(face_gray.shape[0] * 0.62), :]
    eyes_a = eye_cascade.detectMultiScale(
        upper_half,
        scaleFactor=1.08,
        minNeighbors=6,
        minSize=(max(13, box.w // 12), max(10, box.h // 18)),
    )
    eyes_b = eye_glasses_cascade.detectMultiScale(
        upper_half,
        scaleFactor=1.08,
        minNeighbors=6,
        minSize=(max(13, box.w // 12), max(10, box.h // 18)),
    )

    eye_boxes = [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in list(eyes_a) + list(eyes_b)]
    eye_boxes = _remove_duplicate_faces(eye_boxes)
    eye_boxes = sorted(eye_boxes, key=lambda item: item.area(), reverse=True)[:2]
    if not eye_boxes:
        return 0, 0.0, 0.0

    avg_height = sum(e.h for e in eye_boxes) / len(eye_boxes)
    avg_width = sum(e.w for e in eye_boxes) / len(eye_boxes)
    eye_open_ratio = avg_height / max(1, face_gray.shape[0])
    eye_width_ratio = avg_width / max(1, face_gray.shape[1])
    return len(eye_boxes), float(eye_open_ratio), float(eye_width_ratio)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Pequeña normalización para que felicidad/tristeza no ganen siempre por defecto."""
    # Penalización si una emoción gana solo por ausencia de sonrisa, sin señales fuertes.
    scores["Tristeza"] *= 0.92
    scores["Felicidad"] *= 0.96
    # Emociones menos comunes reciben un leve impulso para que puedan competir cuando hay señales visuales.
    for key in ["Enojo", "Sorpresa", "Miedo / inquietud", "Desagrado", "Confusión"]:
        scores[key] *= 1.12
    return scores


def _emotion_scores(face_gray: np.ndarray, box: FaceBox, frame: np.ndarray) -> tuple[str, int, list[str], dict[str, int]]:
    """Sistema de puntajes balanceado para repertorio emocional amplio.

    Como no se usa un modelo pesado de reconocimiento facial, se combinan señales aproximadas:
    sonrisa, ojos, contraste, sombras, actividad de boca, simetría y claridad.
    """
    height, width = face_gray.shape[:2]
    brightness = _region_mean(face_gray)
    contrast = _region_std(face_gray)
    face_area_ratio = box.area() / max(1, frame.shape[0] * frame.shape[1])

    upper = face_gray[:int(height * 0.36), :]
    brow_zone = face_gray[int(height * 0.18):int(height * 0.40), int(width * 0.12):int(width * 0.88)]
    eye_zone = face_gray[int(height * 0.22):int(height * 0.58), :]
    nose_zone = face_gray[int(height * 0.38):int(height * 0.66), int(width * 0.22):int(width * 0.78)]
    lower = face_gray[int(height * 0.58):, :]
    cheek_zone = face_gray[int(height * 0.48):int(height * 0.76), :]

    smile_count, smile_width = _detect_smiles(face_gray, box)
    eye_count, eye_open_ratio, eye_width_ratio = _detect_eyes(face_gray, box)
    mouth = _mouth_shape_features(face_gray)

    upper_edges = _edge_density(upper)
    brow_edges = _edge_density(brow_zone)
    eye_edges = _edge_density(eye_zone)
    nose_edges = _edge_density(nose_zone)
    lower_edges = _edge_density(lower)
    cheek_edges = _edge_density(cheek_zone)

    eye_dark = _dark_ratio(eye_zone, 78)
    brow_dark = _dark_ratio(brow_zone, 82)
    lower_dark = _dark_ratio(lower, 92)
    bright_face = _bright_ratio(face_gray, 174)
    vertical_balance = _region_mean(lower, brightness) - _region_mean(upper, brightness)

    mouth_dark = mouth["mouth_dark"]
    mouth_edges = mouth["mouth_edges"]
    mouth_open = mouth["mouth_open"]
    mouth_width_activity = mouth["mouth_width_activity"]
    mouth_symmetry = mouth["mouth_symmetry"]

    scores: dict[str, float] = {
        "Felicidad": 0,
        "Tristeza": 0,
        "Enojo": 0,
        "Sorpresa": 0,
        "Miedo / inquietud": 0,
        "Desagrado": 0,
        "Confusión": 0,
        "Cansancio": 0,
        "Seriedad": 0,
        "Neutral": 0,
    }
    reasons: list[str] = []

    # Calidad / contexto.
    if face_area_ratio < 0.026:
        scores["Neutral"] += 12
        scores["Confusión"] += 7
        reasons.append("el rostro está algo lejos de la cámara")
    if brightness < 45:
        scores["Cansancio"] += 10
        scores["Neutral"] += 8
        reasons.append("la iluminación es baja")
    if contrast < 22:
        scores["Neutral"] += 12
        scores["Cansancio"] += 8
        reasons.append("los rasgos tienen poco contraste")

    # Felicidad: sonrisa clara, no solo boca con sombras.
    if smile_count >= 1 and smile_width >= 0.25:
        scores["Felicidad"] += 38 + int(smile_width * 28)
        reasons.append("se detecta una sonrisa amplia")
    if mouth_width_activity > 0.54 and mouth_open < 0.42 and mouth_edges > 0.070 and smile_count >= 1:
        scores["Felicidad"] += 14
        reasons.append("la boca tiene forma compatible con sonrisa")

    # Sorpresa: ojos abiertos y boca abierta/activa.
    if eye_count >= 2 and eye_open_ratio >= 0.112:
        scores["Sorpresa"] += 26
        reasons.append("los ojos parecen abiertos")
    if mouth_open > 0.42 and mouth_dark > 0.20:
        scores["Sorpresa"] += 24
        reasons.append("la boca parece abierta")
    if bright_face > 0.16 and eye_width_ratio > 0.16 and smile_count == 0:
        scores["Sorpresa"] += 9

    # Enojo: cejas/ojos tensos, contraste marcado, poca sonrisa.
    if smile_count == 0 and brow_edges > 0.098 and eye_edges > 0.090:
        scores["Enojo"] += 31
        reasons.append("la zona de cejas y ojos se ve marcada")
    if smile_count == 0 and contrast > 54 and brow_dark > 0.22:
        scores["Enojo"] += 20
        scores["Seriedad"] += 7
        reasons.append("los rasgos superiores se ven tensos")
    if smile_count == 0 and nose_edges > 0.105 and cheek_edges > 0.095:
        scores["Enojo"] += 10
        scores["Desagrado"] += 12

    # Desagrado: boca asimétrica, zona inferior marcada, sin sonrisa.
    if smile_count == 0 and mouth_symmetry > 10:
        scores["Desagrado"] += 24
        scores["Confusión"] += 10
        reasons.append("la boca se ve asimétrica")
    if smile_count == 0 and lower_edges > 0.105 and mouth_dark > 0.25 and mouth_open < 0.40:
        scores["Desagrado"] += 20
        reasons.append("la parte baja del rostro se ve tensa")

    # Miedo / inquietud: ojos abiertos o intensos, boca activa, poca sonrisa.
    if smile_count == 0 and eye_count >= 2 and eye_open_ratio >= 0.100 and eye_dark > 0.13:
        scores["Miedo / inquietud"] += 24
        reasons.append("la mirada se ve intensa sin sonrisa")
    if smile_count == 0 and mouth_open > 0.32 and brow_edges > 0.090:
        scores["Miedo / inquietud"] += 18
    if smile_count == 0 and contrast > 50 and mouth_symmetry > 7:
        scores["Miedo / inquietud"] += 10

    # Tristeza / cansancio: se requiere más que solo no sonreír.
    if smile_count == 0:
        scores["Seriedad"] += 12
        scores["Neutral"] += 5
        reasons.append("no se detecta sonrisa clara")
    if smile_count == 0 and eye_count <= 1:
        scores["Cansancio"] += 24
        scores["Tristeza"] += 12
        reasons.append("los ojos se ven poco visibles")
    if smile_count == 0 and 0 < eye_open_ratio < 0.088:
        scores["Cansancio"] += 21
        scores["Tristeza"] += 16
        reasons.append("la apertura de ojos parece baja")
    if smile_count == 0 and lower_dark > 0.38 and mouth_edges < 0.070:
        scores["Tristeza"] += 22
        reasons.append("la boca tiene baja actividad y el rostro se ve apagado")
    if smile_count == 0 and vertical_balance < -9:
        scores["Tristeza"] += 13

    # Confusión: señales mixtas o asimetría con emoción no clara.
    if smile_count == 0 and mouth_symmetry > 7 and 0.085 <= eye_open_ratio <= 0.115:
        scores["Confusión"] += 20
        reasons.append("hay señales faciales mixtas")
    if abs(upper_edges - lower_edges) < 0.010 and 30 <= contrast <= 58 and smile_count == 0:
        scores["Confusión"] += 8

    # Neutral real: rostro centrado, sin señales fuertes, buena calidad.
    if smile_count == 0 and eye_count >= 1 and 34 <= contrast <= 56 and 52 <= brightness <= 180:
        scores["Neutral"] += 22
    if mouth_edges < 0.060 and eye_edges < 0.080 and brow_edges < 0.085:
        scores["Neutral"] += 12

    scores = _normalize_scores(scores)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    emotion, top_score = ranked[0]
    second_emotion, second_score = ranked[1]
    margin = top_score - second_score

    # Etiquetas mixtas cuando hay competencia real.
    if margin < 6:
        pair = {emotion, second_emotion}
        if pair <= {"Enojo", "Seriedad", "Desagrado"} or pair & {"Enojo", "Desagrado"}:
            emotion = "Seriedad / posible enojo"
        elif pair & {"Sorpresa", "Miedo / inquietud"}:
            emotion = "Sorpresa / inquietud"
        elif pair & {"Tristeza", "Cansancio"}:
            emotion = "Tristeza / cansancio"
        elif pair & {"Confusión", "Neutral"}:
            emotion = "Confusión / neutral"
        else:
            emotion = "Neutral / poco claro"

    # Confianza: más conservadora para que no parezca que siempre está seguro.
    quality_bonus = 0
    if face_area_ratio >= 0.040:
        quality_bonus += 5
    if brightness >= 50 and contrast >= 28:
        quality_bonus += 5
    if eye_count >= 1:
        quality_bonus += 3

    confidence = int(44 + min(34, top_score * 0.42) + min(12, max(0, margin) * 0.42) + quality_bonus)
    if margin < 6:
        confidence = min(confidence, 69)
    confidence = max(45, min(confidence, 94))

    score_preview = {key: int(round(value)) for key, value in ranked[:4]}
    selected_reasons: list[str] = []
    for reason in reasons:
        if reason not in selected_reasons:
            selected_reasons.append(reason)
        if len(selected_reasons) == 3:
            break

    return emotion, confidence, selected_reasons, score_preview


def _estimate_expression(frame: np.ndarray, box: FaceBox, index: int) -> FacialEmotionResult:
    gray = _prepare_gray(frame)
    face_gray = _safe_roi(gray, box)
    face_image = _encode_image_to_data_url(_safe_face_crop(frame, box, padding=0.16))

    if face_gray.size == 0:
        return FacialEmotionResult(
            persona=f"Persona {index}",
            emotion="No analizado",
            confidence=0,
            explanation="No se pudo recortar correctamente este rostro.",
            face_detected=False,
            mode="OpenCV emociones balanceadas",
            face_image=face_image,
            box=box,
        )

    emotion, confidence, reasons, score_preview = _emotion_scores(face_gray, box, frame)
    reason_text = ", ".join(reasons) if reasons else "la expresión no tiene señales visuales fuertes"

    explanations = {
        "Felicidad": "La expresión se asocia con felicidad, agrado o comodidad.",
        "Tristeza": "La expresión puede asociarse con tristeza o bajo ánimo.",
        "Tristeza / cansancio": "La expresión puede asociarse con tristeza, cansancio o bajo ánimo.",
        "Cansancio": "La expresión puede estar relacionada con cansancio o baja energía.",
        "Enojo": "La expresión puede asociarse con enojo, tensión o molestia.",
        "Seriedad / posible enojo": "La expresión parece seria y podría indicar tensión, molestia o enojo leve.",
        "Sorpresa": "La expresión puede asociarse con sorpresa o atención alta.",
        "Miedo / inquietud": "La expresión puede asociarse con inquietud, nervios o miedo leve.",
        "Sorpresa / inquietud": "La expresión puede asociarse con sorpresa, inquietud o atención alta.",
        "Desagrado": "La expresión puede asociarse con incomodidad o desagrado.",
        "Confusión": "La expresión puede asociarse con duda, confusión o señales mixtas.",
        "Confusión / neutral": "La expresión tiene señales mixtas y no muestra una emoción fuerte.",
        "Seriedad": "La expresión parece seria o concentrada.",
        "Neutral": "La expresión parece neutral o de baja intensidad emocional.",
        "Neutral / poco claro": "La emoción no es clara; mejora la iluminación o mira más al frente.",
    }

    explanation = (
        f"Persona {index}: {explanations.get(emotion, 'La emoción estimada es aproximada.')} "
        f"Se tuvo en cuenta que {reason_text}. "
        f"Puntajes principales: {score_preview}."
    )

    return FacialEmotionResult(
        persona=f"Persona {index}",
        emotion=emotion,
        confidence=confidence,
        explanation=explanation,
        face_detected=True,
        mode="OpenCV emociones balanceadas",
        face_image=face_image,
        box=box,
    )


def detect_faces_from_base64(image_base64: str) -> dict[str, Any]:
    frame = decode_base64_image(image_base64)
    if frame is None:
        return {
            "ok": False,
            "mensaje": "No se pudo leer la imagen de la cámara.",
            "rostros": [],
            "cantidad": 0,
            "maximo": MAX_FACES,
        }

    faces = detect_faces_opencv(frame, MAX_FACES)
    return {
        "ok": True,
        "rostros": [face.to_dict() for face in faces],
        "cantidad": len(faces),
        "maximo": MAX_FACES,
        "modo_analisis": "OpenCV detección multi-rostro mejorada",
    }


def analyze_facial_emotions_from_base64(image_base64: str, client_faces: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    frame = decode_base64_image(image_base64)
    if frame is None:
        return {
            "sentimiento": "No analizado",
            "confianza": 0,
            "explicacion": "No se pudo leer la imagen capturada por la cámara.",
            "rostro_detectado": False,
            "modo_analisis": "Error de imagen",
            "resultados": [],
            "cantidad_rostros": 0,
            "maximo_rostros": MAX_FACES,
        }

    faces = _sanitize_client_faces(client_faces or [], frame, MAX_FACES)
    detection_mode = "FaceDetector del navegador + OpenCV emociones" if faces else "OpenCV detección facial"
    if not faces:
        faces = detect_faces_opencv(frame, MAX_FACES)
        detection_mode = "OpenCV detección facial"
    if not faces:
        return {
            "sentimiento": "Sin rostro",
            "confianza": 0,
            "explicacion": "No se detectó ningún rostro. Acércate a la cámara, mejora la luz y mira al frente.",
            "rostro_detectado": False,
            "modo_analisis": detection_mode,
            "resultados": [],
            "cantidad_rostros": 0,
            "maximo_rostros": MAX_FACES,
        }

    results = [_estimate_expression(frame, face, idx + 1).to_dict() for idx, face in enumerate(faces)]
    summary = " | ".join(f"{item['persona']}: {item['sentimiento']} ({item['confianza']}%)" for item in results)
    return {
        "sentimiento": "Análisis múltiple" if len(results) > 1 else results[0]["sentimiento"],
        "confianza": round(sum(item["confianza"] for item in results) / len(results)),
        "explicacion": summary,
        "rostro_detectado": True,
        "modo_analisis": detection_mode,
        "resultados": results,
        "cantidad_rostros": len(results),
        "maximo_rostros": MAX_FACES,
    }


# Compatibilidad con versiones anteriores del proyecto.
def analyze_facial_emotion_from_base64(image_base64: str) -> dict[str, Any]:
    return analyze_facial_emotions_from_base64(image_base64)
