import os

from flask import Flask, jsonify, render_template, request

from database.models import init_db
from repositories.analysis_repository import clear_user_history, get_user_history, save_analysis
from repositories.user_repository import save_or_update_user
from services.facial_emotion_service import analyze_facial_emotions_from_base64, detect_faces_from_base64
from services.sentiment_service import analyze_sentiment, warmup_ai_model
from utils.validators import is_valid_text

WEB_USER_ID = 0
WEB_USERNAME = "usuario_web"
WEB_NAME = "Usuario Web"

app = Flask(
    __name__,
    template_folder="frontend",
    static_folder="frontend/static",
)

# Render ejecuta la app con gunicorn, por eso la base de datos
# se inicializa también al importar el módulo y no solo en __main__.
init_db()


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/analizar")
def analizar():
    data = request.get_json(silent=True) or {}
    texto = data.get("texto", "")

    if not is_valid_text(texto):
        return jsonify({
            "ok": False,
            "mensaje": "Escribe un texto más claro y completo para analizarlo.",
        }), 400

    resultado = analyze_sentiment(texto)

    save_or_update_user(WEB_USER_ID, WEB_USERNAME, WEB_NAME)
    save_analysis(
        telegram_id=WEB_USER_ID,
        username=WEB_USERNAME,
        texto=texto.strip(),
        sentimiento=resultado["sentimiento"],
        confianza=resultado["confianza"],
        explicacion=resultado["explicacion"],
    )

    return jsonify({
        "ok": True,
        "texto": texto.strip(),
        "sentimiento": resultado["sentimiento"],
        "confianza": resultado["confianza"],
        "explicacion": resultado["explicacion"],
        "modo_analisis": resultado.get("modo_analisis", "IA/local"),
    })


@app.post("/api/detectar-rostros")
def detectar_rostros():
    data = request.get_json(silent=True) or {}
    image_base64 = data.get("imagen", "")

    if not image_base64:
        return jsonify({
            "ok": False,
            "mensaje": "No se recibió ninguna imagen de la cámara.",
            "rostros": [],
        }), 400

    resultado = detect_faces_from_base64(image_base64)
    status = 200 if resultado.get("ok") else 400
    return jsonify(resultado), status


@app.post("/api/analizar-rostro")
def analizar_rostro():
    data = request.get_json(silent=True) or {}
    image_base64 = data.get("imagen", "")

    if not image_base64:
        return jsonify({
            "ok": False,
            "mensaje": "No se recibió ninguna imagen de la cámara.",
        }), 400

    client_faces = data.get("rostros") or data.get("faces") or []
    resultado = analyze_facial_emotions_from_base64(image_base64, client_faces=client_faces)

    save_or_update_user(WEB_USER_ID, WEB_USERNAME, WEB_NAME)

    if resultado.get("resultados"):
        for item in resultado["resultados"]:
            save_analysis(
                telegram_id=WEB_USER_ID,
                username=WEB_USERNAME,
                texto=f"Análisis facial por cámara - {item['persona']}",
                sentimiento=item["sentimiento"],
                confianza=item["confianza"],
                explicacion=item["explicacion"],
            )
    else:
        save_analysis(
            telegram_id=WEB_USER_ID,
            username=WEB_USERNAME,
            texto="Análisis de emoción facial mediante cámara",
            sentimiento=resultado["sentimiento"],
            confianza=resultado["confianza"],
            explicacion=resultado["explicacion"],
        )

    return jsonify({
        "ok": True,
        "texto": "Imagen capturada desde cámara",
        "sentimiento": resultado["sentimiento"],
        "confianza": resultado["confianza"],
        "explicacion": resultado["explicacion"],
        "rostro_detectado": resultado["rostro_detectado"],
        "modo_analisis": resultado["modo_analisis"],
        "resultados": resultado.get("resultados", []),
        "cantidad_rostros": resultado.get("cantidad_rostros", 0),
        "maximo_rostros": resultado.get("maximo_rostros", 2),
    })


@app.get("/api/historial")
def historial():
    save_or_update_user(WEB_USER_ID, WEB_USERNAME, WEB_NAME)
    return jsonify({
        "ok": True,
        "historial": get_user_history(WEB_USER_ID),
    })


@app.delete("/api/historial")
def limpiar_historial():
    deleted = clear_user_history(WEB_USER_ID)
    return jsonify({
        "ok": True,
        "eliminados": deleted,
        "mensaje": "Historial eliminado correctamente.",
    })


if __name__ == "__main__":
    if os.getenv("PRELOAD_AI_MODEL", "false").lower() == "true":
        print("Preparando analizador de sentimientos por texto...")
        warmup_ai_model()

    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    print(f"Frontend ejecutándose en http://127.0.0.1:{port}")
    print("Nota: en Render la cámara funciona con HTTPS. En local funciona mejor desde Chrome/Edge.")
    app.run(host="0.0.0.0", port=port, debug=debug)
