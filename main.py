from telegram.ext import ApplicationBuilder

from bot.handlers import register_handlers
from config.settings import TELEGRAM_BOT_TOKEN
from database.models import init_db
from services.sentiment_service import analyze_sentiment, warmup_ai_model


def run_demo_mode() -> None:
    """Permite probar el análisis sin token ni conexión con Telegram."""
    print("SentimentBot Telegram - modo demo local")
    print("No se encontró TELEGRAM_BOT_TOKEN, así que se ejecutará sin Telegram.")
    print("Escribe un texto para analizarlo o escribe 'salir' para cerrar.\n")

    while True:
        text = input("Texto: ").strip()

        if text.lower() in {"salir", "exit", "quit"}:
            print("Bot finalizado.")
            break

        if len(text) < 3:
            print("No puedo analizar textos vacíos o demasiado cortos.\n")
            continue

        result = analyze_sentiment(text)
        print(f"Sentimiento detectado: {result['sentimiento']}")
        print(f"Confianza: {result['confianza']}%")
        print(f"Explicación: {result['explicacion']}\n")


def run_telegram_bot() -> None:
    init_db()

    print("Preparando analizador de sentimientos...")
    warmup_ai_model()

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    register_handlers(application)

    print("SentimentBot Telegram está ejecutándose en Telegram...")
    application.run_polling()


def main() -> None:
    init_db()

    if not TELEGRAM_BOT_TOKEN:
        run_demo_mode()
        return

    run_telegram_bot()


if __name__ == "__main__":
    main()
