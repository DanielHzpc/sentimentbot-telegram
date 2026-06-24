from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
from repositories.user_repository import save_or_update_user
from repositories.analysis_repository import save_analysis
from services.sentiment_service import analyze_sentiment
from utils.validators import is_valid_text


async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        user = update.effective_user
        text = update.message.text

        if not is_valid_text(text):
            await update.message.reply_text(
                "No puedo analizar este mensaje. Por favor envía un texto más claro y completo."
            )
            return

        await update.message.chat.send_action(action=ChatAction.TYPING)

        telegram_id = user.id
        username = user.username
        nombre = user.full_name

        save_or_update_user(telegram_id, username, nombre)

        result = analyze_sentiment(text)

        save_analysis(
            telegram_id=telegram_id,
            username=username or nombre,
            texto=text,
            sentimiento=result["sentimiento"],
            confianza=result["confianza"],
            explicacion=result["explicacion"],
        )

        await update.message.reply_text(
            f"Sentimiento detectado: {result['sentimiento']}\n"
            f"Confianza: {result['confianza']}%\n"
            f"Explicación: {result['explicacion']}\n"
            f"Modo de análisis: {result.get('modo_analisis', 'IA/local')}"
        )

    except Exception:
        await update.message.reply_text(
            "Ocurrió un error al analizar tu mensaje. Inténtalo nuevamente."
        )


async def unsupported_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Solo puedo analizar mensajes de texto. Por favor envía texto, no imágenes, audios, videos, stickers o archivos."
    )
