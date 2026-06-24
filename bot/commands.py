from telegram import Update
from telegram.ext import ContextTypes
from repositories.analysis_repository import get_user_history, clear_user_history
from repositories.user_repository import save_or_update_user


def get_user_data(update: Update) -> tuple[int, str | None, str | None]:
    user = update.effective_user
    telegram_id = user.id
    username = user.username
    nombre = user.full_name
    return telegram_id, username, nombre


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id, username, nombre = get_user_data(update)
    save_or_update_user(telegram_id, username, nombre)

    await update.message.reply_text(
        "¡Hola! Soy SentimentBot Telegram.\n\n"
        "Envíame un mensaje de texto y analizaré si su sentimiento es Positivo, Negativo o Neutral.\n\n"
        "Puedes usar /help para ver los comandos disponibles."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Comandos disponibles:\n\n"
        "/start - Iniciar el bot\n"
        "/help - Ver ayuda\n"
        "/historial - Ver tus últimos análisis\n"
        "/limpiar - Borrar tu historial\n\n"
        "También puedes enviarme cualquier texto para analizar su sentimiento."
    )


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id, username, nombre = get_user_data(update)
    save_or_update_user(telegram_id, username, nombre)

    history = get_user_history(telegram_id)

    if not history:
        await update.message.reply_text("Todavía no tienes análisis guardados.")
        return

    message = "Tus últimos análisis:\n\n"

    for index, item in enumerate(history, start=1):
        short_text = item["texto"][:60]
        if len(item["texto"]) > 60:
            short_text += "..."

        message += (
            f"{index}. \"{short_text}\"\n"
            f"   Sentimiento: {item['sentimiento']}\n"
            f"   Confianza: {item['confianza']}%\n"
            f"   Fecha: {item['fecha_analisis']}\n\n"
        )

    await update.message.reply_text(message)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    telegram_id, username, nombre = get_user_data(update)
    save_or_update_user(telegram_id, username, nombre)

    deleted_rows = clear_user_history(telegram_id)

    if deleted_rows == 0:
        await update.message.reply_text("No tenías historial para borrar.")
        return

    await update.message.reply_text("Tu historial fue eliminado correctamente.")
