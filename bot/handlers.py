from telegram.ext import Application, CommandHandler, MessageHandler, filters
from bot.commands import start_command, help_command, history_command, clear_command
from bot.messages import text_message_handler, unsupported_message_handler


def register_handlers(application: Application) -> None:
    """Registra comandos y manejadores del bot."""
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("historial", history_command))
    application.add_handler(CommandHandler("limpiar", clear_command))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))
    application.add_handler(MessageHandler(~filters.TEXT, unsupported_message_handler))
