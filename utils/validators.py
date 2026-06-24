from config.settings import MIN_TEXT_LENGTH


def is_valid_text(text: str | None) -> bool:
    """Valida que el mensaje sea texto suficiente para analizar."""
    if text is None:
        return False
    return len(text.strip()) >= MIN_TEXT_LENGTH
