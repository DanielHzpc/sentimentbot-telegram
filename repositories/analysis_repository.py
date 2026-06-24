from database.connection import get_connection
from config.settings import HISTORY_LIMIT


def save_analysis(
    telegram_id: int,
    username: str | None,
    texto: str,
    sentimiento: str,
    confianza: int,
    explicacion: str,
) -> None:
    """Guarda el resultado del análisis en la base de datos."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO analisis (telegram_id, username, texto, sentimiento, confianza, explicacion)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (telegram_id, username, texto, sentimiento, confianza, explicacion),
        )
        conn.commit()


def get_user_history(telegram_id: int, limit: int = HISTORY_LIMIT) -> list[dict]:
    """Obtiene los últimos análisis de un usuario específico."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT texto, sentimiento, confianza, explicacion, fecha_analisis
            FROM analisis
            WHERE telegram_id = ?
            ORDER BY fecha_analisis DESC
            LIMIT ?
            """,
            (telegram_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]


def clear_user_history(telegram_id: int) -> int:
    """Borra el historial de un usuario y retorna la cantidad de registros eliminados."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analisis WHERE telegram_id = ?", (telegram_id,))
        conn.commit()
        return cursor.rowcount
