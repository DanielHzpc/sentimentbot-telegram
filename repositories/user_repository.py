from database.connection import get_connection


def save_or_update_user(telegram_id: int, username: str | None, nombre: str | None) -> None:
    """Guarda el usuario si no existe o actualiza sus datos básicos."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO usuarios (telegram_id, username, nombre)
            VALUES (?, ?, ?)
            ON CONFLICT(telegram_id) DO UPDATE SET
                username = excluded.username,
                nombre = excluded.nombre
            """,
            (telegram_id, username, nombre),
        )
        conn.commit()
