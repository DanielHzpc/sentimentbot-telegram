from database.connection import get_connection


def init_db():
    """Crea las tablas necesarias si todavía no existen."""
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS usuarios (
                id_usuario INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER NOT NULL UNIQUE,
                username TEXT,
                nombre TEXT,
                fecha_registro TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS analisis (
                id_analisis INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER NOT NULL,
                username TEXT,
                texto TEXT NOT NULL,
                sentimiento TEXT NOT NULL,
                confianza INTEGER NOT NULL,
                explicacion TEXT NOT NULL,
                fecha_analisis TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (telegram_id) REFERENCES usuarios (telegram_id)
            )
            """
        )

        conn.commit()
