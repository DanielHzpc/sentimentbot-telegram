import sqlite3
from config.settings import DATABASE_PATH


def get_connection():
    """Crea y retorna una conexión a la base de datos SQLite."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection
