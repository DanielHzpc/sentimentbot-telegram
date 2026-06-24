import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DATABASE_NAME = os.getenv("DATABASE_NAME", "sentimentbot.db")
DATABASE_PATH = BASE_DIR / DATABASE_NAME

HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "10"))
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "3"))

USE_AI_SENTIMENT = os.getenv("USE_AI_SENTIMENT", "true")
AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "finiteautomata/beto-sentiment-analysis")
