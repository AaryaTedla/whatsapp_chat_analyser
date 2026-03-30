"""Configuration management."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Database
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./lumira_reports.db")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./instance/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-3-4b-it:free")

# File upload limits
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB

# Pagination
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

# Analysis
TOP_SENDERS = 10
TOP_WORDS = 20
MAX_AI_SAMPLE = 50
