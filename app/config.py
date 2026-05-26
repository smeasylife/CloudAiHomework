"""Configuration for the local toy project."""

from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
STATIC_DIR = ROOT_DIR / "static"
RUNTIME_DIR = ROOT_DIR / "runtime"
CHROMA_DIR = RUNTIME_DIR / "chroma"
SESSION_FILE = RUNTIME_DIR / "sessions.json"
INTERVIEW_QUESTIONS_FILE = DATA_DIR / "interview_questions.json"

for ENV_FILE in (ROOT_DIR.parent / ".env", ROOT_DIR / ".env"):
    if ENV_FILE.exists():
        for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("google_api_key", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL") or os.getenv("gemini_chat_model", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL") or os.getenv(
    "gemini_embedding_model",
    "gemini-embedding-001",
)

SUB_TOPICS = [
    "출결",
    "성적",
    "동아리",
    "리더십",
    "인성/태도",
    "진로/자율",
    "독서",
    "봉사",
]
