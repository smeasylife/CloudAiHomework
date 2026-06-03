from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
STATIC_DIR = ROOT_DIR / "static"
INTERVIEW_QUESTIONS_FILE = DATA_DIR / "interview_questions.json"


class Settings(BaseSettings):
    google_api_key: str | None = None
    database_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=(ROOT_DIR.parent / ".env", ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
