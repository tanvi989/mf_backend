from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # GCS
    GOOGLE_APPLICATION_CREDENTIALS: str
    BUCKET_NAME: str
    FOLDER_NAME: str
    GCP_PROJECT_ID: str

    # Mongo
    MONGODB_URI: str
    MONGODB_DB_NAME: str

    # Gemini
    GEMINI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"


settings = Settings()
