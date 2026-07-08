"""
Application configuration.
All values are loaded from environment variables (or a .env file in dev).
Never hardcode secrets here.
"""
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    APP_NAME: str = "Enterprise Deepfake Detection Platform"
    ENVIRONMENT: str = Field(default="development")  # development | staging | production
    DEBUG: bool = Field(default=False)
    API_V1_PREFIX: str = "/api/v1"

    # --- Security ---
    SECRET_KEY: str = Field(..., description="Used to sign JWTs. MUST be set via env var in prod.")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # --- CORS ---
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- Database ---
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/deepfake_platform"
    )
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # --- Redis ---
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # --- Object storage (S3-compatible) ---
    S3_ENDPOINT_URL: str | None = None
    S3_ACCESS_KEY: str | None = None
    S3_SECRET_KEY: str | None = None
    S3_BUCKET_NAME: str = "deepfake-media"
    S3_REGION: str = "us-east-1"

    # --- Uploads ---
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_VIDEO_EXTENSIONS: List[str] = Field(default_factory=lambda: [".mp4", ".mov", ".avi", ".mkv"])
    ALLOWED_IMAGE_EXTENSIONS: List[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"])

    # --- Rate limiting ---
    RATE_LIMIT_PER_MINUTE: int = 60

    # --- AI inference service ---
    INFERENCE_SERVICE_URL: str = Field(default="http://localhost:8500")
    INFERENCE_TIMEOUT_SECONDS: int = 120


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
