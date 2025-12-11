from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # API CONFIGURATION
    app_name: str = "ai-analysis-service"
    app_version: str = "0.1.0"
    app_port: int = 8083

    # MODEL CONFIGURATION
    summarizer_model: str = "facebook/bart-large-cnn"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
