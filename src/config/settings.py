from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # API
    app_name: str = "ai-analysis-service"
    app_version: str = "2.0.0"
    app_port: int = 8083

    # vLLM
    vllm_base_url: str = "http://localhost:8000"
    vllm_model_name: str = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
    vllm_api_key: str = "EMPTY"
    vllm_timeout: float = 600.0
    vllm_max_tokens: int = 16384
    vllm_temperature: float = 0.1
    vllm_top_p: float = 0.95

    # Text limits
    max_text_length: int = 500_000
    max_batch_size: int = 50

    # Database (PostgreSQL via CNPG postgres-io cluster)
    DATABASE_URL: str = ""
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10

    # Prompt generation (second LLM call for Flux/SDXL prompts)
    prompt_gen_enabled: bool = True
    prompt_gen_temperature: float = 0.35
    prompt_gen_max_tokens: int = 2048

    # NATS JetStream
    NATS_URL: str = "nats://nats.visiobook-namespace.svc.cluster.local:4222"
    NATS_USER: str = ""
    NATS_PASSWORD: str = ""
    NATS_STREAM_NAME: str = "VISIOBOOK_PROJECT"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
