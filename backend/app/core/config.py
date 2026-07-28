"""Central application configuration, sourced from environment variables / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_env: str = "development"
    app_debug: bool = True
    api_v1_prefix: str = "/api/v1"
    backend_cors_origins: list[str] = ["http://localhost:3000"]

    # --- Auth (used from Milestone 3 onward) ---
    jwt_secret: str = "dev-only-secret-change-me-to-a-random-64-char-value-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # --- Postgres (used from Milestone 2 onward) ---
    database_url: str = "postgresql+asyncpg://researchmind:researchmind@localhost:5432/researchmind"

    # --- Redis / Celery (used from Milestone 5 onward) ---
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # --- Vector store (used from Milestone 12 onward) ---
    vector_store_backend: str = "faiss"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # --- Knowledge graph (used from Milestone 26 onward) ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # --- LLM providers (used from Milestone 17 onward) ---
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    default_llm_provider: str = "openai"

    # --- Embeddings (used from Milestone 11 onward) ---
    embedding_provider: str = "openai"
    jina_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
