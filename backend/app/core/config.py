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
    # Runs ingestion inline instead of dispatching to a worker. Lets the API be
    # exercised end-to-end without a Redis/Celery process (tests, local dev).
    celery_task_always_eager: bool = False

    # --- Storage (used from Milestone 5 onward) ---
    storage_local_path: str = "./storage"

    # --- Document understanding (used from Milestone 7 onward) ---
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = 300
    extract_tables: bool = True
    extract_figures: bool = True
    max_upload_bytes: int = 100 * 1024 * 1024

    # --- Vector store (used from Milestone 12 onward) ---
    # "memory" needs no extra dependency and is exact; "faiss" scales further
    # locally; "qdrant" is the production backend.
    vector_store_backend: str = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "researchmind_chunks"
    qdrant_api_key: str = ""

    # --- Knowledge graph (used from Milestone 26 onward) ---
    # "memory" persists to a JSON file and needs no container; "neo4j" is the
    # production backend and the only one that can run generated Cypher.
    graph_store_backend: str = "memory"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    graph_extraction_enabled: bool = True
    graph_max_query_limit: int = 100

    # --- LLM providers (used from Milestone 17 onward) ---
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    # "echo" is an offline provider used when no API key is configured.
    default_llm_provider: str = "echo"
    fallback_llm_provider: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

    # --- Embeddings (used from Milestone 11 onward) ---
    # "hash" is a deterministic local provider with no network dependency; it
    # lets the retrieval stack run end-to-end without API keys.
    embedding_provider: str = "hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 384
    embedding_batch_size: int = 64

    # --- Reranking (used from Milestone 14 onward) ---
    # "lexical" is dependency-free and deterministic; "cross-encoder" is the
    # quality option and needs sentence-transformers.
    reranker_backend: str = "lexical"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_default_limit: int = 10

    # --- Web search (used from Milestone 24 onward) ---
    # Disabled unless a provider and key are set; external content is
    # untrusted and optional enrichment, never required for a run.
    web_search_provider: str = ""
    tavily_api_key: str = ""
    web_search_enabled: bool = False
    jina_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
