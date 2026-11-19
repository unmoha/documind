"""Application configuration module using pydantic-settings.

Defines strongly-typed settings loaded from environment variables or .env files.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """DocuMind application configuration and environment variables."""

    # Application & Environment
    env: Literal["development", "test", "production"] = Field(
        default="development",
        description="Application deployment environment.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI server host binding.",
    )
    api_port: int = Field(
        default=8000,
        description="FastAPI server port number.",
    )
    streamlit_port: int = Field(
        default=8501,
        description="Streamlit UI server port number.",
    )

    # Persistence Paths
    chroma_persist_dir: Path = Field(
        default=Path("./chroma_db"),
        description="Filesystem path for persistent ChromaDB storage.",
    )
    sqlite_db_path: Path = Field(
        default=Path("./logs.sqlite"),
        description="Filesystem path for SQLite interaction and query logs.",
    )

    # Ingestion & Chunking
    chunk_size: int = Field(
        default=800,
        gt=0,
        description="Maximum character or token count per text chunk.",
    )
    chunk_overlap: int = Field(
        default=120,
        ge=0,
        description="Character or token overlap between consecutive chunks.",
    )
    max_upload_size_bytes: int = Field(
        default=10 * 1024 * 1024,
        gt=0,
        description="Maximum allowed file upload size in bytes (default: 10 MB).",
    )

    # Embeddings
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model identifier for sentence-transformers embedding.",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Hardware device for embeddings (cpu, cuda).",
    )

    # Retrieval
    top_k: int = Field(
        default=4,
        gt=0,
        description="Default number of top chunks to retrieve per query.",
    )
    similarity_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold for retrieved chunks.",
    )

    # LLM Provider
    llm_provider: Literal["gemini", "groq"] = Field(
        default="gemini",
        description="Active LLM provider interface implementation.",
    )
    llm_model: str = Field(
        default="gemini-3.8-flash",
        description="Specific model identifier for the active provider.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for answer generation.",
    )
    request_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Timeout in seconds for external LLM API calls.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries with exponential backoff on transient failures.",
    )

    # Secrets (use SecretStr to prevent accidental leakage in logs or dumps)
    api_token: SecretStr | None = Field(
        default=None,
        description="Optional API protection token for /api/v1 endpoints.",
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="API key for Google Gemini.",
    )
    groq_api_key: SecretStr | None = Field(
        default=None,
        description="Optional API key for Groq.",
    )

    model_config = SettingsConfigDict(
        env_prefix="DOCUMIND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of the application settings."""
    return Settings()
