"""Unit tests for the application configuration module."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from documind.config.settings import Settings, get_settings


def test_default_settings():
    """Verify that default settings instantiate with expected values."""
    settings = Settings()

    assert settings.env == "development"
    assert settings.api_port == 8000
    assert settings.chroma_persist_dir == Path("./chroma_db")
    assert settings.sqlite_db_path == Path("./logs.sqlite")
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.llm_provider == "gemini"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.top_k == 4
    assert settings.temperature == 0.0


def test_env_override(monkeypatch):
    """Verify that DOCUMIND_ prefixed environment variables override defaults."""
    monkeypatch.setenv("DOCUMIND_ENV", "production")
    monkeypatch.setenv("DOCUMIND_API_PORT", "9090")
    monkeypatch.setenv("DOCUMIND_TOP_K", "8")
    monkeypatch.setenv("DOCUMIND_CHUNK_SIZE", "1000")
    monkeypatch.setenv("DOCUMIND_GEMINI_API_KEY", "secret-test-key-1234")

    settings = Settings()

    assert settings.env == "production"
    assert settings.api_port == 9090
    assert settings.top_k == 8
    assert settings.chunk_size == 1000
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "secret-test-key-1234"


def test_secret_str_masking():
    """Verify that SecretStr does not leak plain text in string representation."""
    settings = Settings(gemini_api_key="my-super-secret-key")

    assert settings.gemini_api_key is not None
    assert "my-super-secret-key" not in str(settings)
    assert "my-super-secret-key" not in repr(settings)
    assert "my-super-secret-key" not in str(settings.gemini_api_key)
    assert settings.gemini_api_key.get_secret_value() == "my-super-secret-key"


def test_validation_errors():
    """Verify that invalid field values trigger Pydantic ValidationErrors."""
    with pytest.raises(ValidationError):
        # chunk_size must be gt 0
        Settings(chunk_size=-10)

    with pytest.raises(ValidationError):
        # temperature must be between 0.0 and 2.0
        Settings(temperature=3.5)

    with pytest.raises(ValidationError):
        # env must be one of literal options
        Settings(env="invalid_env")


def test_get_settings_cached():
    """Verify that get_settings returns the same cached instance across calls."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
