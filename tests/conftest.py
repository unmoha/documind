"""Pytest shared fixtures and test configuration."""

import pytest

from documind.config.settings import Settings


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    """Fixture providing isolated temporary settings for testing."""
    return Settings(
        env="test",
        chroma_persist_dir=tmp_path / "chroma_db",
        sqlite_db_path=tmp_path / "test_logs.sqlite",
        gemini_api_key="test-gemini-key",
        top_k=2,
    )
