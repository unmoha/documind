"""Pytest shared fixtures and test configuration."""

from pathlib import Path

import pytest

from documind.config.settings import Settings
from documind.ingestion.embedder import FakeEmbedder
from documind.ingestion.storage import ChromaVectorStore


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Fixture providing isolated temporary settings for testing."""
    return Settings(
        env="test",
        chroma_persist_dir=tmp_path / "chroma_db",
        sqlite_db_path=tmp_path / "test_logs.sqlite",
        gemini_api_key="test-gemini-key",
        top_k=2,
    )


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """Fixture providing a deterministic, offline fake embedder."""
    return FakeEmbedder(dimension=384)


@pytest.fixture
def test_vector_store(tmp_path: Path) -> ChromaVectorStore:
    """Fixture providing an isolated ChromaVectorStore for unit tests."""
    return ChromaVectorStore(
        persist_dir=tmp_path / "test_chroma_db",
        collection_name="test_collection",
    )
