"""Tests for the ingestion command-line interface."""

from pathlib import Path

import pytest

from documind.ingestion.cli import main
from documind.ingestion.embedder import FakeEmbedder
from documind.ingestion.storage import ChromaVectorStore


def test_cli_ingest_and_count_with_fake_embedder(
    tmp_path: Path,
    fake_embedder: FakeEmbedder,
    test_vector_store: ChromaVectorStore,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test CLI commands 'ingest' (new & duplicate) and 'count' using the fake embedder."""
    # Create a test document
    doc_path = tmp_path / "sample.txt"
    doc_path.write_text(
        "DocuMind provides grounded retrieval-augmented generation.\n\n"
        "It splits long documents into manageable chunks and stores embeddings.",
        encoding="utf-8",
    )

    # 1. Count before any ingestion should be 0
    exit_code = main(argv=["count"], store=test_vector_store)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Chunks in collection: 0" in captured.out

    # 2. Ingest document for the first time
    exit_code = main(
        argv=["ingest", str(doc_path)],
        embedder=fake_embedder,
        store=test_vector_store,
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Added 1 chunks, skipped 0 chunks as duplicates." in captured.out

    # 3. Ingest same document again (idempotent duplicate detection)
    exit_code = main(
        argv=["ingest", str(doc_path)],
        embedder=fake_embedder,
        store=test_vector_store,
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Added 0 chunks, skipped 1 chunks as duplicates." in captured.out

    # 4. Count after ingestion should reflect indexed chunks
    exit_code = main(argv=["count"], store=test_vector_store)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Chunks in collection: 1" in captured.out

    # 5. Ingest non-existent file returns error code and message
    exit_code = main(
        argv=["ingest", str(tmp_path / "non_existent.txt")],
        embedder=fake_embedder,
        store=test_vector_store,
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
