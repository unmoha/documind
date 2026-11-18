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


def test_cli_reingest_modified_file_and_delete(
    tmp_path: Path,
    fake_embedder: FakeEmbedder,
    test_vector_store: ChromaVectorStore,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test CLI behavior when modifying a file, re-ingesting, and deleting by filename."""
    doc_path = tmp_path / "document.txt"
    doc_path.write_text(
        "Initial version of document with chunkable text.",
        encoding="utf-8",
    )

    # 1. Initial ingestion
    exit_code = main(
        argv=["ingest", str(doc_path)],
        embedder=fake_embedder,
        store=test_vector_store,
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Added 1 chunks, skipped 0 chunks as duplicates." in captured.out

    # 2. Modify document content (different content hash)
    doc_path.write_text(
        "Revised version of document with completely fresh content.",
        encoding="utf-8",
    )

    # 3. Re-ingest modified file: should replace old chunks and report count
    exit_code = main(
        argv=["ingest", str(doc_path)],
        embedder=fake_embedder,
        store=test_vector_store,
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Added 1 chunks, replaced 1 old chunks." in captured.out

    # 4. Collection count should reflect only the new version's chunk
    exit_code = main(argv=["count"], store=test_vector_store)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Chunks in collection: 1" in captured.out

    # 5. Delete by filename via CLI
    exit_code = main(
        argv=["delete", "document.txt"],
        store=test_vector_store,
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Removed 1 chunks for 'document.txt'." in captured.out

    # 6. Collection count should now be 0
    exit_code = main(argv=["count"], store=test_vector_store)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Chunks in collection: 0" in captured.out

    # 7. Delete non-existent filename reports 0 removed
    exit_code = main(
        argv=["delete", "document.txt"],
        store=test_vector_store,
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Removed 0 chunks for 'document.txt'." in captured.out
