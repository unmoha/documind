"""Unit tests for SQLite interaction logging and pipeline auditing."""

import json

from documind.generation import AnswerGenerator, FakeLLMClient, RAGPipeline
from documind.generation.cli import main
from documind.ingestion.embedder import FakeEmbedder
from documind.ingestion.storage import ChromaVectorStore
from documind.logging import SQLiteLogger
from documind.retrieval import Retriever


def test_sqlite_logger_persistence(tmp_path):
    db_path = tmp_path / "test_logs.sqlite"
    logger = SQLiteLogger(db_path=db_path)

    row_id = logger.log_interaction(
        query="What are the success metrics?",
        best_score=0.768,
        has_relevant_context=True,
        answer="The hit rate is >= 90% [1].",
        citations_json='[{"source_number": 1, "filename": "SPEC.md"}]',
        latency_ms=123.4,
    )

    assert row_id == 1

    logs = logger.get_recent_logs(limit=5)
    assert len(logs) == 1
    assert logs[0].query == "What are the success metrics?"
    assert logs[0].best_score == 0.768
    assert logs[0].has_relevant_context is True
    assert logs[0].latency_ms == 123.4


def test_rag_pipeline_audits_to_sqlite_and_cli_logs(tmp_path, capsys):
    db_path = tmp_path / "audit.sqlite"
    logger = SQLiteLogger(db_path=db_path)
    emb = FakeEmbedder()
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma")
    text = "SQLite records every query and citation for auditing."
    store.collection.upsert(
        ids=["h1_0"],
        documents=[text],
        metadatas=[
            {
                "filename": "SPEC.md",
                "page_number": 1,
                "chunk_index": 4,
                "file_hash": "h1",
                "document_id": "h1",
            }
        ],
        embeddings=emb.embed_documents([text]),
    )

    retriever = Retriever(embedder=emb, store=store, top_k=2, threshold=0.0)
    fake_llm = FakeLLMClient("SQLite logs every query and citation [1].")
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=AnswerGenerator(fake_llm),
        logger=logger,
    )

    pipeline.ask(text)
    entries = logger.get_recent_logs(limit=5)
    assert len(entries) == 1
    assert entries[0].has_relevant_context is True
    citations = json.loads(entries[0].citations_json)
    assert citations[0]["filename"] == "SPEC.md"

    code = main(["logs", "--limit", "5"], logger=logger)
    out = capsys.readouterr().out
    assert code == 0
    assert "#1 [" in out
    assert "grounded=True" in out
