"""Tests for retrieval, using the fake embedder (no model download)."""

import pytest

from documind.ingestion.embedder import FakeEmbedder
from documind.ingestion.storage import ChromaVectorStore
from documind.retrieval import Retriever
from documind.retrieval.cli import main

ITEMS = [
    ("alpha.txt", "Alpha explains how cats sleep during the day."),
    ("beta.txt", "Beta describes how to brew coffee with a paper filter."),
    ("gamma.txt", "Gamma covers the history of the printing press."),
    ("delta.txt", "Delta lists the planets of the solar system."),
    ("epsilon.txt", "Epsilon is a guide to growing tomatoes in pots."),
]


@pytest.fixture
def emb() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def empty_store(tmp_path) -> ChromaVectorStore:
    return ChromaVectorStore(persist_dir=tmp_path)


@pytest.fixture
def filled_store(empty_store, emb) -> ChromaVectorStore:
    texts = [text for _, text in ITEMS]
    empty_store.collection.upsert(
        ids=[f"hash{i}_0" for i in range(len(ITEMS))],
        documents=texts,
        metadatas=[
            {
                "filename": name,
                "page_number": 1,
                "chunk_index": 0,
                "file_hash": f"hash{i}",
                "document_id": f"hash{i}",
            }
            for i, (name, _) in enumerate(ITEMS)
        ],
        embeddings=emb.embed_documents(texts),
    )
    return empty_store


def test_ranking_order_and_exact_match_first(filled_store, emb):
    retriever = Retriever(embedder=emb, store=filled_store, top_k=5, threshold=0.0)
    result = retriever.retrieve(ITEMS[1][1])
    assert result.chunks[0].filename == "beta.txt"
    assert result.chunks[0].score == pytest.approx(1.0, abs=1e-3)
    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results(filled_store, emb):
    retriever = Retriever(embedder=emb, store=filled_store, top_k=2, threshold=0.0)
    assert len(retriever.retrieve(ITEMS[0][1]).chunks) == 2
    assert len(retriever.retrieve(ITEMS[0][1], top_k=4).chunks) == 4


def test_threshold_returns_explicit_no_context(filled_store, emb):
    retriever = Retriever(embedder=emb, store=filled_store, top_k=3, threshold=0.99)
    miss = retriever.retrieve("something totally unrelated about rockets")
    assert miss.has_relevant_context is False
    assert miss.chunks == []
    assert miss.best_score is not None

    hit = retriever.retrieve(ITEMS[2][1])
    assert hit.has_relevant_context is True
    assert hit.chunks[0].filename == "gamma.txt"


def test_empty_collection(empty_store, emb):
    result = Retriever(embedder=emb, store=empty_store, threshold=0.0).retrieve("hi")
    assert result.chunks == []
    assert result.best_score is None
    assert result.has_relevant_context is False


def test_empty_query_and_bad_top_k_raise(filled_store, emb):
    retriever = Retriever(embedder=emb, store=filled_store, threshold=0.0)
    with pytest.raises(ValueError):
        retriever.retrieve("   ")
    with pytest.raises(ValueError):
        retriever.retrieve("cats", top_k=0)


def test_cli_search_prints_ranked_results(filled_store, emb, capsys):
    retriever = Retriever(embedder=emb, store=filled_store, top_k=3, threshold=0.0)
    code = main(["search", ITEMS[0][1]], retriever=retriever)
    out = capsys.readouterr().out
    assert code == 0
    assert "1. [" in out
    assert "alpha.txt" in out
