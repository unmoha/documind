"""Unit tests for grounded generation and the RAG pipeline (no API calls)."""

from documind.generation import (
    AnswerGenerator,
    FakeLLMClient,
    RAGPipeline,
)
from documind.generation.cli import main
from documind.generation.prompts import REFUSAL_MESSAGE
from documind.ingestion.embedder import FakeEmbedder
from documind.ingestion.storage import ChromaVectorStore
from documind.retrieval import RetrievalResult, RetrievedChunk, Retriever


def _sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text="DocuMind targets a retrieval hit rate of at least 90 percent.",
            filename="SPEC.md",
            page_number=1,
            chunk_index=18,
            score=0.82,
        ),
        RetrievedChunk(
            text="End-to-end query latency must stay under 3 seconds.",
            filename="SPEC.md",
            page_number=1,
            chunk_index=19,
            score=0.75,
        ),
    ]


def test_refuses_without_calling_llm_when_no_relevant_context():
    fake_llm = FakeLLMClient()
    generator = AnswerGenerator(llm=fake_llm)
    retrieval = RetrievalResult(
        query="What is the best recipe for injera?",
        chunks=[],
        best_score=0.47,
        threshold=0.50,
        has_relevant_context=False,
    )

    result = generator.generate(retrieval)

    assert result.has_relevant_context is False
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == []
    assert fake_llm.calls == []


def test_generates_grounded_answer_with_numbered_citations():
    fake_llm = FakeLLMClient(
        canned_response="The retrieval hit rate target is >= 90% [1], and latency must be under 3s [2]."
    )
    generator = AnswerGenerator(llm=fake_llm)
    retrieval = RetrievalResult(
        query="What are the success metrics?",
        chunks=_sample_chunks(),
        best_score=0.82,
        threshold=0.50,
        has_relevant_context=True,
    )

    result = generator.generate(retrieval)

    assert result.has_relevant_context is True
    assert "[1]" in result.answer
    assert len(result.citations) == 2
    assert result.citations[0].filename == "SPEC.md"
    assert result.citations[0].source_number == 1
    assert len(fake_llm.calls) == 1
    _, user_prompt = fake_llm.calls[0]
    assert "[1] Source: SPEC.md" in user_prompt
    assert "What are the success metrics?" in user_prompt


def test_end_to_end_rag_pipeline_and_cli(tmp_path, capsys):
    emb = FakeEmbedder()
    store = ChromaVectorStore(persist_dir=tmp_path)
    text = "ChromaDB stores persistent vector embeddings locally."
    store.collection.upsert(
        ids=["h1_0"],
        documents=[text],
        metadatas=[
            {
                "filename": "SPEC.md",
                "page_number": 1,
                "chunk_index": 3,
                "file_hash": "h1",
                "document_id": "h1",
            }
        ],
        embeddings=emb.embed_documents([text]),
    )

    retriever = Retriever(embedder=emb, store=store, top_k=2, threshold=0.0)
    fake_llm = FakeLLMClient(
        "ChromaDB stores persistent vector embeddings locally [1]."
    )
    pipeline = RAGPipeline(retriever=retriever, generator=AnswerGenerator(fake_llm))

    code = main(["ask", text], pipeline=pipeline)
    out = capsys.readouterr().out

    assert code == 0
    assert "ChromaDB stores persistent vector embeddings locally [1]." in out
    assert "[1] SPEC.md" in out
