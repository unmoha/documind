"""Unit tests for the DocuMind evaluation harness."""

from documind.generation.generator import AnswerGenerator
from documind.generation.llm import FakeLLMClient
from documind.ingestion.embedder import FakeEmbedder
from documind.ingestion.storage import ChromaVectorStore
from documind.retrieval.retriever import Retriever
from evals.run_eval import EvalCase, load_dataset, main, run_evaluation


def test_load_default_dataset():
    cases = load_dataset()
    assert len(cases) >= 4
    assert any(c.answerable for c in cases)
    assert any(not c.answerable for c in cases)


def test_run_evaluation_computes_metrics(tmp_path):
    emb = FakeEmbedder()
    store = ChromaVectorStore(persist_dir=tmp_path)
    text = "DocuMind targets 90% retrieval hit rate using ChromaDB."
    store.collection.upsert(
        ids=["h1_0"],
        documents=[text],
        metadatas=[
            {
                "filename": "SPEC.md",
                "page_number": 1,
                "chunk_index": 0,
                "file_hash": "h1",
                "document_id": "h1",
            }
        ],
        embeddings=emb.embed_documents([text]),
    )

    cases = [
        EvalCase(
            id="hit",
            question=text,
            expected_filename="SPEC.md",
            expected_keywords=["90%", "ChromaDB"],
            answerable=True,
        ),
        EvalCase(
            id="miss",
            question="completely unrelated quantum banana recipe",
            answerable=False,
        ),
    ]

    retriever = Retriever(embedder=emb, store=store, top_k=2, threshold=0.95)
    generator = AnswerGenerator(llm=FakeLLMClient("Hit rate is 90% [1]."))
    report = run_evaluation(cases=cases, retriever=retriever, generator=generator)

    assert report.retrieval_hit_rate == 1.0
    assert report.keyword_recall == 1.0
    assert report.refusal_accuracy == 1.0
    assert report.citation_rate == 1.0
    assert (
        main(["--dataset", "evals/dataset.json"], embedder=emb, llm=FakeLLMClient())
        == 0
    )
