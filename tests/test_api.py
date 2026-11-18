"""FastAPI endpoint tests using FastAPI TestClient and fake models."""

from fastapi.testclient import TestClient

from documind.api.main import AppServices, create_app
from documind.generation.generator import AnswerGenerator, RAGPipeline
from documind.generation.llm import FakeLLMClient
from documind.ingestion.embedder import FakeEmbedder
from documind.ingestion.storage import ChromaVectorStore
from documind.logging.database import SQLiteLogger
from documind.retrieval.retriever import Retriever


def _make_test_client(tmp_path) -> TestClient:
    embedder = FakeEmbedder()
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma")
    logger = SQLiteLogger(db_path=tmp_path / "logs.sqlite")
    retriever = Retriever(embedder=embedder, store=store, top_k=3, threshold=0.0)
    llm = FakeLLMClient("DocuMind uses ChromaDB for local vector storage [1].")
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=AnswerGenerator(llm=llm),
        logger=logger,
    )
    services = AppServices(
        embedder=embedder,
        store=store,
        pipeline=pipeline,
        logger=logger,
    )
    return TestClient(create_app(services=services))


def test_health_endpoint(tmp_path):
    client = _make_test_client(tmp_path)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"


def test_upload_list_query_logs_and_delete_lifecycle(tmp_path):
    client = _make_test_client(tmp_path)

    # 1. Upload a markdown document
    content = b"# Architecture\nDocuMind uses ChromaDB for local vector storage."
    upload_resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("arch.md", content, "text/markdown")},
    )
    assert upload_resp.status_code == 200
    up_data = upload_resp.json()
    assert up_data["filename"] == "arch.md"
    assert up_data["chunks_indexed"] >= 1
    doc_id = up_data["document_id"]

    # 2. List documents
    list_resp = client.get("/api/v1/documents")
    assert list_resp.status_code == 200
    assert list_resp.json()["total_documents"] == 1

    # 3. Query endpoint
    query_resp = client.post(
        "/api/v1/query",
        json={"question": "What does DocuMind use for vector storage?"},
    )
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert q_data["has_relevant_context"] is True
    assert "[1]" in q_data["answer"]
    assert q_data["citations"][0]["filename"] == "arch.md"

    # 4. Audit logs endpoint
    logs_resp = client.get("/api/v1/logs?limit=10")
    assert logs_resp.status_code == 200
    assert logs_resp.json()["count"] == 1

    # 5. Delete document
    del_resp = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["chunks_deleted"] >= 1


def test_upload_rejects_unsupported_extension(tmp_path):
    client = _make_test_client(tmp_path)
    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
    )
    assert resp.status_code == 400
