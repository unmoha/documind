"""FastAPI application exposing ingestion, retrieval, QA, and audit endpoints."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile

from documind.api.schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    HealthResponse,
    LogsResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from documind.config.settings import get_settings
from documind.generation.generator import AnswerGenerator, RAGPipeline
from documind.generation.llm import GeminiLLMClient
from documind.ingestion.embedder import BaseEmbedder, SentenceTransformerEmbedder
from documind.ingestion.errors import EmptyPDFError
from documind.ingestion.pipeline import ingest_file
from documind.ingestion.storage import ChromaVectorStore
from documind.logging.database import SQLiteLogger
from documind.retrieval.retriever import Retriever

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class AppServices:
    """Container of core backend services attached to app.state."""

    embedder: BaseEmbedder
    store: ChromaVectorStore
    pipeline: RAGPipeline
    logger: SQLiteLogger


def build_default_services() -> AppServices:
    """Instantiate production services backed by settings."""
    embedder = SentenceTransformerEmbedder()
    store = ChromaVectorStore()
    logger = SQLiteLogger()
    retriever = Retriever(embedder=embedder, store=store)
    generator = AnswerGenerator(llm=GeminiLLMClient())
    pipeline = RAGPipeline(retriever=retriever, generator=generator, logger=logger)
    return AppServices(
        embedder=embedder,
        store=store,
        pipeline=pipeline,
        logger=logger,
    )


def get_services(request: Request) -> AppServices:
    """Resolve AppServices from app.state, initializing defaults on first use."""
    services: AppServices | None = getattr(request.app.state, "services", None)
    if services is None:
        services = build_default_services()
        request.app.state.services = services
    return services


ServicesDep = Annotated[AppServices, Depends(get_services)]


def create_app(services: AppServices | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="DocuMind API",
        version="0.1.0",
        description="Production-quality RAG backend with strict grounded citations.",
    )
    if services is not None:
        app.state.services = services

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        settings = get_settings()
        return HealthResponse(
            status="ok",
            version="0.1.0",
            environment=settings.env,
        )

    @app.post("/api/v1/documents/upload", response_model=UploadResponse)
    async def upload_document(
        file: Annotated[UploadFile, File(...)],
        svc: ServicesDep,
    ) -> UploadResponse:
        filename = Path(file.filename or "upload.txt").name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )

        raw_bytes = await file.read()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / filename
            tmp_path.write_bytes(raw_bytes)
            try:
                result = ingest_file(
                    tmp_path,
                    embedder=svc.embedder,
                    store=svc.store,
                )
            except EmptyPDFError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return UploadResponse(
            document_id=result.document_id,
            filename=result.filename,
            chunks_indexed=result.chunks_indexed,
            chunks_replaced=result.chunks_replaced,
            status=result.status,
        )

    @app.get("/api/v1/documents", response_model=DocumentListResponse)
    def list_documents(svc: ServicesDep) -> DocumentListResponse:
        docs = svc.store.list_documents()
        return DocumentListResponse(
            total_documents=len(docs),
            total_chunks=svc.store.count(),
            documents=docs,
        )

    @app.delete(
        "/api/v1/documents/{document_id}", response_model=DeleteDocumentResponse
    )
    def delete_document(
        document_id: str,
        svc: ServicesDep,
    ) -> DeleteDocumentResponse:
        deleted = svc.store.delete_document(document_id)
        if deleted == 0:
            deleted = svc.store.delete_by_filename(document_id)
        if deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_id}' not found.",
            )
        return DeleteDocumentResponse(
            document_id=document_id,
            chunks_deleted=deleted,
        )

    @app.post("/api/v1/query", response_model=QueryResponse)
    def query_documents(
        payload: QueryRequest,
        svc: ServicesDep,
    ) -> QueryResponse:
        try:
            result = svc.pipeline.ask(payload.question, top_k=payload.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return QueryResponse(
            question=result.question,
            answer=result.answer,
            citations=result.citations,
            has_relevant_context=result.has_relevant_context,
            best_score=result.best_score,
            latency_ms=result.latency_ms,
        )

    @app.get("/api/v1/logs", response_model=LogsResponse)
    def get_logs(
        svc: ServicesDep,
        limit: int = 50,
    ) -> LogsResponse:
        entries = svc.logger.get_recent_logs(limit=limit)
        return LogsResponse(count=len(entries), logs=entries)

    return app


app = create_app()
