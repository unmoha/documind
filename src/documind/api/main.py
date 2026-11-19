"""FastAPI application exposing ingestion, retrieval, QA, and audit endpoints."""

import hmac
import logging
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

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
from documind.logging.database import SQLiteLogger, scrub_secrets
from documind.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
UPLOAD_CHUNK_SIZE = 64 * 1024  # 64 KB read buffer
WINDOWS_INVALID_CHARS = set('<>:"/\\|?*')
WINDOWS_RESERVED_STEMS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def sanitize_filename(
    filename: str | None, default_fallback: str = "upload.txt"
) -> str:
    """Sanitize a filename while preserving Unicode letters and digits (e.g. Amharic).

    Replaces control characters, path separators, and Windows-invalid characters with '_',
    strips trailing dots and spaces, handles Windows reserved device names, limits length
    to 100 characters, and falls back to a safe default if empty.
    """
    if not filename:
        return default_fallback

    # Normalize unicode
    raw = unicodedata.normalize("NFC", str(filename))

    # Strip path separators by extracting basename across / and \
    raw = raw.replace("\\", "/").split("/")[-1]

    # Replace control characters, path separators, and Windows invalid chars with '_'
    chars: list[str] = []
    for ch in raw:
        if unicodedata.category(ch).startswith("C") or ch in WINDOWS_INVALID_CHARS:
            chars.append("_")
        else:
            chars.append(ch)

    sanitized = "".join(chars).rstrip(". ")
    if not sanitized or sanitized in {".", ".."}:
        return default_fallback

    suffix = Path(sanitized).suffix
    stem = sanitized[: -len(suffix)] if suffix else sanitized

    # Handle Windows reserved device names
    if stem.upper() in WINDOWS_RESERVED_STEMS:
        stem = f"_{stem.upper()}"
        sanitized = f"{stem}{suffix}"

    # Limit length to 100 characters, preserving the extension
    if len(sanitized) > 100:
        if suffix and len(suffix) < 100:
            stem = stem[: 100 - len(suffix)].rstrip(". ")
            sanitized = f"{stem}{suffix}"
        else:
            sanitized = sanitized[:100].rstrip(". ")

    if not sanitized:
        return default_fallback

    return sanitized


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

    @app.middleware("http")
    async def verify_api_token(request: Request, call_next):
        if request.url.path.startswith("/api/v1"):
            settings = get_settings()
            if settings.api_token is not None:
                expected_token = settings.api_token.get_secret_value()
                if expected_token:
                    provided_token = request.headers.get("X-API-Token")
                    if not provided_token or not hmac.compare_digest(
                        provided_token.encode("utf-8"),
                        expected_token.encode("utf-8"),
                    ):
                        return JSONResponse(
                            status_code=401,
                            content={"detail": "Invalid or missing API token."},
                        )
        return await call_next(request)

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
        settings = get_settings()
        filename = sanitize_filename(file.filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )

        # Enforce max upload size while reading in chunks
        max_size = settings.max_upload_size_bytes
        byte_chunks: list[bytes] = []
        total_read = 0

        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowed upload size of {max_size} bytes.",
                )
            byte_chunks.append(chunk)

        raw_bytes = b"".join(byte_chunks)

        # Validate content by magic bytes only (do not rely on client content-type)
        if suffix == ".pdf":
            if not raw_bytes.startswith(b"%PDF-"):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid PDF file: Missing %PDF- magic header.",
                )
        elif suffix in {".txt", ".md"}:
            if b"\x00" in raw_bytes:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid text file: File contains binary null bytes.",
                )
            try:
                raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid text file: Content must be valid UTF-8: {exc}",
                ) from exc

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
            safe_err = scrub_secrets(str(exc))
            logger.error("LLM generation error: %s", safe_err)
            raise HTTPException(
                status_code=502,
                detail="LLM generation service failed or is temporarily unavailable.",
            ) from None

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
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> LogsResponse:
        entries = svc.logger.get_recent_logs(limit=limit)
        return LogsResponse(count=len(entries), logs=entries)

    return app


app = create_app()
