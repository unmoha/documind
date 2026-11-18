"""Pydantic request and response schemas for the DocuMind FastAPI backend."""

from pydantic import BaseModel, Field

from documind.generation.generator import Citation
from documind.logging.database import InteractionLog


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int
    chunks_replaced: int
    status: str


class DocumentListResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: list[dict]


class DeleteDocumentResponse(BaseModel):
    document_id: str
    chunks_deleted: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0)


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    has_relevant_context: bool
    best_score: float | None = None
    latency_ms: float


class LogsResponse(BaseModel):
    count: int
    logs: list[InteractionLog]
