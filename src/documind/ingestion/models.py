"""Data models for document ingestion and text chunking."""

from pydantic import BaseModel, ConfigDict, Field


class DocumentPage(BaseModel):
    """Represents an extracted page or slice from a source document."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        description="Extracted raw text content of the page.",
    )
    filename: str = Field(
        description="Source document filename (e.g., report.pdf).",
    )
    page_number: int = Field(
        default=1,
        ge=1,
        description="1-based page number within the document.",
    )
    file_hash: str = Field(
        description="SHA-256 hex digest of the raw source file contents.",
    )


class DocumentChunk(BaseModel):
    """Represents a discrete text chunk ready for vector embedding and retrieval."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        description="Text content of the chunk.",
    )
    filename: str = Field(
        description="Source document filename.",
    )
    page_number: int = Field(
        ge=1,
        description="1-based page number where the chunk originated.",
    )
    chunk_index: int = Field(
        ge=0,
        description="0-based sequential index of the chunk within the document.",
    )
    file_hash: str = Field(
        description="SHA-256 hex digest of the raw source file.",
    )


class IngestionResult(BaseModel):
    """Result summary of an ingestion operation for a single source file."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(
        description="Unique identifier for the document (content hash).",
    )
    filename: str = Field(
        description="Filename of the ingested document.",
    )
    file_hash: str = Field(
        description="SHA-256 hexadecimal digest of the raw document content.",
    )
    chunks_indexed: int = Field(
        ge=0,
        description="Number of chunks stored in vector database.",
    )
    chunks_replaced: int = Field(
        default=0,
        ge=0,
        description="Number of stale chunks replaced during re-ingestion of a modified file.",
    )
    status: str = Field(
        description="Outcome of the ingestion operation (indexed, already_indexed, skipped).",
    )
