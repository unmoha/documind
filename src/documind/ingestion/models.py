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
