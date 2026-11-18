"""Document ingestion, parsing, chunking, embedding, and vector storage pipeline."""

from documind.ingestion.chunker import (
    RecursiveCharacterChunker,
    chunk_document,
)
from documind.ingestion.embedder import (
    BaseEmbedder,
    FakeEmbedder,
    SentenceTransformerEmbedder,
)
from documind.ingestion.errors import (
    EmptyPDFError,
    IngestionError,
)
from documind.ingestion.loaders import (
    compute_file_hash,
    load_document,
    load_markdown,
    load_pdf,
    load_txt,
)
from documind.ingestion.models import (
    DocumentChunk,
    DocumentPage,
    IngestionResult,
)
from documind.ingestion.pipeline import (
    ingest_file,
)
from documind.ingestion.storage import (
    ChromaVectorStore,
)

__all__ = [
    "BaseEmbedder",
    "ChromaVectorStore",
    "DocumentChunk",
    "DocumentPage",
    "EmptyPDFError",
    "FakeEmbedder",
    "IngestionError",
    "IngestionResult",
    "RecursiveCharacterChunker",
    "SentenceTransformerEmbedder",
    "chunk_document",
    "compute_file_hash",
    "ingest_file",
    "load_document",
    "load_markdown",
    "load_pdf",
    "load_txt",
]
