"""Orchestrator pipeline for document ingestion, embedding, and vector storage."""

from pathlib import Path

from documind.ingestion.chunker import RecursiveCharacterChunker, chunk_document
from documind.ingestion.embedder import BaseEmbedder, SentenceTransformerEmbedder
from documind.ingestion.errors import EmptyPDFError
from documind.ingestion.loaders import compute_file_hash, load_document
from documind.ingestion.models import IngestionResult
from documind.ingestion.storage import ChromaVectorStore


def ingest_file(
    file_path: Path | str,
    embedder: BaseEmbedder | None = None,
    store: ChromaVectorStore | None = None,
    chunker: RecursiveCharacterChunker | None = None,
    force: bool = False,
) -> IngestionResult:
    """Ingest a single document into the persistent vector database.

    Extracts text, splits into recursive character chunks, computes local
    dense vector embeddings without query prefixes, and stores records
    in ChromaDB with source citation metadata.

    Ingestion is strictly idempotent: re-ingesting an already-indexed document
    (matching SHA-256 content hash) returns immediately without creating duplicates
    or recomputing embeddings unless force=True.

    Args:
        file_path: Path to the PDF, Markdown, or text file.
        embedder: Embedding engine instance. Defaults to SentenceTransformerEmbedder.
        store: ChromaVectorStore instance. Defaults to ChromaVectorStore with settings path.
        chunker: Text chunker instance. Defaults to RecursiveCharacterChunker.
        force: If True, overwrite and re-index an already stored document.

    Returns:
        IngestionResult with document_id, filename, chunk count, and status.

    Raises:
        FileNotFoundError: If the source file does not exist.
        EmptyPDFError: If a PDF file yields no extractable text.
        ValueError: If file format is unsupported or parsing fails.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    # Compute content hash of raw bytes
    file_bytes = path.read_bytes()
    file_hash = compute_file_hash(file_bytes)

    # Initialize store if not provided
    if store is None:
        store = ChromaVectorStore()

    # Idempotency check: if already indexed and not force, return immediately
    if not force and store.has_document(file_hash):
        existing_chunks = store.get_document_chunks(file_hash)
        return IngestionResult(
            document_id=file_hash,
            filename=path.name,
            file_hash=file_hash,
            chunks_indexed=len(existing_chunks),
            chunks_replaced=0,
            status="already_indexed",
        )

    # Load document pages
    pages = load_document(path)

    # Validate PDF extractable text
    is_pdf = path.suffix.lower() == ".pdf"
    if is_pdf and (not pages or not any(p.text.strip() for p in pages)):
        raise EmptyPDFError(
            f"PDF file '{path.name}' yielded no extractable text. "
            "The document may be empty, image-only, or scanned without an OCR layer."
        )

    # Chunk pages
    chunks = chunk_document(pages, chunker=chunker)
    if not chunks:
        return IngestionResult(
            document_id=file_hash,
            filename=path.name,
            file_hash=file_hash,
            chunks_indexed=0,
            chunks_replaced=0,
            status="skipped",
        )

    # Initialize embedder only when needed
    if embedder is None:
        embedder = SentenceTransformerEmbedder()

    # Embed chunks in batches without query prefix
    chunk_texts = [c.text for c in chunks]
    embeddings = embedder.embed_documents(chunk_texts)

    # If stale chunks exist for the same filename with a different hash, purge them
    replaced_count = store.delete_stale_chunks(
        filename=path.name,
        current_file_hash=file_hash,
    )

    # If force re-indexing with identical hash, remove old chunks first
    if force and store.has_document(file_hash):
        store.delete_document(file_hash)

    # Store in ChromaDB
    store.add_chunks(chunks=chunks, embeddings=embeddings)

    return IngestionResult(
        document_id=file_hash,
        filename=path.name,
        file_hash=file_hash,
        chunks_indexed=len(chunks),
        chunks_replaced=replaced_count,
        status="indexed",
    )
