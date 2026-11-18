"""Persistent vector storage implementation using ChromaDB.

Manages collection lifecycle, vector persistence, metadata indexing, and
idempotent document chunk upsert and deletion operations.
"""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from documind.config.settings import get_settings
from documind.ingestion.models import DocumentChunk

DEFAULT_COLLECTION_NAME = "documind_documents"


class ChromaVectorStore:
    """Persistent vector store wrapping a local ChromaDB collection."""

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Initialize persistent ChromaDB client and acquire collection.

        Args:
            persist_dir: Directory path for ChromaDB storage. Defaults to
                Settings.chroma_persist_dir.
            collection_name: Name of the vector collection. Defaults to
                'documind_documents'.
        """
        settings = get_settings()
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def has_document(self, file_hash: str) -> bool:
        """Check whether a document with the given content hash is already stored.

        Args:
            file_hash: SHA-256 content hash of the document.

        Returns:
            True if one or more chunks with this file_hash exist in the collection.
        """
        results = self.collection.get(
            where={"file_hash": file_hash},
            limit=1,
        )
        return bool(results and results.get("ids"))

    def get_document_chunks(self, file_hash: str) -> list[dict[str, Any]]:
        """Retrieve all indexed chunks and metadata for a given content hash.

        Args:
            file_hash: SHA-256 content hash of the document.

        Returns:
            List of chunk records with 'id', 'text', and 'metadata'.
        """
        results = self.collection.get(
            where={"file_hash": file_hash},
            include=["documents", "metadatas"],
        )
        chunks: list[dict[str, Any]] = []
        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []

        for i, chunk_id in enumerate(ids):
            chunks.append(
                {
                    "id": chunk_id,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return chunks

    def get_chunks_by_filename(self, filename: str) -> list[dict[str, Any]]:
        """Retrieve all indexed chunks and metadata for a given filename.

        Args:
            filename: Base filename of the document.

        Returns:
            List of chunk records with 'id', 'text', and 'metadata'.
        """
        results = self.collection.get(
            where={"filename": filename},
            include=["documents", "metadatas"],
        )
        chunks: list[dict[str, Any]] = []
        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []

        for i, chunk_id in enumerate(ids):
            chunks.append(
                {
                    "id": chunk_id,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return chunks

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Store document chunks and their vector embeddings in ChromaDB.

        Uses deterministic IDs based on file_hash and chunk_index to ensure
        idempotency and eliminate duplicate records.

        Args:
            chunks: List of DocumentChunk instances.
            embeddings: Corresponding vector embeddings for each chunk.

        Returns:
            Number of chunks added or updated.

        Raises:
            ValueError: If chunk count and embedding count do not match.
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between chunks count ({len(chunks)}) and "
                f"embeddings count ({len(embeddings)})"
            )

        ids = [f"{c.file_hash}_{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "filename": c.filename,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "file_hash": c.file_hash,
                "document_id": c.file_hash,
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def delete_document(self, file_hash: str) -> int:
        """Delete all chunks belonging to a document from the collection.

        Args:
            file_hash: SHA-256 content hash of the document to delete.

        Returns:
            Number of chunks that were removed.
        """
        existing = self.collection.get(where={"file_hash": file_hash})
        existing_ids = existing.get("ids") or []
        count = len(existing_ids)
        if count > 0:
            self.collection.delete(where={"file_hash": file_hash})
        return count

    def delete_by_filename(self, filename: str) -> int:
        """Delete all chunks belonging to a filename from the collection.

        Args:
            filename: Base filename of the document to delete.

        Returns:
            Number of chunks that were removed.
        """
        existing = self.collection.get(where={"filename": filename})
        existing_ids = existing.get("ids") or []
        count = len(existing_ids)
        if count > 0:
            self.collection.delete(where={"filename": filename})
        return count

    def delete_stale_chunks(self, filename: str, current_file_hash: str) -> int:
        """Delete all chunks for a filename whose file_hash does not match current_file_hash.

        Args:
            filename: Base filename of the document.
            current_file_hash: Current SHA-256 hash of the ingested document.

        Returns:
            Number of stale chunks that were removed.
        """
        existing = self.collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        existing_ids = existing.get("ids") or []
        existing_metas = existing.get("metadatas") or []

        stale_ids = [
            chunk_id
            for chunk_id, meta in zip(existing_ids, existing_metas, strict=False)
            if meta and meta.get("file_hash") != current_file_hash
        ]
        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return len(stale_ids)

    def count(self) -> int:
        """Return the total number of chunks stored across all documents.

        Returns:
            Integer chunk count.
        """
        return self.collection.count()

    def list_documents(self) -> list[dict[str, Any]]:
        """List distinct indexed documents and their chunk counts.

        Returns:
            List of dictionaries with 'document_id', 'filename', and 'chunk_count'.
        """
        results = self.collection.get(include=["metadatas"])
        docs: dict[str, dict[str, Any]] = {}
        for meta in results.get("metadatas") or []:
            if not meta:
                continue
            f_hash = str(meta.get("file_hash", ""))
            if not f_hash:
                continue
            if f_hash not in docs:
                docs[f_hash] = {
                    "document_id": f_hash,
                    "filename": meta.get("filename", ""),
                    "chunk_count": 0,
                }
            docs[f_hash]["chunk_count"] += 1
        return list(docs.values())
