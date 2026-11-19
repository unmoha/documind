"""Query-time retrieval over the persistent ChromaDB collection."""

from pydantic import BaseModel, Field

from documind.config.settings import get_settings
from documind.ingestion.embedder import BaseEmbedder
from documind.ingestion.storage import ChromaVectorStore


class RetrievedChunk(BaseModel):
    """One retrieved chunk with its provenance and similarity score."""

    text: str
    filename: str
    page_number: int
    chunk_index: int
    score: float = Field(
        ge=0.0, le=1.0, description="Cosine similarity, higher is better."
    )


class RetrievalResult(BaseModel):
    """Outcome of a retrieval call, including the explicit no-context case."""

    query: str
    chunks: list[RetrievedChunk]
    best_score: float | None = None
    threshold: float
    has_relevant_context: bool


def distance_to_similarity(distance: float) -> float:
    """Convert Chroma cosine distance (1 - cosine similarity) to a 0..1 similarity."""
    return max(0.0, min(1.0, 1.0 - distance))


class Retriever:
    """Embeds a query and returns the most similar stored chunks."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: ChromaVectorStore,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> None:
        """Create a retriever.

        Args:
            embedder: Embedder used for queries; must match the document embedder.
            store: Vector store holding the chunks.
            top_k: Number of chunks to return. Defaults to settings.top_k.
            threshold: Minimum best score for context to count as relevant.
                Defaults to settings.similarity_threshold.
        """
        settings = get_settings()
        self.embedder = embedder
        self.store = store
        self.top_k = top_k if top_k is not None else settings.top_k
        self.threshold = (
            threshold if threshold is not None else settings.similarity_threshold
        )

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Return the top_k most similar chunks, or an explicit no-context result.

        Raises:
            ValueError: If the query is empty or top_k is not positive.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")
        if len(query) > 2000:
            raise ValueError("Query length must not exceed 2000 characters.")
        k = top_k if top_k is not None else self.top_k
        if k <= 0 or k > 20:
            raise ValueError("top_k must be between 1 and 20.")

        total = self.store.count()
        if total == 0:
            return RetrievalResult(
                query=query,
                chunks=[],
                best_score=None,
                threshold=self.threshold,
                has_relevant_context=False,
            )

        raw = self.store.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for text, meta, dist in zip(docs, metas, dists, strict=True):
            meta = meta or {}
            chunks.append(
                RetrievedChunk(
                    text=text,
                    filename=str(meta.get("filename", "")),
                    page_number=int(meta.get("page_number", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=distance_to_similarity(float(dist)),
                )
            )
        chunks.sort(key=lambda c: c.score, reverse=True)

        best = chunks[0].score if chunks else None
        if best is None or best < self.threshold:
            return RetrievalResult(
                query=query,
                chunks=[],
                best_score=best,
                threshold=self.threshold,
                has_relevant_context=False,
            )
        return RetrievalResult(
            query=query,
            chunks=chunks,
            best_score=best,
            threshold=self.threshold,
            has_relevant_context=True,
        )
