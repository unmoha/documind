"""Vector embedding generation modules.

Provides abstract interface BaseEmbedder, production SentenceTransformerEmbedder
for local BGE embedding, and deterministic FakeEmbedder for fast offline testing.
"""

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from documind.config.settings import get_settings


class BaseEmbedder(ABC):
    """Abstract base class for vector embedding generation."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a sequence of document chunk texts in batches.

        Does not add any query prefix to document chunks.

        Args:
            texts: Sequence of chunk text strings.

        Returns:
            List of float vector embeddings.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for semantic search retrieval.

        Args:
            text: Query text string.

        Returns:
            Float vector embedding.
        """
        ...


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local dense embedding model using sentence-transformers (BAAI/bge-small-en-v1.5).

    Batches document embeddings and normalizes vectors for cosine similarity.
    Does not add query instruction prefixes to document chunks.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the sentence-transformer embedder.

        Args:
            model_name: HuggingFace model identifier. Defaults to settings.embedding_model.
            device: Computing device ('cpu', 'cuda'). Defaults to settings.embedding_device.
            batch_size: Batch size for chunk embedding inference.
            normalize_embeddings: Whether to L2-normalize vectors to unit length.
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    def _get_model(self):
        """Lazy-load the underlying SentenceTransformer model instance."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                model_name_or_path=self.model_name,
                device=self.device,
            )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document chunk texts in batches without query prefixes.

        Args:
            texts: Sequence of chunk texts to encode.

        Returns:
            List of vector embeddings as lists of floats.
        """
        if not texts:
            return []

        model = self._get_model()
        # BGE document chunks should NOT have a query prefix applied
        embeddings = model.encode(
            sentences=list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query with the BGE retrieval prefix.

        Args:
            text: Search query string.

        Returns:
            Vector embedding as a list of floats.
        """
        model = self._get_model()
        query_text = f"Represent this sentence for searching relevant passages: {text}"
        embedding = model.encode(
            sentences=query_text,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embedding.tolist()


class FakeEmbedder(BaseEmbedder):
    """Deterministic, lightweight fake embedder for offline unit testing.

    Generates reproducible unit-normalized float vectors from content hashes
    without requiring PyTorch weights or downloading HuggingFace models.
    """

    def __init__(self, dimension: int = 384) -> None:
        """Initialize fake embedder with desired vector dimension.

        Args:
            dimension: Output embedding vector length. Defaults to 384
                (matching BAAI/bge-small-en-v1.5).
        """
        self.dimension = dimension

    def _hash_to_vector(self, text: str) -> list[float]:
        """Compute a deterministic unit-normalized float vector for input text."""
        raw_hash = hashlib.sha256(text.encode("utf-8")).digest()
        vector: list[float] = []
        for i in range(self.dimension):
            byte_val = raw_hash[i % len(raw_hash)]
            vector.append(float((byte_val ^ (i & 0xFF)) - 128))

        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [round(x / norm, 6) for x in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate fake embeddings for a list of document chunk texts."""
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Generate fake embedding for a query string."""
        return self._hash_to_vector(text)
