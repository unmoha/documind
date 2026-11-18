"""Command-line interface for searching the DocuMind collection."""

import argparse
from collections.abc import Sequence

from documind.retrieval.retriever import Retriever


def _preview(text: str, limit: int = 150) -> str:
    """Collapse whitespace and cut text to a short single-line preview."""
    return " ".join(text.split())[:limit]


def handle_search(retriever: Retriever, question: str, top_k: int | None) -> int:
    """Run a search and print ranked results. Returns a process exit code."""
    result = retriever.retrieve(question, top_k=top_k)
    print(f"Query: {result.query}")
    if result.best_score is None:
        print("The collection is empty. Ingest a document first.")
        return 1

    print(f"Best score: {result.best_score:.3f} (threshold {result.threshold:.2f})")
    if not result.has_relevant_context:
        print("No relevant context found.")
        return 0

    for rank, chunk in enumerate(result.chunks, start=1):
        print(
            f"{rank}. [{chunk.score:.3f}] {chunk.filename} "
            f"p{chunk.page_number} #{chunk.chunk_index}"
        )
        print(f"   {_preview(chunk.text)}")
    return 0


def main(argv: Sequence[str] | None = None, retriever: Retriever | None = None) -> int:
    """CLI entry point. A retriever can be injected for testing."""
    parser = argparse.ArgumentParser(prog="documind.retrieval.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search", help="Search stored chunks for a question.")
    search.add_argument("question")
    search.add_argument("--top-k", type=int, default=None)
    search.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum best score; defaults to settings.similarity_threshold.",
    )
    args = parser.parse_args(argv)

    if retriever is None:
        from documind.ingestion.embedder import SentenceTransformerEmbedder
        from documind.ingestion.storage import ChromaVectorStore

        retriever = Retriever(
            embedder=SentenceTransformerEmbedder(),
            store=ChromaVectorStore(),
            threshold=args.threshold,
        )
    return handle_search(retriever, args.question, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
