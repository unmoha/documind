"""Command-line interface for document ingestion and collection inspection.

Provides minimal commands for adding documents to the vector store and
checking total chunk counts:
- `uv run python -m documind.ingestion.cli ingest <path>`
- `uv run python -m documind.ingestion.cli delete <filename>`
- `uv run python -m documind.ingestion.cli count`
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from documind.ingestion.embedder import BaseEmbedder, SentenceTransformerEmbedder
from documind.ingestion.pipeline import ingest_file
from documind.ingestion.storage import ChromaVectorStore


def create_parser() -> argparse.ArgumentParser:
    """Construct argument parser with subcommands for ingestion and collection queries."""
    parser = argparse.ArgumentParser(
        prog="python -m documind.ingestion.cli",
        description="DocuMind ingestion CLI: ingest documents and inspect chunk collections.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Subcommand to execute.",
    )

    # Ingest subcommand
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a document file (PDF, Markdown, or text) into ChromaDB.",
    )
    ingest_parser.add_argument(
        "path",
        type=Path,
        help="Path to the document file to ingest.",
    )

    # Count subcommand
    subparsers.add_parser(
        "count",
        help="Print the total number of chunks stored in the collection.",
    )

    # Delete subcommand
    delete_parser = subparsers.add_parser(
        "delete",
        help="Remove all chunks for a document filename from ChromaDB.",
    )
    delete_parser.add_argument(
        "filename",
        type=str,
        help="Filename or path of the document whose chunks should be deleted.",
    )

    return parser


def handle_ingest(
    file_path: Path,
    embedder: BaseEmbedder | None = None,
    store: ChromaVectorStore | None = None,
) -> int:
    """Ingest a single document and print chunk indexing summary.

    Args:
        file_path: Path to the target file.
        embedder: Optional embedding instance (defaults to SentenceTransformerEmbedder).
        store: Optional vector store instance (defaults to ChromaVectorStore).

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    if embedder is None:
        embedder = SentenceTransformerEmbedder()
    if store is None:
        store = ChromaVectorStore()

    try:
        result = ingest_file(file_path=file_path, embedder=embedder, store=store)
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"Ingestion failed: {err}", file=sys.stderr)
        return 1

    if result.status == "already_indexed":
        added = 0
        skipped = result.chunks_indexed
    elif result.status == "indexed":
        added = result.chunks_indexed
        skipped = 0
    else:
        added = 0
        skipped = 0

    if result.chunks_replaced > 0:
        print(f"Added {added} chunks, replaced {result.chunks_replaced} old chunks.")
    else:
        print(f"Added {added} chunks, skipped {skipped} chunks as duplicates.")
    return 0


def handle_delete(
    filename: str,
    store: ChromaVectorStore | None = None,
) -> int:
    """Delete all chunks for a filename and print summary.

    Args:
        filename: Name or path of the document to delete.
        store: Optional vector store instance (defaults to ChromaVectorStore).

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    if store is None:
        store = ChromaVectorStore()

    target_name = Path(filename).name
    try:
        removed = store.delete_by_filename(target_name)
    except Exception as err:
        print(f"Delete failed: {err}", file=sys.stderr)
        return 1

    print(f"Removed {removed} chunks for '{target_name}'.")
    return 0


def handle_count(store: ChromaVectorStore | None = None) -> int:
    """Print the number of chunks in the collection.

    Args:
        store: Optional vector store instance (defaults to ChromaVectorStore).

    Returns:
        Exit code: 0 on success.
    """
    if store is None:
        store = ChromaVectorStore()

    chunk_count = store.count()
    print(f"Chunks in collection: {chunk_count}")
    return 0


def main(
    argv: Sequence[str] | None = None,
    embedder: BaseEmbedder | None = None,
    store: ChromaVectorStore | None = None,
) -> int:
    """CLI entrypoint dispatching commands.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].
        embedder: Optional embedder to use for ingestion (defaults to real embedder).
        store: Optional vector store instance (defaults to persistent ChromaVectorStore).

    Returns:
        Integer process exit code.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return handle_ingest(file_path=args.path, embedder=embedder, store=store)
    elif args.command == "count":
        return handle_count(store=store)
    elif args.command == "delete":
        return handle_delete(filename=args.filename, store=store)

    return 0


if __name__ == "__main__":
    sys.exit(main())
