"""Command-line interface for end-to-end grounded question answering."""

import argparse
from collections.abc import Sequence

from documind.generation.generator import AnswerGenerator, RAGPipeline
from documind.generation.llm import GeminiLLMClient
from documind.retrieval.retriever import Retriever


def handle_ask(pipeline: RAGPipeline, question: str, top_k: int | None) -> int:
    """Run retrieve + generate and display the answer with citations."""
    result = pipeline.ask(question, top_k=top_k)
    print(f"Question: {result.question}")
    print(f"Latency:  {result.latency_ms:.1f} ms")
    print("\nAnswer:")
    print(result.answer)
    if result.citations:
        print("\nSources:")
        for c in result.citations:
            print(
                f"  [{c.source_number}] {c.filename} "
                f"(p{c.page_number}, chunk #{c.chunk_index}, score={c.score:.3f})"
            )
    return 0


def main(argv: Sequence[str] | None = None, pipeline: RAGPipeline | None = None) -> int:
    """CLI entry point. A RAGPipeline can be injected for testing."""
    parser = argparse.ArgumentParser(prog="documind.generation.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="Ask a grounded question over ingested documents.")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)

    if pipeline is None:
        from documind.ingestion.embedder import SentenceTransformerEmbedder
        from documind.ingestion.storage import ChromaVectorStore

        retriever = Retriever(
            embedder=SentenceTransformerEmbedder(),
            store=ChromaVectorStore(),
        )
        generator = AnswerGenerator(llm=GeminiLLMClient())
        pipeline = RAGPipeline(retriever=retriever, generator=generator)

    return handle_ask(pipeline, args.question, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
