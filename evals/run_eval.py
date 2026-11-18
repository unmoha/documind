"""Evaluation harness measuring retrieval hit rate, refusal accuracy, and latency."""

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from documind.generation.generator import AnswerGenerator, RAGPipeline
from documind.generation.llm import BaseLLMClient, GeminiLLMClient
from documind.ingestion.embedder import BaseEmbedder, SentenceTransformerEmbedder
from documind.ingestion.pipeline import ingest_file
from documind.ingestion.storage import ChromaVectorStore
from documind.retrieval.retriever import Retriever

DEFAULT_DATASET = Path(__file__).parent / "dataset.json"


class EvalCase(BaseModel):
    id: str
    question: str
    expected_filename: str | None = None
    expected_keywords: list[str] = []
    answerable: bool


class EvalReport(BaseModel):
    total_cases: int
    answerable_cases: int
    unanswerable_cases: int
    retrieval_hit_rate: float
    keyword_recall: float
    refusal_accuracy: float
    citation_rate: float
    p50_latency_ms: float
    p95_latency_ms: float


def load_dataset(path: Path = DEFAULT_DATASET) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(item) for item in raw]


def run_evaluation(
    cases: list[EvalCase],
    retriever: Retriever,
    generator: AnswerGenerator | None = None,
) -> EvalReport:
    """Evaluate retrieval accuracy and optional end-to-end grounded generation."""
    hits = 0
    keyword_hits = 0
    keyword_total = 0
    correct_refusals = 0
    cited_answers = 0
    latencies: list[float] = []

    answerable = [c for c in cases if c.answerable]
    unanswerable = [c for c in cases if not c.answerable]

    pipeline = (
        RAGPipeline(retriever=retriever, generator=generator)
        if generator is not None
        else None
    )

    for case in cases:
        t0 = time.perf_counter()
        retrieval = retriever.retrieve(case.question)
        if pipeline is not None:
            gen_result = pipeline.ask(case.question)
            elapsed = (time.perf_counter() - t0) * 1000
        else:
            gen_result = None
            elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        if case.answerable:
            retrieved_files = {c.filename for c in retrieval.chunks}
            if retrieval.has_relevant_context and (
                case.expected_filename is None
                or case.expected_filename in retrieved_files
            ):
                hits += 1

            combined_text = " ".join(c.text for c in retrieval.chunks).lower()
            for kw in case.expected_keywords:
                keyword_total += 1
                if kw.lower() in combined_text:
                    keyword_hits += 1

            if (
                gen_result is not None
                and gen_result.has_relevant_context
                and gen_result.citations
            ):
                cited_answers += 1
        else:
            if not retrieval.has_relevant_context:
                correct_refusals += 1

    sorted_lat = sorted(latencies) if latencies else [0.0]
    p95_idx = min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))

    return EvalReport(
        total_cases=len(cases),
        answerable_cases=len(answerable),
        unanswerable_cases=len(unanswerable),
        retrieval_hit_rate=round(hits / max(1, len(answerable)), 4),
        keyword_recall=round(keyword_hits / max(1, keyword_total), 4),
        refusal_accuracy=round(correct_refusals / max(1, len(unanswerable)), 4),
        citation_rate=(
            round(cited_answers / max(1, len(answerable)), 4)
            if generator is not None
            else 0.0
        ),
        p50_latency_ms=round(statistics.median(sorted_lat), 2),
        p95_latency_ms=round(sorted_lat[p95_idx], 2),
    )


def main(
    argv: Sequence[str] | None = None,
    embedder: BaseEmbedder | None = None,
    llm: BaseLLMClient | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="evals.run_eval")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also run end-to-end LLM generation and measure citation rate.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args(argv)

    emb = embedder or SentenceTransformerEmbedder()
    store = ChromaVectorStore()
    if store.count() == 0:
        for doc in [Path("docs/SPEC.md"), Path("README.md")]:
            if doc.is_file():
                ingest_file(doc, embedder=emb, store=store)

    retriever = Retriever(embedder=emb, store=store)
    generator = AnswerGenerator(llm=llm or GeminiLLMClient()) if args.with_llm else None

    cases = load_dataset(args.dataset)
    report = run_evaluation(cases=cases, retriever=retriever, generator=generator)

    print("=== DocuMind Evaluation Report ===")
    print(
        f"Total cases:        {report.total_cases} ({report.answerable_cases} answerable, {report.unanswerable_cases} unanswerable)"
    )
    print(f"Retrieval Hit Rate: {report.retrieval_hit_rate * 100:.1f}%")
    print(f"Keyword Recall:     {report.keyword_recall * 100:.1f}%")
    print(f"Refusal Accuracy:   {report.refusal_accuracy * 100:.1f}%")
    if args.with_llm:
        print(f"Citation Rate:      {report.citation_rate * 100:.1f}%")
    print(f"p50 Latency:        {report.p50_latency_ms:.1f} ms")
    print(f"p95 Latency:        {report.p95_latency_ms:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
