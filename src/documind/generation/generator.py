"""Grounded answer synthesis combining retrieval results and an LLM client."""

import time

from pydantic import BaseModel, Field

from documind.generation.llm import BaseLLMClient
from documind.generation.prompts import (
    REFUSAL_MESSAGE,
    SYSTEM_INSTRUCTION,
    build_user_prompt,
)
from documind.retrieval.retriever import RetrievalResult, RetrievedChunk, Retriever


class Citation(BaseModel):
    """Source reference attached to a generated answer."""

    source_number: int = Field(ge=1)
    filename: str
    page_number: int
    chunk_index: int
    score: float
    excerpt: str


class GeneratedAnswer(BaseModel):
    """Structured response from the RAG generation pipeline."""

    question: str
    answer: str
    citations: list[Citation]
    has_relevant_context: bool
    best_score: float | None = None
    latency_ms: float = Field(ge=0.0)


def _to_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            source_number=idx,
            filename=chunk.filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            score=chunk.score,
            excerpt=" ".join(chunk.text.split())[:160],
        )
        for idx, chunk in enumerate(chunks, start=1)
    ]


class AnswerGenerator:
    """Synthesizes a cited answer from retrieved chunks or refuses cleanly."""

    def __init__(self, llm: BaseLLMClient) -> None:
        self.llm = llm

    def generate(self, retrieval: RetrievalResult) -> GeneratedAnswer:
        """Produce a grounded answer from a RetrievalResult."""
        start = time.perf_counter()
        if not retrieval.has_relevant_context or not retrieval.chunks:
            elapsed = (time.perf_counter() - start) * 1000
            return GeneratedAnswer(
                question=retrieval.query,
                answer=REFUSAL_MESSAGE,
                citations=[],
                has_relevant_context=False,
                best_score=retrieval.best_score,
                latency_ms=round(elapsed, 2),
            )

        user_prompt = build_user_prompt(retrieval.query, retrieval.chunks)
        answer_text = self.llm.generate(SYSTEM_INSTRUCTION, user_prompt)
        elapsed = (time.perf_counter() - start) * 1000

        refused = REFUSAL_MESSAGE.lower() in answer_text.lower()
        return GeneratedAnswer(
            question=retrieval.query,
            answer=answer_text,
            citations=[] if refused else _to_citations(retrieval.chunks),
            has_relevant_context=not refused,
            best_score=retrieval.best_score,
            latency_ms=round(elapsed, 2),
        )


class RAGPipeline:
    """End-to-end retrieve-then-generate orchestrator."""

    def __init__(self, retriever: Retriever, generator: AnswerGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def ask(self, question: str, top_k: int | None = None) -> GeneratedAnswer:
        """Retrieve relevant chunks for a question and synthesize a cited answer."""
        retrieval = self.retriever.retrieve(question, top_k=top_k)
        return self.generator.generate(retrieval)
