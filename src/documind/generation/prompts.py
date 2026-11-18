"""Prompt templates for strictly grounded RAG answer synthesis."""

from documind.retrieval.retriever import RetrievedChunk

SYSTEM_INSTRUCTION = """You are DocuMind, a precise document question-answering assistant.
Answer the user's question using ONLY the numbered context passages provided below.

Rules:
1. Every factual claim in your answer MUST be supported by the context and cited inline using its source number, e.g. [1] or [1][2].
2. Do NOT use outside knowledge, assumptions, or extrapolation.
3. If the provided passages do not contain enough information to answer the question, respond with:
   "I cannot answer this question based on the provided documents."
4. Keep the answer clear, direct, and well-structured."""

REFUSAL_MESSAGE = "I cannot answer this question based on the provided documents."


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as numbered source blocks for the prompt."""
    blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        header = (
            f"[{idx}] Source: {chunk.filename} "
            f"(page {chunk.page_number}, chunk #{chunk.chunk_index}, "
            f"score {chunk.score:.3f})"
        )
        blocks.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Combine formatted context passages with the user's question."""
    context_block = format_context(chunks)
    return (
        f"Context Passages:\n\n{context_block}\n\n"
        f"Question: {question.strip()}\n\n"
        "Grounded Answer:"
    )
