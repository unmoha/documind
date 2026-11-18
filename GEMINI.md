You are helping me build "DocuMind", a production-quality RAG system for my AI engineer portfolio.

Stack (do not change without asking): Python 3.11, uv, FastAPI, Pydantic v2, ChromaDB (persistent), sentence-transformers (BAAI/bge-small-en-v1.5), Google Gemini API (free tier) behind an LLM provider interface, Streamlit UI, SQLite for logs, pytest, Ruff, Docker.

Rules:
- Only free tools and libraries. Never hardcode secrets; use pydantic-settings and a .env file with a committed .env.example.
- Use type hints, docstrings, and small single-purpose modules.
- Structure: src/documind/{config,ingestion,retrieval,generation,llm,api,logging}, ui/, evals/, tests/.
- Define the LLM client behind an abstract interface so providers (Gemini, Groq) can be swapped.
- Handle errors explicitly: timeouts, rate limits (retry with backoff), empty inputs.
- Work on ONE task at a time. Before writing code, show a short plan. After writing code, list the files changed and how to run and test them.
- Never invent library APIs. If unsure, say so and check the docs.
- Keep answers grounded: the system must only answer from retrieved context and cite sources.
