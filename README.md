# DocuMind

![CI](https://github.com/unmoha/documind/actions/workflows/ci.yml/badge.svg)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DocuMind is an open-source, production-grade Retrieval-Augmented Generation (RAG) system built for AI engineer portfolio demonstrations. It provides strict answer groundedness, transparent source citations, provider-swappable LLM execution, persistent local vector storage, and SQLite-backed audit trails.

---

## Architecture Overview

```
[ Ingest ] ──> [ Chunk ] ──> [ Embed ] ──> [ Store ] ──> [ Retrieve ] ──> [ Generate & Cite ]
 Documents      Recursive     BGE-Small     ChromaDB       Top-k Cosine       Gemini / Groq
 (PDF/MD/TXT)   Splitting      (Local)     (Persistent)     Similarity        Strict Grounding
```

For complete architectural specifications, see [docs/SPEC.md](docs/SPEC.md).

---

## Tech Stack

- **Runtime & Tooling**: Python 3.11, [uv](https://github.com/astral-sh/uv), Ruff, pytest
- **Backend API**: FastAPI, Pydantic v2, pydantic-settings
- **Vector Search & Embeddings**: ChromaDB (persistent), `sentence-transformers` (`BAAI/bge-small-en-v1.5`)
- **LLM Engine**: Google Gemini API (free tier) via abstract provider interface
- **Frontend UI**: Streamlit
- **Observability**: SQLite audit database
- **Deployment**: Docker containerization

---

## Repository Structure

```
documind/
├── .env.example              # Environment variables template
├── docs/
│   └── SPEC.md               # Technical specification and system architecture
├── src/
│   └── documind/
│       ├── config/           # Pydantic v2 settings and environment loading
│       ├── ingestion/        # Document parsing, cleaning, and chunking
│       ├── retrieval/        # Local BGE embeddings and ChromaDB store
│       ├── generation/       # Grounded prompt construction and citation linking
│       ├── llm/              # Abstract LLM client interface (Gemini, Groq)
│       ├── api/              # FastAPI routers and schemas
│       └── logging/          # SQLite interaction logging and auditing
├── ui/                       # Streamlit interactive web application
├── evals/                    # Retrieval hit-rate and faithfulness benchmarks
├── tests/                    # Unit and integration test suite
└── pyproject.toml            # Project dependencies and tool configurations
```

---

## Getting Started

### Prerequisites
- Python 3.11
- [uv](https://docs.astral.sh/uv/)

### Installation
```bash
# Clone the repository
git clone https://github.com/anwarmohammed/documind.git
cd documind

# Create virtual environment and sync dependencies with uv
uv venv --python 3.11
uv pip install -e ".[dev]"

# Copy environment variables template
cp .env.example .env
# Edit .env and supply your DOCUMIND_GEMINI_API_KEY
```

### Running Tests and Linting
```bash
# Run test suite
uv run pytest

# Check formatting and linting
uv run ruff check .
```

### Starting the Services (Local uv)
```bash
# Start FastAPI backend
uv run uvicorn documind.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit UI
uv run streamlit run ui/app.py
```

### Running with Docker & Docker Compose
The system includes a secure, non-root multi-stage Dockerfile and a full-stack Compose configuration.

```bash
# 1. Ensure DOCUMIND_GEMINI_API_KEY is set in your .env file
# DOCUMIND_GEMINI_API_KEY="your-gemini-api-key"

# 2. Build the multi-stage image and start both API and UI services
docker compose up --build

# 3. Access the services:
# - FastAPI Swagger Docs: http://localhost:8000/docs
# - FastAPI Health:       http://localhost:8000/health
# - Streamlit Web UI:     http://localhost:8501

# 4. Stop the services (data in ChromaDB and SQLite remains persisted in Docker volumes):
docker compose down

# To reset database volumes:
# docker compose down -v
```

---

## License
MIT License.
