# syntax=docker/dockerfile:1

# ==============================================================================
# Stage 1: Build virtual environment with uv
# ==============================================================================
FROM python:3.11-slim-bookworm AS builder

# Install uv binary from official distribution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Compile bytecode for faster container startup, use copy mode inside container
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy dependency manifests first to leverage Docker layer caching
COPY pyproject.toml uv.lock README.md ./

# Install project dependencies without installing the application package itself
RUN uv sync --frozen --no-install-project --no-dev

# Copy application and UI source code
COPY src ./src
COPY ui ./ui

# Install the project into the virtual environment
RUN uv sync --frozen --no-dev

# ==============================================================================
# Stage 2: Production runtime image (non-root, minimal surface, no baked secrets)
# ==============================================================================
FROM python:3.11-slim-bookworm AS runtime

# Install curl for healthcheck validation; purge apt lists to keep image slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a dedicated non-root group and user
RUN groupadd -g 10001 documind && \
    useradd -u 10001 -g documind -m -d /home/documind -s /bin/bash documind

WORKDIR /app

# Create persistent storage directories with proper non-root user permissions
RUN mkdir -p /app/data/chroma /app/data/logs /home/documind/.cache && \
    chown -R documind:documind /app /home/documind

# Copy the pre-built virtual environment and application code from the builder stage
COPY --from=builder --chown=documind:documind /app/.venv /app/.venv
COPY --from=builder --chown=documind:documind /app/src /app/src
COPY --from=builder --chown=documind:documind /app/ui /app/ui
COPY --from=builder --chown=documind:documind /app/pyproject.toml /app/README.md ./

# Configure environment: active venv in PATH, unbuffered stdout, default volume paths
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DOCUMIND_CHROMA_PERSIST_DIR=/app/data/chroma \
    DOCUMIND_SQLITE_DB_PATH=/app/data/logs/logs.sqlite \
    HF_HOME=/home/documind/.cache/huggingface

# Run as non-root user
USER documind

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command starts FastAPI backend server
CMD ["uvicorn", "documind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
