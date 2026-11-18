"""SQLite interaction logging and query/response auditing."""

import sqlite3
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from documind.config.settings import get_settings


class InteractionLog(BaseModel):
    """Structured record of a single RAG query, retrieval, and generation cycle."""

    id: int | None = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    query: str
    best_score: float | None = None
    has_relevant_context: bool
    answer: str
    citations_json: str
    latency_ms: float


class SQLiteLogger:
    """Manages SQLite database creation and interaction audit logging."""

    def __init__(self, db_path: Path | None = None) -> None:
        settings = get_settings()
        self.db_path = db_path if db_path is not None else settings.sqlite_db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Create the interaction logs table if it does not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interaction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    best_score REAL,
                    has_relevant_context BOOLEAN NOT NULL,
                    answer TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    latency_ms REAL NOT NULL
                )
                """
            )
            conn.commit()

    def log_interaction(
        self,
        query: str,
        best_score: float | None,
        has_relevant_context: bool,
        answer: str,
        citations_json: str,
        latency_ms: float,
    ) -> int:
        """Insert a new interaction log entry and return its row ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO interaction_logs (
                    timestamp, query, best_score, has_relevant_context,
                    answer, citations_json, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    query,
                    best_score,
                    int(has_relevant_context),
                    answer,
                    citations_json,
                    latency_ms,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_recent_logs(self, limit: int = 10) -> list[InteractionLog]:
        """Retrieve recent interaction logs for auditing."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, timestamp, query, best_score, has_relevant_context,
                       answer, citations_json, latency_ms
                FROM interaction_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                InteractionLog(
                    id=row[0],
                    timestamp=row[1],
                    query=row[2],
                    best_score=row[3],
                    has_relevant_context=bool(row[4]),
                    answer=row[5],
                    citations_json=row[6],
                    latency_ms=row[7],
                )
                for row in rows
            ]
