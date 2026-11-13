"""Vector store implementation for GeoPol Bot.

Provides a persistent SQLite-backed vector storage and semantic retrieval engine
with exact cosine similarity ranking and metadata filtering.
Designed to be lightweight, zero-dependency, and fully compatible with pgvector schemas.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from backend.app.knowledge.schemas import DocumentChunk, RetrievedEvidence

logger = logging.getLogger("geopol_bot.vector_store")

DEFAULT_DB_PATH = Path("data/geopol_vectors.db")


class SQLiteVectorStore:
    """Persistent SQLite-backed vector store with cosine similarity ranking."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema for document chunks and vector embeddings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    publication_date TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_topic ON document_chunks(topic);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(document_id);"
            )
            conn.commit()

    def count(self) -> int:
        """Return total number of indexed chunks."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM document_chunks;")
            return cursor.fetchone()[0]

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Upsert a list of chunks with embeddings into the vector store."""
        if not chunks:
            return 0

        inserted_count = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for chunk in chunks:
                if chunk.embedding is None:
                    continue

                embedding_json = json.dumps(chunk.embedding)
                cursor.execute(
                    """
                    INSERT INTO document_chunks (
                        chunk_id, document_id, chunk_index, title, source_name,
                        source_url, publication_date, topic, content, embedding_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        content=excluded.content,
                        title=excluded.title,
                        source_name=excluded.source_name,
                        source_url=excluded.source_url,
                        publication_date=excluded.publication_date,
                        topic=excluded.topic,
                        embedding_json=excluded.embedding_json;
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.title,
                        chunk.source_name,
                        chunk.source_url,
                        chunk.publication_date,
                        chunk.topic,
                        chunk.content,
                        embedding_json,
                    ),
                )
                inserted_count += 1
            conn.commit()

        logger.info("Upserted %d chunks into vector store at %s", inserted_count, self.db_path)
        return inserted_count

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        similarity_threshold: float = 0.30,
        filter_topic: Optional[str] = None,
    ) -> List[RetrievedEvidence]:
        """Perform cosine similarity vector search across indexed chunks.

        Returns ranked RetrievedEvidence meeting or exceeding the similarity threshold.
        """
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = q_vec / q_norm

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if filter_topic:
                cursor.execute(
                    "SELECT chunk_id, title, source_name, source_url, publication_date, topic, content, embedding_json FROM document_chunks WHERE topic = ?;",
                    (filter_topic,),
                )
            else:
                cursor.execute(
                    "SELECT chunk_id, title, source_name, source_url, publication_date, topic, content, embedding_json FROM document_chunks;"
                )

            rows = cursor.fetchall()
            if not rows:
                return []

            scored_candidates: List[Tuple[float, sqlite3.Row]] = []
            for row in rows:
                doc_vec = np.array(json.loads(row["embedding_json"]), dtype=np.float32)
                doc_norm = np.linalg.norm(doc_vec)
                if doc_norm == 0:
                    continue
                score = float(np.dot(q_vec, doc_vec / doc_norm))

                if score >= similarity_threshold:
                    scored_candidates.append((score, row))

            # Rank by descending similarity score
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            results: List[RetrievedEvidence] = []
            for rank, (score, row) in enumerate(scored_candidates[:top_k], start=1):
                results.append(
                    RetrievedEvidence(
                        chunk_id=row["chunk_id"],
                        document_title=row["title"],
                        source_name=row["source_name"],
                        source_url=row["source_url"],
                        publication_date=row["publication_date"],
                        topic=row["topic"],
                        content=row["content"],
                        score=round(score, 4),
                        rank=rank,
                    )
                )

            return results

    def get_all_sources(self) -> List[dict]:
        """Return distinct source documents currently indexed in the store."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT document_id, title, source_name, source_url, publication_date, topic
                FROM document_chunks
                ORDER BY publication_date DESC;
                """
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
