"""Corpus Ingestion script and pipeline for GeoPol Bot.

Loads verified documents from permitted official sources, chunks them preserving
all metadata, generates normalized embeddings, and persists them into the vector store.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from backend.app.knowledge.schemas import DocumentSource, DocumentChunk
from backend.app.knowledge.chunker import DocumentChunker
from backend.app.knowledge.embeddings import EmbeddingService
from backend.app.knowledge.vector_store import SQLiteVectorStore

logger = logging.getLogger("geopol_bot.ingest")

CORPUS_JSON_PATH = Path(__file__).parent / "corpus" / "corpus.json"


def load_corpus_from_json(json_path: Optional[Path] = None) -> List[DocumentSource]:
    """Read verifiable documents from JSON file."""
    path = json_path or CORPUS_JSON_PATH
    if not path.exists():
        raise FileNotFoundError(f"Corpus JSON file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [DocumentSource(**item) for item in data]


def ingest_corpus(
    corpus_path: Optional[Path] = None,
    vector_store: Optional[SQLiteVectorStore] = None,
    embedding_service: Optional[EmbeddingService] = None,
    chunk_size: int = 600,
    chunk_overlap: int = 120,
) -> int:
    """Execute complete ingestion pipeline: load -> chunk -> embed -> index."""
    logger.info("Starting knowledge pipeline ingestion...")

    # Step 1: Load primary documents
    documents = load_corpus_from_json(corpus_path)
    logger.info("Loaded %d official documents from corpus.", len(documents))

    # Step 2: Chunk documents with metadata preservation
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[DocumentChunk] = chunker.chunk_corpus(documents)
    logger.info("Extracted %d chunks from documents.", len(chunks))

    # Step 3: Generate embeddings
    embedder = embedding_service or EmbeddingService()
    texts_to_embed = [chunk.content for chunk in chunks]
    logger.info("Generating embeddings for %d passages...", len(texts_to_embed))
    vectors = embedder.embed_batch(texts_to_embed)

    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector

    # Step 4: Index into vector store
    store = vector_store or SQLiteVectorStore()
    inserted = store.upsert_chunks(chunks)
    logger.info("Ingestion complete. %d chunks active in vector index.", inserted)

    return inserted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = ingest_corpus()
    print(f"Successfully indexed {count} evidence chunks into vector store.")
