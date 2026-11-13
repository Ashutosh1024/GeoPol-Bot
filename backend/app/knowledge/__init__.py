"""Knowledge Pipeline package for GeoPol Bot."""

from backend.app.knowledge.schemas import (
    DocumentSource,
    DocumentChunk,
    RetrievalQuery,
    RetrievedEvidence,
    RetrievalResponse,
)
from backend.app.knowledge.chunker import DocumentChunker
from backend.app.knowledge.embeddings import EmbeddingService
from backend.app.knowledge.vector_store import SQLiteVectorStore
from backend.app.knowledge.ingest import ingest_corpus, load_corpus_from_json
from backend.app.knowledge.retriever import KnowledgeRetriever

__all__ = [
    "DocumentSource",
    "DocumentChunk",
    "RetrievalQuery",
    "RetrievedEvidence",
    "RetrievalResponse",
    "DocumentChunker",
    "EmbeddingService",
    "SQLiteVectorStore",
    "ingest_corpus",
    "load_corpus_from_json",
    "KnowledgeRetriever",
]
