"""Knowledge Retriever service for GeoPol Bot.

Orchestrates semantic search, vector similarity ranking, and source deduplication
against verified primary geopolitical evidence.
"""

import time
import logging
from typing import Optional, List

from backend.app.knowledge.schemas import (
    RetrievalQuery,
    RetrievalResponse,
    RetrievedEvidence,
)
from backend.app.knowledge.embeddings import EmbeddingService
from backend.app.knowledge.vector_store import SQLiteVectorStore

logger = logging.getLogger("geopol_bot.retriever")


class KnowledgeRetriever:
    """Service retrieving relevant, verified geopolitical evidence for incoming queries."""

    def __init__(
        self,
        vector_store: Optional[SQLiteVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.vector_store = vector_store or SQLiteVectorStore()
        self.embedding_service = embedding_service or EmbeddingService()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.30,
        filter_topic: Optional[str] = None,
    ) -> RetrievalResponse:
        """Embed user query and retrieve top-k ranked evidence chunks from verifiable corpus."""
        start_time = time.perf_counter()

        # Step 1: Generate query vector
        query_vector = self.embedding_service.embed_text(query)

        # Step 2: Search vector store
        ranked_evidence: List[RetrievedEvidence] = self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter_topic=filter_topic,
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return RetrievalResponse(
            query=query,
            results=ranked_evidence,
            count=len(ranked_evidence),
            latency_ms=latency_ms,
        )

    def retrieve_by_query_obj(self, query_obj: RetrievalQuery) -> RetrievalResponse:
        """Convenience method accepting a RetrievalQuery schema object."""
        return self.retrieve(
            query=query_obj.query,
            top_k=query_obj.top_k,
            similarity_threshold=query_obj.similarity_threshold,
            filter_topic=query_obj.filter_topic,
        )
