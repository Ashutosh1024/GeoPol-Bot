"""FastAPI endpoints for Knowledge Retrieval and Ingestion."""

from fastapi import APIRouter, Depends, status, Query
from backend.app.core.config import Settings, get_settings
from backend.app.knowledge.schemas import (
    RetrievalQuery,
    RetrievalResponse,
)
from backend.app.knowledge.retriever import KnowledgeRetriever
from backend.app.knowledge.vector_store import SQLiteVectorStore
from backend.app.knowledge.ingest import ingest_corpus

router = APIRouter()


def get_retriever(settings: Settings = Depends(get_settings)) -> KnowledgeRetriever:
    """Dependency provider returning an initialized KnowledgeRetriever."""
    return KnowledgeRetriever()


@router.post(
    "/knowledge/retrieve",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Geopolitical Evidence",
    description="Semantically searches verified primary source geopolitical corpus and returns ranked evidence.",
)
async def retrieve_evidence(
    query_payload: RetrievalQuery,
    retriever: KnowledgeRetriever = Depends(get_retriever),
) -> RetrievalResponse:
    """Execute semantic vector retrieval against verified corpus."""
    return retriever.retrieve_by_query_obj(query_payload)


@router.get(
    "/knowledge/sources",
    status_code=status.HTTP_200_OK,
    summary="List Verified Corpus Sources",
    description="Returns all official treaties, conventions, and agreements currently indexed.",
)
async def list_sources(
    vector_store: SQLiteVectorStore = Depends(lambda: SQLiteVectorStore()),
) -> dict:
    """Return list of distinct source documents in the vector store."""
    sources = vector_store.get_all_sources()
    return {
        "count": len(sources),
        "total_chunks": vector_store.count(),
        "sources": sources,
    }


@router.post(
    "/knowledge/ingest",
    status_code=status.HTTP_200_OK,
    summary="Trigger Corpus Ingestion",
    description="Chunks and generates vector embeddings for the official verifiable corpus.",
)
async def trigger_ingestion() -> dict:
    """Run ingestion pipeline on the verified corpus."""
    count = ingest_corpus()
    return {
        "status": "success",
        "message": f"Successfully indexed {count} evidence chunks into vector store.",
        "indexed_chunks": count,
    }
