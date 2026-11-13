"""Pydantic schemas for GeoPol Bot Knowledge Retrieval & Ingestion Pipeline."""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentSource(BaseModel):
    """Represents a verified primary source document."""
    id: str = Field(..., description="Unique slug identifier for the document")
    title: str = Field(..., description="Official document title")
    source_name: str = Field(..., description="Official issuing authority or publication body")
    source_url: str = Field(..., description="Direct verifiable URL to the official text")
    publication_date: str = Field(..., description="ISO publication or ratification date (YYYY-MM-DD)")
    topic: str = Field(..., description="Geopolitical thematic topic")
    summary: str = Field(..., description="Concise editorial summary of the document")
    content: str = Field(..., description="Full verified text content")


class DocumentChunk(BaseModel):
    """Represents an extracted text chunk with complete preserved provenance metadata."""
    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. doc-id#chunk-0")
    document_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., description="Zero-indexed chunk sequence within parent document")
    title: str = Field(..., description="Parent document title")
    source_name: str = Field(..., description="Official issuing entity")
    source_url: str = Field(..., description="Direct verifiable link")
    publication_date: str = Field(..., description="Publication date of the evidence")
    topic: str = Field(..., description="Topical category")
    content: str = Field(..., description="Extracted textual passage")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding values")


class RetrievalQuery(BaseModel):
    """Request payload for semantic knowledge retrieval."""
    query: str = Field(..., min_length=2, max_length=1000, description="The query string to match against evidence")
    top_k: int = Field(default=3, ge=1, le=10, description="Maximum number of evidence chunks to retrieve")
    similarity_threshold: float = Field(default=0.30, ge=0.0, le=1.0, description="Minimum cosine similarity score required")
    filter_topic: Optional[str] = Field(default=None, description="Optional topic filter")


class RetrievedEvidence(BaseModel):
    """Represents a retrieved evidence item with relevance scoring and provenance."""
    chunk_id: str = Field(..., description="Chunk ID")
    document_title: str = Field(..., description="Official source title")
    source_name: str = Field(..., description="Issuing authority")
    source_url: str = Field(..., description="Clickable verified URL")
    publication_date: str = Field(..., description="Publication date")
    topic: str = Field(..., description="Topic")
    content: str = Field(..., description="Exact textual evidence excerpt")
    score: float = Field(..., description="Cosine similarity score (0.0 - 1.0)")
    rank: int = Field(..., description="Relevance rank (1 = highest)")


class RetrievalResponse(BaseModel):
    """Response payload for knowledge retrieval."""
    query: str = Field(..., description="Query executed")
    results: List[RetrievedEvidence] = Field(default_factory=list, description="Ranked evidence chunks")
    count: int = Field(..., description="Number of retrieved evidence chunks meeting threshold")
    latency_ms: float = Field(..., description="Retrieval latency in milliseconds")
