"""Pydantic schemas for GeoPol Bot evidence-grounded answer generation."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from backend.app.domain_gate.schemas import DomainGateResponse
from backend.app.knowledge.schemas import RetrievedEvidence


class AnswerStatus(str, Enum):
    """Categorical outcome of the end-to-end question answering pipeline."""
    ANSWERED = "ANSWERED"
    INSUFFICIENT_EVIDENCE_ABSTENTION = "INSUFFICIENT_EVIDENCE_ABSTENTION"
    REJECTED_OUT_OF_DOMAIN = "REJECTED_OUT_OF_DOMAIN"
    REQUIRES_CLARIFICATION = "REQUIRES_CLARIFICATION"
    API_ERROR = "API_ERROR"


class Citation(BaseModel):
    """Verified citation pointing directly to a retrieved primary source chunk."""
    chunk_id: str = Field(..., description="ID of the retrieved evidence chunk")
    document_title: str = Field(..., description="Official title of the source document")
    source_name: str = Field(..., description="Issuing government or multilateral organization")
    source_url: str = Field(..., description="Direct verifiable URL to official text")
    publication_date: str = Field(..., description="Publication date of the evidence")
    relevant_quote: Optional[str] = Field(None, description="Exact supporting excerpt from the text")


class StructuredAnswerPayload(BaseModel):
    """Structured answer format separating verified facts, strategic interpretations, and uncertainties."""
    is_sufficient_evidence: bool = Field(
        ...,
        description="False if retrieved evidence does not contain sufficient facts to answer the query.",
    )
    abstention_statement: Optional[str] = Field(
        None,
        description="Clear abstention message if evidence is insufficient.",
    )
    verified_facts: List[str] = Field(
        default_factory=list,
        description="Factual statements derived strictly and directly from the retrieved primary evidence.",
    )
    strategic_interpretations: List[str] = Field(
        default_factory=list,
        description="Analytical perspectives, geopolitical implications, and strategic consequences. Provide at least one point when evidence is sufficient.",
    )
    uncertainties_and_limitations: List[str] = Field(
        default_factory=list,
        description="Unresolved questions, missing information, or legal/political ambiguities not answered by the evidence. Provide at least one point when evidence is sufficient.",
    )
    cited_chunk_ids: List[str] = Field(
        default_factory=list,
        description="List of chunk IDs from the retrieved evidence that support the verified facts.",
    )


class GenerationRequest(BaseModel):
    """Request payload for geopolitical question answering."""
    query: str = Field(..., min_length=2, max_length=4000, description="The user's geopolitical inquiry")
    conversation_context: Optional[List[str]] = Field(
        default=None,
        max_length=5,
        description="Recent conversation turns for context",
    )
    similarity_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score required for retrieved evidence",
    )
    top_k: int = Field(default=4, ge=1, le=8, description="Maximum evidence passages to retrieve")
    allow_foundational_fallback: bool = Field(
        default=True,
        description="When True, answers in-domain queries using broad geopolitical knowledge if local corpus lacks direct chunks.",
    )


class GenerationResponse(BaseModel):
    """Complete response payload detailing gate classification, evidence, and structured answer."""
    query: str = Field(..., description="Original user query")
    status: AnswerStatus = Field(..., description="Execution status of the pipeline")
    gate_result: DomainGateResponse = Field(..., description="Domain gate classification metadata")
    retrieved_evidence: List[RetrievedEvidence] = Field(default_factory=list, description="Evidence chunks retrieved")
    answer: Optional[StructuredAnswerPayload] = Field(None, description="Structured answer model if generated")
    markdown_output: str = Field(..., description="Formatted readable Markdown response for display")
    validated_citations: List[Citation] = Field(default_factory=list, description="Citations verified against retrieved documents")
    unverified_citations_count: int = Field(default=0, description="Count of invalid or fabricated citations stripped")
    grounding_source: str = Field(
        default="primary_corpus",
        description="Source of grounding: 'primary_corpus' or 'foundational_geopolitical_knowledge'",
    )
    latency_ms: float = Field(..., description="End-to-end processing latency in milliseconds")
    model_used: str = Field(..., description="OpenAI model utilized")
