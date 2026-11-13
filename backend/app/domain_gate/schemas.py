"""Pydantic schemas for the GeoPol Bot Domain Gate."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class DomainLabel(str, Enum):
    """Tri-state domain classification label."""
    IN_DOMAIN = "IN_DOMAIN"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    UNCERTAIN = "UNCERTAIN"


class DomainStructuredClassification(BaseModel):
    """Pydantic model passed to OpenAI Structured Outputs for guaranteed schema conformity."""
    label: DomainLabel = Field(
        ...,
        description="Classification label: IN_DOMAIN, OUT_OF_DOMAIN, or UNCERTAIN.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 in this classification decision.",
    )
    primary_category: str = Field(
        ...,
        description="The primary topical category identifying the subject matter.",
    )
    reasoning: str = Field(
        ...,
        max_length=500,
        description="Concise factual explanation justifying why this query received the assigned label.",
    )
    clarification_prompt: Optional[str] = Field(
        None,
        description="If label is UNCERTAIN, provide a polite targeted clarification question for the user.",
    )


class DomainGateRequest(BaseModel):
    """API request payload for classifying a query."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's input query or question to be evaluated.",
    )
    conversation_context: Optional[List[str]] = Field(
        default=None,
        max_length=5,
        description="Optional preceding conversation turns for contextual disambiguation.",
    )


class DomainGateResponse(BaseModel):
    """Full API response payload returning domain gate classification results."""
    query: str = Field(..., description="The query that was evaluated")
    label: DomainLabel = Field(..., description="Final classification label")
    confidence: float = Field(..., description="Classification confidence score")
    primary_category: str = Field(..., description="Identified category")
    reasoning: str = Field(..., description="Explanation of decision")
    allowed_to_proceed: bool = Field(
        ...,
        description="True ONLY if IN_DOMAIN. OUT_OF_DOMAIN and UNCERTAIN must NEVER proceed to answer generation.",
    )
    clarification_prompt: Optional[str] = Field(
        None,
        description="Suggested clarification prompt if UNCERTAIN.",
    )
    processing_time_ms: float = Field(
        ...,
        description="Classification execution time in milliseconds.",
    )
    model_used: str = Field(
        ...,
        description="The OpenAI model or fallback rule engine used for classification.",
    )
