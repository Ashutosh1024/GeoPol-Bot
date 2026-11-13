"""Generation module for GeoPol Bot."""

from backend.app.generation.schemas import (
    AnswerStatus,
    Citation,
    StructuredAnswerPayload,
    GenerationRequest,
    GenerationResponse,
)
from backend.app.generation.engine import GeoPolGenerationEngine
from backend.app.generation.prompts import (
    ANSWER_GENERATION_SYSTEM_PROMPT,
    format_evidence_context,
)

__all__ = [
    "AnswerStatus",
    "Citation",
    "StructuredAnswerPayload",
    "GenerationRequest",
    "GenerationResponse",
    "GeoPolGenerationEngine",
    "ANSWER_GENERATION_SYSTEM_PROMPT",
    "format_evidence_context",
]
