"""FastAPI endpoint for Domain Gate Classification."""

from fastapi import APIRouter, Depends, status
from backend.app.core.config import Settings, get_settings
from backend.app.domain_gate.schemas import DomainGateRequest, DomainGateResponse
from backend.app.domain_gate.classifier import DomainGateClassifier

router = APIRouter()


def get_gate_classifier(settings: Settings = Depends(get_settings)) -> DomainGateClassifier:
    """Dependency provider returning an initialized DomainGateClassifier."""
    return DomainGateClassifier(settings=settings)


@router.post(
    "/gate/classify",
    response_model=DomainGateResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Query Domain",
    description=(
        "Evaluates a user query against the GeoPol Bot geopolitics scope policy. "
        "Returns IN_DOMAIN, OUT_OF_DOMAIN, or UNCERTAIN. "
        "Enforces that only IN_DOMAIN queries are allowed to proceed to downstream generation."
    ),
)
async def classify_query(
    request: DomainGateRequest,
    classifier: DomainGateClassifier = Depends(get_gate_classifier),
) -> DomainGateResponse:
    """Classify the input query into geopolitics domain categories."""
    return classifier.classify(
        query=request.query,
        conversation_context=request.conversation_context,
    )
