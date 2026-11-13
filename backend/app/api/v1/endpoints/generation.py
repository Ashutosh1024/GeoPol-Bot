"""FastAPI endpoints for Answer Generation."""

from fastapi import APIRouter, Depends, status
from backend.app.core.config import Settings, get_settings
from backend.app.generation.schemas import GenerationRequest, GenerationResponse
from backend.app.generation.engine import GeoPolGenerationEngine

router = APIRouter()


def get_generation_engine(settings: Settings = Depends(get_settings)) -> GeoPolGenerationEngine:
    """Dependency provider returning an initialized GeoPolGenerationEngine."""
    return GeoPolGenerationEngine(settings=settings)


@router.post(
    "/chat/generate",
    response_model=GenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Grounded Geopolitical Answer",
    description=(
        "Executes full pipeline: Domain Gate validation -> Semantic RAG retrieval -> "
        "OpenAI Responses API structured generation (Facts vs Interpretations vs Uncertainties) -> "
        "Post-generation citation validation and abstention enforcement."
    ),
)
async def generate_chat_response(
    request: GenerationRequest,
    engine: GeoPolGenerationEngine = Depends(get_generation_engine),
) -> GenerationResponse:
    """Execute end-to-end evidence-grounded answer generation."""
    return engine.generate_answer(request)
