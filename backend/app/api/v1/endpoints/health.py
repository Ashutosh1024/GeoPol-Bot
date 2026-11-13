"""Health check endpoints for GeoPol Bot."""

import time
from fastapi import APIRouter, Depends, status
from backend.app.core.config import Settings, get_settings
from backend.app.schemas.health import ComponentStatus, HealthCheckResponse

router = APIRouter()

# Track module load time as an initial fallback for uptime calculation
START_TIME = time.time()


def get_uptime_seconds() -> float:
    """Calculate the uptime of the application in seconds."""
    return round(time.time() - START_TIME, 2)


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Returns current service status, uptime, environment, and component readiness.",
)
async def check_health(settings: Settings = Depends(get_settings)) -> HealthCheckResponse:
    """Evaluate and report system health and component statuses."""
    components = {
        "openai_api": ComponentStatus(
            status="ready" if settings.is_openai_configured else "not_configured",
            details=f"API key: {settings.masked_openai_api_key} (Model: {settings.openai_model})",
        ),
        "domain_gate": ComponentStatus(
            status="ready" if settings.is_openai_configured else "standby",
            details=f"Tri-state Structured Classifier active (Model: {settings.openai_model})"
            if settings.is_openai_configured
            else "Awaiting OpenAI API key configuration",
        ),
        "knowledge_pipeline": ComponentStatus(
            status="ready",
            details="Verifiable primary corpus, chunking, and SQLite vector store active",
        ),
        "generation_engine": ComponentStatus(
            status="ready" if settings.is_openai_configured else "standby",
            details=f"Evidence-grounded generator active (Model: {settings.openai_model})"
            if settings.is_openai_configured
            else "Awaiting OpenAI API key configuration",
        ),
    }

    # If in production and OpenAI key is missing, mark degraded, else healthy in dev
    overall_status = "healthy"
    if settings.environment == "production" and not settings.is_openai_configured:
        overall_status = "degraded"

    return HealthCheckResponse(
        status=overall_status,
        app_name=settings.project_name,
        version=settings.project_version,
        environment=settings.environment,
        uptime_seconds=get_uptime_seconds(),
        components=components,
    )


@router.get(
    "/healthz",
    status_code=status.HTTP_200_OK,
    summary="Lightweight Liveness Probe",
    description="Kubernetes/Docker liveness probe returning simple ok status.",
)
async def liveness_probe() -> dict:
    """Return lightweight liveness status."""
    return {"status": "ok", "app": "GeoPol Bot"}
