"""Health check response models."""

from typing import Dict, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ComponentStatus(BaseModel):
    """Status details for an individual component."""
    status: str = Field(..., description="Component state: ready, pending, not_configured, degraded, etc.")
    details: Optional[str] = Field(None, description="Human-readable status summary or diagnostic note")


class HealthCheckResponse(BaseModel):
    """Standardized health check response schema for GeoPol Bot."""
    status: str = Field(..., description="Overall service status: healthy, degraded, or unhealthy")
    app_name: str = Field(..., description="Name of the service")
    version: str = Field(..., description="Current running application version")
    environment: str = Field(..., description="Deployment environment: development, staging, production")
    uptime_seconds: float = Field(..., description="Uptime in seconds since FastAPI startup")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the health check",
    )
    components: Dict[str, ComponentStatus] = Field(
        default_factory=dict,
        description="Component-level health check statuses",
    )
