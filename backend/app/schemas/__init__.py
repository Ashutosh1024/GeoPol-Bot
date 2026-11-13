"""Pydantic schemas package for GeoPol Bot."""
from backend.app.schemas.health import HealthCheckResponse, ComponentStatus

__all__ = ["HealthCheckResponse", "ComponentStatus"]
