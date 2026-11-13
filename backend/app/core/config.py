"""Configuration management for GeoPol Bot using pydantic-settings.

Reads environment variables from .env or process environment.
Ensures secrets such as OPENAI_API_KEY are strictly contained on the backend.
"""

from functools import lru_cache
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Project metadata
    project_name: str = Field(default="GeoPol Bot", description="Name of the application")
    project_version: str = Field(default="0.1.0", description="Semantic application version")
    environment: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=True, description="Enable debug mode")

    # Server configuration
    backend_host: str = Field(default="127.0.0.1", description="FastAPI host binding")
    backend_port: int = Field(default=8000, description="FastAPI port binding")
    api_v1_prefix: str = Field(default="/api/v1", description="API route prefix for v1")

    # CORS configuration
    cors_origins: Union[List[str], str] = Field(
        default=["http://localhost:8501", "http://127.0.0.1:8501"],
        description="Allowed CORS origins",
    )

    # OpenAI configuration (Backend only - never exposed to frontend)
    openai_api_key: Union[str, None] = Field(
        default=None,
        description="OpenAI API Key for Domain Gating and Generation. Strictly backend-only.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Default OpenAI model for classification and generation",
    )
    openai_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout for OpenAI API requests in seconds",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse comma-separated string of origins if provided via environment variable."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def is_openai_configured(self) -> bool:
        """Check if an OpenAI API key is present."""
        return bool(self.openai_api_key and self.openai_api_key.strip() and not self.openai_api_key.startswith("sk-your-openai"))

    @property
    def masked_openai_api_key(self) -> str:
        """Return a safely masked representation of the API key for health/diagnostics."""
        if not self.is_openai_configured:
            return "NOT_CONFIGURED"
        key = self.openai_api_key.strip()  # type: ignore[union-attr]
        if len(key) <= 8:
            return "****"
        return f"****...{key[-4:]}"


@lru_cache()
def get_settings() -> Settings:
    """Cached factory function returning application settings."""
    return Settings()
