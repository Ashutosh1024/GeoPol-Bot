"""Unit tests for configuration management and security safeguards."""

import pytest
from backend.app.core.config import Settings


def test_default_settings():
    """Verify default settings load cleanly."""
    settings = Settings(
        project_name="GeoPol Bot",
        openai_api_key="sk-test-mock-key-1234567890abcdef",
    )
    assert settings.project_name == "GeoPol Bot"
    assert settings.backend_port == 8000
    assert settings.api_v1_prefix == "/api/v1"
    assert settings.is_openai_configured is True


def test_cors_origins_parsing():
    """Verify comma-separated string origins are parsed into a list."""
    settings = Settings(
        cors_origins="http://example.com, https://geopol.ai",
        openai_api_key=None,
    )
    assert isinstance(settings.cors_origins, list)
    assert "http://example.com" in settings.cors_origins
    assert "https://geopol.ai" in settings.cors_origins


def test_masked_api_key_security():
    """Verify raw API keys are NEVER returned in cleartext by masked property."""
    # When empty or unset
    settings_empty = Settings(openai_api_key=None)
    assert settings_empty.masked_openai_api_key == "NOT_CONFIGURED"
    assert settings_empty.is_openai_configured is False

    # When placeholder
    settings_placeholder = Settings(openai_api_key="sk-your-openai-api-key-here")
    assert settings_placeholder.is_openai_configured is False

    # When real or mock key provided
    raw_key = "sk-proj-1234567890abcdefghijklmnopqrstuvwxyz"
    settings_with_key = Settings(openai_api_key=raw_key)
    masked = settings_with_key.masked_openai_api_key
    assert masked != raw_key
    assert masked == f"****...{raw_key[-4:]}"
    assert raw_key[3:-4] not in masked
