"""Unit tests for the GeoPol Bot health endpoints."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint responds with service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "GeoPol Bot"
    assert data["status"] == "online"
    assert "health_url" in data


def test_healthz_liveness():
    """Verify lightweight liveness probe."""
    response = client.get("/api/v1/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "GeoPol Bot"


def test_health_detailed_endpoint():
    """Verify detailed health check endpoint adheres to schema and contracts."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ["healthy", "degraded", "unhealthy"]
    assert data["app_name"] == "GeoPol Bot"
    assert data["version"] == "0.1.0"
    assert isinstance(data["uptime_seconds"], (int, float))
    assert "timestamp" in data

    # Verify component checks
    components = data["components"]
    assert "openai_api" in components
    assert "domain_gate" in components
    assert "knowledge_pipeline" in components
    assert "generation_engine" in components

    # Ensure raw API key is never leaked in health response
    raw_response_text = response.text
    assert "sk-proj-" not in raw_response_text
    assert "****" in raw_response_text or "NOT_CONFIGURED" in raw_response_text
