"""API v1 Router aggregating all endpoint routers for GeoPol Bot."""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import health, gate, knowledge, generation

api_router = APIRouter()

# Health endpoints
api_router.include_router(health.router, prefix="", tags=["System Health"])

# Domain Gate classification endpoints
api_router.include_router(gate.router, prefix="", tags=["Domain Gate"])

# Knowledge Pipeline endpoints
api_router.include_router(knowledge.router, prefix="", tags=["Knowledge Pipeline"])

# Answer Generation endpoints
api_router.include_router(generation.router, prefix="", tags=["Answer Generation"])
