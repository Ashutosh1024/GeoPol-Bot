"""Main entry point for the GeoPol Bot FastAPI backend."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.core.config import get_settings
from backend.app.api.v1.router import api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("geopol_bot.backend")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle events."""
    settings = get_settings()
    logger.info("Initializing %s v%s in %s environment...", settings.project_name, settings.project_version, settings.environment)
    if settings.is_openai_configured:
        logger.info("OpenAI API configured (Key: %s, Model: %s)", settings.masked_openai_api_key, settings.openai_model)
    else:
        logger.warning("OpenAI API key NOT configured. Gating and generation will run in mock/standby mode until key is provided.")

    yield

    logger.info("Shutting down %s...", settings.project_name)


def create_app() -> FastAPI:
    """FastAPI application factory for GeoPol Bot."""
    settings = get_settings()

    app = FastAPI(
        title=settings.project_name,
        version=settings.project_version,
        description=(
            "Specialized Geopolitics Chatbot API. "
            "Enforces strict domain gating, verified RAG evidence retrieval, "
            "and OpenAI Responses API answer generation."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.api_v1_prefix}/openapi.json",
        lifespan=lifespan,
    )

    # CORS Middleware configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API v1 router
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/", summary="Root Endpoint", tags=["System Health"])
    async def root() -> JSONResponse:
        """Root endpoint returning service identity and health links."""
        return JSONResponse(
            content={
                "service": settings.project_name,
                "version": settings.project_version,
                "status": "online",
                "docs_url": "/docs",
                "health_url": f"{settings.api_v1_prefix}/health",
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=settings.debug,
    )
