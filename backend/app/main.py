"""
UKSL Translator backend — FastAPI entrypoint.

Run with:
    uvicorn app.main:app --reload

Phase 2 scope: application bootstrap, config, logging, health check, and
service-layer interfaces (see app/services/) wired up but not yet backed by
real ML models — those land in later phases per PROJECT_STATUS.md.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.translate import router as translate_router
from app.api.websocket.routes import router as websocket_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "Starting UKSL Translator backend | env=%s model_type=%s features(hands=%s pose=%s face=%s)",
        settings.app_env,
        settings.model_type,
        settings.features_hands,
        settings.features_pose,
        settings.features_face,
    )
    yield
    logger.info("Shutting down UKSL Translator backend")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="UKSL Translator API",
        version="0.1.0",
        description="Bidirectional Ukrainian Sign Language translator backend.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(translate_router)
    app.include_router(websocket_router)

    return app


app = create_app()
