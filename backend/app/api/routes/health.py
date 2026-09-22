from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.health import HealthResponse
from app.services.inference_provider import get_inference_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    settings = get_settings()
    inference_service = get_inference_service()

    if not inference_service.is_ready():
        ml_pipeline_status = "not_implemented"
    elif getattr(inference_service, "is_demo_mode", False):
        # Phase 9: a checkpoint trained only on synthetic data -- honestly
        # distinct from "ready", which would imply real УЖМ recognition.
        ml_pipeline_status = "demo_mode"
    else:
        ml_pipeline_status = "ready"

    return HealthResponse(
        status="ok",
        app_env=settings.app_env,
        model_type=settings.model_type,
        features={
            "hands": settings.features_hands,
            "pose": settings.features_pose,
            "face": settings.features_face,
        },
        ml_pipeline_status=ml_pipeline_status,
    )
