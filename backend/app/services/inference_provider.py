"""
inference_provider — process-wide, lazily-constructed InferenceService.

Loading a checkpoint is one-time work (deserializing tensors), and its
readiness doesn't change while the process runs, so /health and the
WebSocket handler share one cached instance instead of each loading their
own copy -- mirrors websocket/handler.py's landmark-extractor caching.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.inference_service import (
    InferenceService,
    NotConfiguredInferenceService,
)
from app.services.lstm_inference_service import (
    CheckpointNotFoundError,
    LSTMSignRecognizer,
)

logger = get_logger(__name__)

_inference_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is not None:
        return _inference_service

    settings = get_settings()
    try:
        _inference_service = LSTMSignRecognizer(
            checkpoint_path=settings.model_checkpoint_path_resolved, device=settings.model_device
        )
        logger.info(
            "Loaded inference checkpoint from %s (demo_mode=%s)",
            settings.model_checkpoint_path_resolved,
            _inference_service.is_demo_mode,
        )
    except CheckpointNotFoundError as exc:
        logger.warning("Inference service unavailable: %s", exc)
        _inference_service = NotConfiguredInferenceService()
    return _inference_service


def reset_inference_service_cache() -> None:
    """Test-only: forces the next get_inference_service() call to reload
    (e.g. after pointing MODEL_CHECKPOINT_PATH at a freshly trained
    checkpoint via a settings override)."""
    global _inference_service
    _inference_service = None
