"""
fingerspelling_provider -- process-wide, lazily-constructed
FingerspellingSignRecognizer (or its NotConfigured stand-in). Mirrors
inference_provider.py's caching: loading a checkpoint is one-time work,
so the websocket handler shares one instance across connections.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.fingerspelling_service import (
    CheckpointNotFoundError,
    FingerspellingSignRecognizer,
    NotConfiguredFingerspellingRecognizer,
)

logger = get_logger(__name__)

_fingerspelling_service: FingerspellingSignRecognizer | NotConfiguredFingerspellingRecognizer | None = None


def get_fingerspelling_service() -> FingerspellingSignRecognizer | NotConfiguredFingerspellingRecognizer:
    global _fingerspelling_service
    if _fingerspelling_service is not None:
        return _fingerspelling_service

    settings = get_settings()
    try:
        _fingerspelling_service = FingerspellingSignRecognizer(
            checkpoint_path=settings.fingerspelling_checkpoint_path_resolved,
            device=settings.fingerspelling_device,
        )
        logger.info(
            "Loaded fingerspelling checkpoint from %s (test_accuracy=%.3f)",
            settings.fingerspelling_checkpoint_path_resolved,
            _fingerspelling_service.test_accuracy,
        )
    except CheckpointNotFoundError as exc:
        logger.warning("Fingerspelling classifier unavailable: %s", exc)
        _fingerspelling_service = NotConfiguredFingerspellingRecognizer()
    return _fingerspelling_service


def reset_fingerspelling_service_cache() -> None:
    """Test-only: forces the next get_fingerspelling_service() call to reload."""
    global _fingerspelling_service
    _fingerspelling_service = None
