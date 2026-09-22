"""
InferenceService — abstraction over sign-recognition inference.

STATUS: interface only. Real implementation requires the CV pipeline
(MediaPipe landmark extraction, Phase 6-7) and a trained temporal model
(Phase 9-10), neither of which exist yet.

We deliberately do NOT return a fake/hardcoded prediction here (see
master-prompt rule "НЕ РОБИ FAKE AI"). Calling predict() before a real
model is wired up raises MLNotReadyError with a clear message, so any
caller (WebSocket handler, tests) gets an honest signal instead of a
silently-wrong answer.

Once Phase 9/10 land, a concrete `MediaPipeSignRecognizer` (or
`TransformerSignRecognizer`, `VideoJEPASignRecognizer`, etc.) will
subclass `InferenceService` and be selected via `MODEL_TYPE` in .env /
configs/model.yaml — the rest of the backend (WebSocket handler) will
be unaffected by which concrete model is active.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class MLNotReadyError(RuntimeError):
    """Raised when inference is requested but no trained model is loaded yet."""


@dataclass
class SignPrediction:
    sign: str
    text: str
    confidence: float
    is_final: bool = False


class InferenceService(ABC):
    """Abstract interface every concrete sign-recognition backend must implement."""

    @abstractmethod
    def predict(self, landmark_sequence: list[list[float]]) -> SignPrediction:
        """
        Args:
            landmark_sequence: a temporal buffer of normalized landmark feature
                vectors (see ml/preprocessing/normalization.py, Phase 6-7).

        Returns:
            SignPrediction with sign label, mapped Ukrainian gloss text,
            confidence score, and whether this is a final (vs. interim)
            prediction for the current temporal window.
        """
        raise NotImplementedError

    @abstractmethod
    def is_ready(self) -> bool:
        """True once a real (trained) model checkpoint is loaded."""
        raise NotImplementedError


class NotConfiguredInferenceService(InferenceService):
    """
    Placeholder used until a real model checkpoint exists.

    This is NOT a fake classifier: it never returns a plausible-looking
    prediction. It fails loudly and explicitly, which is the correct
    behavior for an ML dependency that has not been trained yet.
    """

    def predict(self, landmark_sequence: list[list[float]]) -> SignPrediction:
        raise MLNotReadyError(
            "Sign-recognition ML pipeline is not implemented yet: no trained model is "
            "loaded. This requires Phase 6-7 (MediaPipe landmark extraction) and "
            "Phase 9-10 (baseline model training + real-time inference) to be "
            "completed with a real dataset. See PROJECT_STATUS.md."
        )

    def is_ready(self) -> bool:
        return False
