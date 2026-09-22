"""
recognizer — loads a trained checkpoint (Phase 9, ml/training/train.py) and
runs inference on a buffered feature-vector sequence. Framework-agnostic (no
FastAPI/backend imports) so it's testable standalone; Phase 10 wraps it in
backend/app/services/lstm_inference_service.py for the InferenceService
interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ml.features.feature_vector import FeatureConfig
from ml.models.lstm import LSTMConfig, LSTMSignClassifier


class CheckpointNotFoundError(RuntimeError):
    """Raised when no checkpoint file exists at the configured path."""


@dataclass(frozen=True)
class RecognitionResult:
    gloss: str
    confidence: float
    is_demo_mode: bool


class SignRecognizer:
    """Loads a checkpoint once; `predict()` runs the model on an
    already-buffered window of exactly `sequence_length` feature vectors --
    owning that sliding-window buffer is the caller's job (a per-connection
    concern in the WebSocket handler), not this class's."""

    def __init__(self, checkpoint_path: Path | str, device: str = "cpu") -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(
                f"No trained checkpoint at {checkpoint_path}. Run `python -m "
                "ml.training.train` first (see PROJECT_STATUS.md, Phase 9)."
            )

        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        model_config = LSTMConfig(**checkpoint["model_config"])
        self._model = LSTMSignClassifier(model_config).to(self.device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        self.sequence_length: int = checkpoint["sequence_length"]
        self.feature_config = FeatureConfig(**checkpoint["feature_config"])
        self.is_demo_mode: bool = checkpoint["demo_mode"]
        self.source_tags: list[str] = checkpoint["source_tags"]
        self._index_to_label: dict[int, str] = {
            index: label for label, index in checkpoint["label_to_index"].items()
        }
        # The checkpoint's own authoritative class list, for callers that
        # need to check it against something external (e.g. a startup
        # vocabulary-compatibility check) -- never used internally by
        # predict()/predict_topk(), which only ever go through
        # _index_to_label built directly above.
        self.labels: frozenset[str] = frozenset(self._index_to_label.values())

    @torch.no_grad()
    def predict(self, feature_sequence: list[list[float]]) -> RecognitionResult:
        if len(feature_sequence) != self.sequence_length:
            raise ValueError(
                f"Expected exactly {self.sequence_length} frames, got {len(feature_sequence)} "
                "-- the caller must buffer a full window before calling predict()."
            )
        tensor = torch.tensor([feature_sequence], dtype=torch.float32, device=self.device)
        logits = self._model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        return RecognitionResult(
            gloss=self._index_to_label[predicted_index],
            confidence=float(probabilities[predicted_index].item()),
            is_demo_mode=self.is_demo_mode,
        )

    @torch.no_grad()
    def predict_topk(self, feature_sequence: list[list[float]], k: int = 3) -> list[RecognitionResult]:
        """Diagnostic-only: the full ranked distribution over all classes,
        not just the argmax predict() returns -- lets a live-vs-replay
        investigation see e.g. "WANT was rank 2 at 0.31, not just absent"
        instead of a single opaque top-1 number."""
        if len(feature_sequence) != self.sequence_length:
            raise ValueError(
                f"Expected exactly {self.sequence_length} frames, got {len(feature_sequence)} "
                "-- the caller must buffer a full window before calling predict_topk()."
            )
        tensor = torch.tensor([feature_sequence], dtype=torch.float32, device=self.device)
        logits = self._model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top = torch.topk(probabilities, k=min(k, probabilities.shape[0]))
        return [
            RecognitionResult(
                gloss=self._index_to_label[int(index)],
                confidence=float(prob),
                is_demo_mode=self.is_demo_mode,
            )
            for prob, index in zip(top.values.tolist(), top.indices.tolist())
        ]
