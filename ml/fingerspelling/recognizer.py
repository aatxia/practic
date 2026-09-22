"""
recognizer -- loads a trained fingerspelling checkpoint (ml/fingerspelling/
train.py) and classifies a single hand-landmark feature vector. Unlike
ml/inference/recognizer.py's SignRecognizer (a temporal window over many
frames for isolated *words*), this is a single-frame classifier for a
static handshape (one dactyl letter) -- no buffering, no sequence_length.

Framework-agnostic (no FastAPI/backend imports) so it's testable standalone;
backend/app/services/fingerspelling_service.py wraps this for live use.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ml.fingerspelling.model import FingerspellingMLP


class CheckpointNotFoundError(RuntimeError):
    """Raised when no fingerspelling checkpoint file exists at the configured path."""


@dataclass(frozen=True)
class LetterPrediction:
    letter: str
    confidence: float


class FingerspellingRecognizer:
    """Loads a checkpoint once; `predict()` classifies one 63-dim hand
    feature vector (ml/fingerspelling/dataset.py's hand_to_feature_vector())."""

    def __init__(self, checkpoint_path: Path | str, device: str = "cpu") -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(
                f"No trained fingerspelling checkpoint at {checkpoint_path}. Run "
                "`python -m ml.fingerspelling.build_dataset` then `python -m "
                "ml.fingerspelling.train` first (see PROJECT_STATUS.md)."
            )

        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        label_to_index: dict[str, int] = checkpoint["label_to_index"]
        self._model = FingerspellingMLP(
            num_classes=len(label_to_index), hidden_size=checkpoint["hidden_size"]
        ).to(self.device)
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()

        self._index_to_label: dict[int, str] = {index: label for label, index in label_to_index.items()}
        self.test_accuracy: float = checkpoint.get("test_accuracy", 0.0)

    @torch.no_grad()
    def predict(self, feature_vector: list[float]) -> LetterPrediction:
        if len(feature_vector) != 63:
            raise ValueError(f"Expected a 63-dim hand feature vector, got {len(feature_vector)}")
        tensor = torch.tensor([feature_vector], dtype=torch.float32, device=self.device)
        logits = self._model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        return LetterPrediction(
            letter=self._index_to_label[predicted_index],
            confidence=float(probabilities[predicted_index].item()),
        )

    @torch.no_grad()
    def predict_topk(self, feature_vector: list[float], k: int = 3) -> list[LetterPrediction]:
        """Diagnostic-only: the full ranked distribution over all letter
        classes, not just predict()'s argmax -- e.g. to see that the
        expected letter was rank 2 at a real but sub-threshold probability,
        not simply absent. Mirrors ml/inference/recognizer.py's
        SignRecognizer.predict_topk() for the word-level model."""
        if len(feature_vector) != 63:
            raise ValueError(f"Expected a 63-dim hand feature vector, got {len(feature_vector)}")
        tensor = torch.tensor([feature_vector], dtype=torch.float32, device=self.device)
        logits = self._model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top = torch.topk(probabilities, k=min(k, probabilities.shape[0]))
        return [
            LetterPrediction(letter=self._index_to_label[int(index)], confidence=float(prob))
            for prob, index in zip(top.values.tolist(), top.indices.tolist())
        ]
