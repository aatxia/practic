"""
FingerspellingSignRecognizer -- adapter around ml/fingerspelling/
recognizer.py's real trained checkpoint (26 of 33 Ukrainian letters,
trained on real photos -- ml/fingerspelling/, PROJECT_STATUS.md). Unlike
InferenceService's word-level LSTM, this classifies one hand-landmark
feature vector per frame, no temporal buffering.

Distinct from InferenceService's NotConfiguredInferenceService: a missing
fingerspelling checkpoint is not an error condition callers need to
surface to the client (it's an additive, optional capability, not the
core "is ML implemented at all" question InferenceService answers) -- the
websocket handler just checks is_ready() and skips the letter path
silently when False, same as it already treats each modality (hand/pose/
face) as independently optional.
"""
from __future__ import annotations

from ml.fingerspelling.recognizer import (
    CheckpointNotFoundError,
    FingerspellingRecognizer,
    LetterPrediction,
)

__all__ = ["CheckpointNotFoundError", "FingerspellingSignRecognizer", "LetterPrediction"]


class FingerspellingSignRecognizer:
    def __init__(self, checkpoint_path, device: str = "cpu") -> None:
        self._recognizer = FingerspellingRecognizer(checkpoint_path, device=device)
        self.test_accuracy: float = self._recognizer.test_accuracy

    def is_ready(self) -> bool:
        return True

    def predict(self, feature_vector: list[float]) -> LetterPrediction:
        return self._recognizer.predict(feature_vector)

    def predict_topk(self, feature_vector: list[float], k: int = 3) -> list[LetterPrediction]:
        """Diagnostic-only passthrough -- see FingerspellingRecognizer.predict_topk()."""
        return self._recognizer.predict_topk(feature_vector, k=k)

    def predict_with_alternatives(self, feature_vector: list[float], k: int = 3) -> tuple[LetterPrediction, list[LetterPrediction]]:
        """The websocket handler's hot-path call: the top prediction PLUS
        the ranked alternatives, from exactly ONE forward pass (see
        LSTMSignRecognizer.predict_with_alternatives's docstring -- same
        reasoning, mirrored for the letter classifier)."""
        results = self._recognizer.predict_topk(feature_vector, k=k)
        return results[0], results


class NotConfiguredFingerspellingRecognizer:
    """Mirrors NotConfiguredInferenceService's honesty, but for the
    optional letter path -- predict() still fails loudly if ever called
    without a checkpoint (a caller bug), it's just that is_ready() lets
    the websocket handler avoid calling it at all rather than needing to
    catch an error every frame."""

    def is_ready(self) -> bool:
        return False

    def predict(self, feature_vector: list[float]) -> LetterPrediction:
        raise RuntimeError(
            "Fingerspelling classifier is not loaded (no checkpoint found) -- "
            "callers must check is_ready() before calling predict()."
        )
