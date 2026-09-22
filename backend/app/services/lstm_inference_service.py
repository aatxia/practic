"""
LSTMSignRecognizer — InferenceService adapter around ml/inference/recognizer.py
(Phase 9's trained checkpoint). Translates between the WebSocket handler's
buffered feature vectors and the app-level SignPrediction/MLNotReadyError
contract; the actual checkpoint loading and forward pass live in ml/inference/
(framework-agnostic, reused as-is here).
"""
from __future__ import annotations

from pathlib import Path

from app.services.inference_service import InferenceService, SignPrediction
from ml.inference.recognizer import CheckpointNotFoundError, RecognitionResult, SignRecognizer

__all__ = ["CheckpointNotFoundError", "LSTMSignRecognizer"]


class LSTMSignRecognizer(InferenceService):
    def __init__(self, checkpoint_path: Path | str, device: str = "cpu") -> None:
        self._recognizer = SignRecognizer(checkpoint_path, device=device)
        self.sequence_length: int = self._recognizer.sequence_length
        self.is_demo_mode: bool = self._recognizer.is_demo_mode

    def is_ready(self) -> bool:
        return True

    def predict(self, landmark_sequence: list[list[float]]) -> SignPrediction:
        result = self._recognizer.predict(landmark_sequence)
        # demo_mode checkpoints are trained on synthetic placeholder glosses
        # (see ml/datasets/synthetic.py) -- tagging the text makes it
        # impossible to mistake this output for real УЖМ recognition.
        text = f"[DEMO] {result.gloss}" if result.is_demo_mode else result.gloss
        return SignPrediction(sign=result.gloss, text=text, confidence=result.confidence, is_final=False)

    def predict_topk(self, landmark_sequence: list[list[float]], k: int = 3) -> list[RecognitionResult]:
        """Diagnostic-only passthrough -- see SignRecognizer.predict_topk()."""
        return self._recognizer.predict_topk(landmark_sequence, k=k)

    def predict_with_alternatives(
        self, landmark_sequence: list[list[float]], k: int = 3
    ) -> tuple[SignPrediction, list[RecognitionResult]]:
        """The websocket handler's hot-path call: the same SignPrediction
        predict() returns, PLUS the ranked alternatives, from exactly ONE
        forward pass -- not predict() followed by a second predict_topk()
        call, which would run the model twice per frame. This is what
        lets a person correct a wrong CONFIRMED word after the fact (see
        types/api.ts's top_k field) from real model output, never an
        invented list of "similar" words."""
        results = self._recognizer.predict_topk(landmark_sequence, k=k)
        top = results[0]
        text = f"[DEMO] {top.gloss}" if top.is_demo_mode else top.gloss
        prediction = SignPrediction(sign=top.gloss, text=text, confidence=top.confidence, is_final=False)
        return prediction, results
