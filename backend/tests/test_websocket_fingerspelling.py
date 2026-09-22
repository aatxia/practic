"""
Live integration of the real fingerspelling classifier (ml/fingerspelling/)
into the WebSocket handler -- separate from test_websocket_inference.py's
word-level (still synthetic-only) LSTM coverage. Uses a real trained tiny
checkpoint (two classes, well-separated) and a stub landmark extractor, same
spirit as test_websocket_inference.py's _RightHandOnlyExtractor.
"""
import base64

import cv2
import numpy as np
import websocket.handler as ws_handler
from app.main import app
from fastapi.testclient import TestClient

from ml.fingerspelling.dataset import hand_to_feature_vector
from ml.fingerspelling.recognizer import LetterPrediction
from ml.fingerspelling.train import main as train_main
from ml.preprocessing.landmarks import FrameLandmarks

client = TestClient(app)

# A real (if arbitrary) 21-point hand shape -- fed through the same
# hand_to_feature_vector() the live handler uses, so the tiny checkpoint
# below is trained on the exact feature space a real detection would land in.
_HAND_A = np.array([[i * 0.01, i * 0.02, 0.0] for i in range(21)], dtype=np.float32)
_HAND_B = np.array([[i * 0.02, i * 0.01, 0.0] for i in range(21)], dtype=np.float32)


def _blank_frame_data_url(width: int = 64, height: int = 48) -> str:
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _train_tiny_checkpoint(tmp_path):
    vector_a, reason_a = hand_to_feature_vector(None, _HAND_A)
    vector_b, reason_b = hand_to_feature_vector(None, _HAND_B)
    assert reason_a is None and reason_b is None

    rng = np.random.default_rng(0)
    features = np.concatenate(
        [
            vector_a + rng.normal(scale=0.01, size=(30, 63)).astype(np.float32),
            vector_b + rng.normal(scale=0.01, size=(30, 63)).astype(np.float32),
        ]
    )
    labels = np.array(["А"] * 30 + ["Б"] * 30, dtype="<U8")

    features_path = tmp_path / "features.npz"
    np.savez_compressed(features_path, features=features, labels=labels)

    checkpoint_path = tmp_path / "checkpoints" / "latest.pt"
    train_main(["--features", str(features_path), "--output", str(checkpoint_path), "--epochs", "40"])
    return checkpoint_path


class _RightHandOnlyExtractor:
    """Reports a fixed single-hand detection every frame -- no pose/face,
    matching test_websocket_inference.py's stub extractor convention."""

    def __init__(self, hand: np.ndarray):
        self._hand = hand

    def extract(self, frame_bgr):
        return FrameLandmarks(right_hand=self._hand)


class _BothHandsExtractor:
    def extract(self, frame_bgr):
        return FrameLandmarks(left_hand=_HAND_A, right_hand=_HAND_B)


class _MovingRightHandExtractor:
    """A right hand whose SHAPE (not just position -- normalize_hand() is
    translation/scale invariant, see ml/preprocessing/normalization.py) keeps
    changing call to call, cycling between _HAND_A and _HAND_B -- simulates
    a hand still in motion (e.g. a one-handed dynamic word sign in
    progress), unlike _RightHandOnlyExtractor's fixed handshape (a genuinely
    held letter)."""

    def __init__(self, cycle: int = 2):
        self._cycle = cycle
        self._call = 0

    def extract(self, frame_bgr):
        t = (self._call % self._cycle) / self._cycle
        self._call += 1
        hand = _HAND_A * (1 - t) + _HAND_B * t
        return FrameLandmarks(right_hand=hand.astype(np.float32))


class _FixedLetterService:
    """Always predicts the same letter regardless of input -- isolates the
    motion gate (this is about WHETHER a letter is allowed to confirm, not
    about classifier accuracy, already covered by test_confirms_a_real_
    letter_once_held_stably above)."""

    def is_ready(self) -> bool:
        return True

    def predict(self, feature_vector: list[float]) -> LetterPrediction:
        return LetterPrediction(letter="А", confidence=0.99)

    def predict_with_alternatives(self, feature_vector: list[float], k: int = 3):
        result = self.predict(feature_vector)
        return result, [result]


def test_no_letter_message_when_no_fingerspelling_checkpoint_is_loaded(
    monkeypatch, reset_inference_caches, reset_fingerspelling_caches
):
    """The fingerspelling path is additive/optional -- a missing checkpoint
    must never surface as an error, unlike the word-level ML-not-ready path."""
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("FINGERSPELLING_CHECKPOINT_PATH", "does/not/exist.pt")
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: _RightHandOnlyExtractor(_HAND_A))

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        ws.send_json({"type": "frame", "timestamp": 1, "data": _blank_frame_data_url()})
        status = ws.receive_json()
        assert status["type"] == "landmarks_status"
        # Next message is the word-level ML-not-ready error, not a letter message.
        response = ws.receive_json()
        assert response["type"] == "error"


def test_confirms_a_real_letter_once_held_stably(
    tmp_path, monkeypatch, reset_inference_caches, reset_fingerspelling_caches
):
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("FINGERSPELLING_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("FINGERSPELLING_STABILITY_FRAMES", "3")
    monkeypatch.setenv("FINGERSPELLING_CONFIDENCE_THRESHOLD", "0.5")
    # Word gets first priority when a real word checkpoint is loaded (see
    # recognition_mode in websocket/handler.py) -- isolate from the real
    # default checkpoint so this test is purely about fingerspelling, same
    # pattern as the other tests in this file.
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "does/not/exist.pt")
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: _RightHandOnlyExtractor(_HAND_A))

    frame_data = _blank_frame_data_url()
    letter_messages = []

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        for i in range(5):
            ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
            ws.receive_json()  # landmarks_status
            letter_msg = ws.receive_json()
            assert letter_msg["type"] in {"letter_prediction", "letter_confirmed"}
            letter_messages.append(letter_msg)
            ws.receive_json()  # word-level ML-not-ready error (unrelated path)

    assert any(m["type"] == "letter_confirmed" for m in letter_messages)
    confirmed = next(m for m in letter_messages if m["type"] == "letter_confirmed")
    assert confirmed["letter"] == "А"
    assert confirmed["is_final"] is True
    assert 0.0 <= confirmed["confidence"] <= 1.0


def test_confirmed_letter_carries_real_ranked_alternatives_for_manual_correction(
    tmp_path, monkeypatch, reset_inference_caches, reset_fingerspelling_caches
):
    """A signer correcting a wrong CONFIRMED letter needs the model's real
    top-k, not a guessed "similar letters" list -- confirms it's the actual
    trained checkpoint's ranked output (both real trained classes, "А" and
    "Б"), never present on the interim (not yet confirmed) messages."""
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("FINGERSPELLING_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("FINGERSPELLING_STABILITY_FRAMES", "3")
    monkeypatch.setenv("FINGERSPELLING_CONFIDENCE_THRESHOLD", "0.5")
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "does/not/exist.pt")
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: _RightHandOnlyExtractor(_HAND_A))

    frame_data = _blank_frame_data_url()
    letter_messages = []

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        for i in range(5):
            ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
            ws.receive_json()  # landmarks_status
            letter_messages.append(ws.receive_json())
            ws.receive_json()  # word-level ML-not-ready error (unrelated path)

    interim = [m for m in letter_messages if m["type"] == "letter_prediction"]
    assert interim, "expected at least one interim message before confirmation"
    for message in interim:
        assert message["top_k"] is None

    confirmed = next(m for m in letter_messages if m["type"] == "letter_confirmed")
    assert confirmed["top_k"] is not None
    candidate_letters = {c["letter"] for c in confirmed["top_k"]}
    assert candidate_letters == {"А", "Б"}
    # A real ranked distribution -- confidences sum close to 1 (softmax
    # over exactly the checkpoint's 2 trained classes), never fabricated.
    assert abs(sum(c["confidence"] for c in confirmed["top_k"]) - 1.0) < 0.01
    top_candidate = max(confirmed["top_k"], key=lambda c: c["confidence"])
    assert top_candidate["letter"] == confirmed["letter"]


def test_fingerspelling_debug_logging_does_not_change_behavior(
    tmp_path, monkeypatch, reset_inference_caches, reset_fingerspelling_caches
):
    """ws_debug_log_fingerspelling_pipeline (see app/core/config.py) is
    opt-in and must never change the actual letter_prediction/
    letter_confirmed stream, only add logging on the side."""
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("FINGERSPELLING_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("FINGERSPELLING_STABILITY_FRAMES", "3")
    monkeypatch.setenv("FINGERSPELLING_CONFIDENCE_THRESHOLD", "0.5")
    monkeypatch.setenv("WS_DEBUG_LOG_FINGERSPELLING_PIPELINE", "true")
    # See the isolation comment in test_confirms_a_real_letter_once_held_stably above.
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "does/not/exist.pt")
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: _RightHandOnlyExtractor(_HAND_A))

    frame_data = _blank_frame_data_url()
    letter_messages = []

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        for i in range(10):
            ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
            ws.receive_json()  # landmarks_status
            letter_msg = ws.receive_json()
            letter_messages.append(letter_msg)
            ws.receive_json()  # word-level ML-not-ready error (unrelated path)

    assert any(m["type"] == "letter_confirmed" for m in letter_messages)
    confirmed = next(m for m in letter_messages if m["type"] == "letter_confirmed")
    assert confirmed["letter"] == "А"
    assert confirmed["is_final"] is True


def test_no_letter_message_when_both_hands_detected(
    tmp_path, monkeypatch, reset_inference_caches, reset_fingerspelling_caches
):
    """Ambiguous for a single-handshape classifier -- silently skipped, same
    honesty convention as the offline dataset builder's skip logic."""
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("FINGERSPELLING_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: _BothHandsExtractor())

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "frame", "timestamp": 1, "data": _blank_frame_data_url()})
        status = ws.receive_json()
        assert status["type"] == "landmarks_status"
        response = ws.receive_json()
        assert response["type"] == "error"  # word-level path, not a letter message


def test_a_moving_one_handed_hand_never_confirms_a_letter(
    monkeypatch, reset_inference_caches, reset_fingerspelling_caches
):
    """Root-cause regression test: confirmed live (streaming real recorded
    word clips, e.g. data/real/raw/aatxia/WANT_00.mp4, through the real /ws
    path -- see PROJECT_STATUS.md) that a one-handed DYNAMIC word sign can
    hold the SAME letter prediction stably enough, long enough, to reach
    fingerspelling_stability_frames purely because the hand shape
    mid-gesture resembles a known static letter -- while the hand is
    genuinely still moving. The classifier here always predicts the same
    letter (_FixedLetterService) so the ONLY thing that can prevent
    confirmation is the motion gate, isolating it from classifier accuracy."""
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("FINGERSPELLING_STABILITY_FRAMES", "3")
    monkeypatch.setenv("FINGERSPELLING_CONFIDENCE_THRESHOLD", "0.5")
    # Isolate from the real word-level checkpoint (unrelated to this test) --
    # same "point at nothing" pattern the other tests in this file rely on
    # to get a predictable not-ready error for the word-level path.
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "does/not/exist.pt")
    # _get_landmark_extractor() is called once per frame in the real handler
    # (memoized via module globals); the stub must be memoized the same way
    # here -- a fresh instance per call would reset its internal frame
    # counter every time and the hand would never appear to move at all
    # (see test_facial_grammar_websocket.py's _FacialLandmarkExtractor for
    # the same gotcha).
    extractor_stub = _MovingRightHandExtractor()
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: extractor_stub)
    monkeypatch.setattr(ws_handler, "get_fingerspelling_service", lambda: _FixedLetterService())

    frame_data = _blank_frame_data_url()
    letter_messages = []

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        for i in range(20):
            ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
            ws.receive_json()  # landmarks_status
            letter_msg = ws.receive_json()
            letter_messages.append(letter_msg)
            ws.receive_json()  # word-level ML-not-ready error (unrelated path)

    assert len(letter_messages) == 20
    assert all(m["type"] == "letter_prediction" for m in letter_messages)
    assert all(m["is_final"] is False for m in letter_messages)
    assert not any(m["type"] == "letter_confirmed" for m in letter_messages)


def test_a_held_still_hand_still_confirms_a_letter_normally(
    monkeypatch, reset_inference_caches, reset_fingerspelling_caches
):
    """Regression guard for the fix above: the motion gate must only block a
    MOVING hand -- a genuinely held handshape (fixed every frame, zero
    motion) must keep confirming exactly as before."""
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("FINGERSPELLING_STABILITY_FRAMES", "3")
    monkeypatch.setenv("FINGERSPELLING_CONFIDENCE_THRESHOLD", "0.5")
    # See the isolation comment in test_confirms_a_real_letter_once_held_stably above.
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "does/not/exist.pt")
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: _RightHandOnlyExtractor(_HAND_A))
    monkeypatch.setattr(ws_handler, "get_fingerspelling_service", lambda: _FixedLetterService())

    frame_data = _blank_frame_data_url()
    letter_messages = []

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        for i in range(5):
            ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
            ws.receive_json()  # landmarks_status
            letter_msg = ws.receive_json()
            letter_messages.append(letter_msg)
            ws.receive_json()  # word-level ML-not-ready error

    assert any(m["type"] == "letter_confirmed" for m in letter_messages)
