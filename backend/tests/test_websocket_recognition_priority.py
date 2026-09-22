"""
recognition_mode (websocket/handler.py) -- word gets first priority; only
after the word buffer has been full for a grace period without confirming
does fingerspelling get allowed to confirm at all, composing a word
letter by letter instead. Before this, both classifiers could confirm
independently and simultaneously -- see PROJECT_STATUS.md for the
motivating symptom ("Ж 6/8" while "Слово: —" scrolled forever, both
racing at once).

Uses fully deterministic stubs for BOTH classifiers (not a real trained
checkpoint's behavior on synthetic stub features, which isn't guaranteed
one way or the other -- confirmed flaky in practice) so confirmation
timing is exact and reproducible.
"""
import base64
import logging

import cv2
import numpy as np
import websocket.handler as ws_handler
from app.main import app
from app.services.inference_service import SignPrediction
from fastapi.testclient import TestClient
from tests.ws_test_helpers import receive_skip_boundary

from ml.fingerspelling.recognizer import LetterPrediction
from ml.inference.recognizer import RecognitionResult
from ml.preprocessing.landmarks import FrameLandmarks

client = TestClient(app)


def _blank_frame_data_url(width: int = 64, height: int = 48) -> str:
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class _NeverConfirmsWordService:
    """Deterministically never confirms: alternates between two glosses
    every call, so the aggregator's stability streak can never build past
    1 -- unlike a real trained checkpoint's behavior on synthetic stub
    features, which depends on that specific checkpoint's random decision
    boundary and isn't reliable to assert on."""

    is_demo_mode = False
    sequence_length = 32

    def __init__(self):
        self._call = 0

    def is_ready(self) -> bool:
        return True

    def predict(self, landmark_sequence: list[list[float]]) -> SignPrediction:
        self._call += 1
        gloss = "TAK" if self._call % 2 == 0 else "NI"
        return SignPrediction(sign=gloss, text=gloss, confidence=0.9, is_final=False)

    def predict_with_alternatives(self, landmark_sequence: list[list[float]], k: int = 3):
        result = self.predict(landmark_sequence)
        return result, [RecognitionResult(gloss=result.sign, confidence=result.confidence, is_demo_mode=False)]


class _FixedLetterService:
    """Always predicts the same letter -- isolates the routing/priority
    logic under test from fingerspelling classifier accuracy."""

    def is_ready(self) -> bool:
        return True

    def predict(self, feature_vector: list[float]) -> LetterPrediction:
        return LetterPrediction(letter="А", confidence=0.99)

    def predict_with_alternatives(self, feature_vector: list[float], k: int = 3):
        result = self.predict(feature_vector)
        return result, [result]


_HAND_A = np.array([[i * 0.01, i * 0.02, 0.0] for i in range(21)], dtype=np.float32)


class _HandPresenceExtractor:
    """Reports a fixed, held-still right-hand shape (real fingerspelling
    behavior -- motion isn't this test's concern) for the first
    `no_hand_after` calls, then no hand at all -- a sustained pause."""

    def __init__(self, no_hand_after: int | None = None):
        self._call = 0
        self._no_hand_after = no_hand_after

    def extract(self, frame_bgr):
        self._call += 1
        if self._no_hand_after is not None and self._call > self._no_hand_after:
            return FrameLandmarks()
        return FrameLandmarks(right_hand=_HAND_A)


def test_letters_never_confirm_while_word_still_has_its_grace_period(
    monkeypatch, reset_inference_caches, reset_fingerspelling_caches
):
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("WS_MIN_HAND_MOTION", "-1")  # isolate from the motion gate, not this test's concern
    monkeypatch.setenv("FINGERSPELLING_MAX_HAND_MOTION", "100")  # same, for the letter side
    monkeypatch.setenv("WS_GLOSS_STABILITY_FRAMES", "2")  # keep the grace period short for a fast test
    monkeypatch.setenv("FINGERSPELLING_STABILITY_FRAMES", "3")
    extractor_stub = _HandPresenceExtractor()
    word_stub = _NeverConfirmsWordService()
    letter_stub = _FixedLetterService()
    # _get_landmark_extractor()/get_inference_service()/get_fingerspelling_
    # service() are each called once per frame in the real handler
    # (memoized via module globals); every stub here must be memoized the
    # same way -- a fresh instance per call would reset internal frame
    # counters every time (see test_websocket_fingerspelling.py's
    # _MovingRightHandExtractor for the same gotcha).
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: extractor_stub)
    monkeypatch.setattr(ws_handler, "get_inference_service", lambda: word_stub)
    monkeypatch.setattr(ws_handler, "get_fingerspelling_service", lambda: letter_stub)

    frame_data = _blank_frame_data_url()
    letter_messages = []
    word_messages = []

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        # 32 to fill the word buffer + 2*3=6 grace-period frames + a few
        # more for the letter to confirm (fingerspelling_stability_frames=3)
        # once the fallback opens the gate, with margin.
        for i in range(50):
            ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
            ws.receive_json()  # landmarks_status
            letter_messages.append(ws.receive_json())
            word_messages.append(ws.receive_json())

    # The word service never confirms by construction -- verify it really
    # never reached final_prediction.
    assert not any(m.get("type") == "final_prediction" for m in word_messages)
    # Real interim letter guesses DID happen throughout (never silently
    # withheld) ...
    assert any(m["type"] == "letter_prediction" for m in letter_messages)
    # ... but none of them were allowed to confirm before the fallback.
    first_confirmed_index = next(
        (i for i, m in enumerate(letter_messages) if m["type"] == "letter_confirmed"), None
    )
    assert first_confirmed_index is not None, "letter should confirm eventually once word falls back"
    assert first_confirmed_index >= 37, "must not confirm before the word model's grace period elapsed"


def test_a_held_still_hand_falls_back_to_letter_via_the_real_motion_gate(
    monkeypatch, reset_inference_caches, reset_fingerspelling_caches
):
    """Regression test for a bug caught only by live testing, not by the
    test above: a genuinely held-still hand (zero motion -- real,
    grammatically correct fingerspelling) fails the word gate's motion
    check (ws_min_hand_motion, default 0.4) on every single frame, so it
    never reaches gloss_aggregator.update() at all. The fallback tracking
    originally lived only inside that update() branch, so a static letter
    could never advance word_buffer_full_streak and recognition_mode got
    stuck at "word" forever -- confirmed via live testing: streaming a real
    held letter photo for 60 frames never produced letter_confirmed.
    Unlike the test above, this one deliberately leaves the real motion
    gates enabled (their defaults) instead of disabling them, so it
    exercises the motion-gate-reject path directly."""
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("WS_GLOSS_STABILITY_FRAMES", "2")  # keep the grace period short for a fast test
    monkeypatch.setenv("FINGERSPELLING_STABILITY_FRAMES", "3")
    extractor_stub = _HandPresenceExtractor()  # same fixed hand every frame -> zero motion
    word_stub = _NeverConfirmsWordService()
    letter_stub = _FixedLetterService()
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: extractor_stub)
    monkeypatch.setattr(ws_handler, "get_inference_service", lambda: word_stub)
    monkeypatch.setattr(ws_handler, "get_fingerspelling_service", lambda: letter_stub)

    frame_data = _blank_frame_data_url()
    letter_messages = []

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        for i in range(60):
            ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
            ws.receive_json()  # landmarks_status
            letter_messages.append(ws.receive_json())
            ws.receive_json()  # word error ("no hand motion detected") every frame

    first_confirmed_index = next(
        (i for i, m in enumerate(letter_messages) if m["type"] == "letter_confirmed"), None
    )
    assert first_confirmed_index is not None, (
        "a held-still hand must still fall back to letter mode and confirm "
        "-- the motion-gate-reject path has to count toward the fallback too"
    )


def test_recognition_mode_falls_back_to_letter_then_resets_to_word_on_a_sustained_pause(
    monkeypatch, reset_inference_caches, reset_fingerspelling_caches, caplog
):
    """Verifies the actual mode transitions (not just their side effects)
    via the LIVE WORD DEBUG diagnostic log's mode= field (fires every 5th
    frame regardless of hand presence, unlike the fingerspelling log which
    only fires when a hand is detected) -- word -> letter once the grace
    period elapses, then back to word after a sustained no-hand pause (the
    same boundary letter_aggregator/gloss_aggregator already reset on
    individually), ready for a fresh word attempt."""
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("WS_MIN_HAND_MOTION", "-1")
    monkeypatch.setenv("FINGERSPELLING_MAX_HAND_MOTION", "100")
    monkeypatch.setenv("WS_GLOSS_STABILITY_FRAMES", "2")
    monkeypatch.setenv("FINGERSPELLING_STABILITY_FRAMES", "3")
    monkeypatch.setenv("WS_DEBUG_LOG_WORD_PIPELINE", "true")
    # no_hand_after=45: word's grace period (32 buffer-fill + 2*3=6) elapses
    # around frame 38, falling back to "letter"; a sustained pause then
    # starts at frame 46, long enough (>= 8 frames, see
    # _MODE_RESET_NO_HAND_FRAMES) to reset back to "word" before frame 65.
    extractor_stub = _HandPresenceExtractor(no_hand_after=45)
    word_stub = _NeverConfirmsWordService()
    letter_stub = _FixedLetterService()
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: extractor_stub)
    monkeypatch.setattr(ws_handler, "get_inference_service", lambda: word_stub)
    monkeypatch.setattr(ws_handler, "get_fingerspelling_service", lambda: letter_stub)

    frame_data = _blank_frame_data_url()

    with caplog.at_level(logging.INFO, logger="websocket.handler"):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            for i in range(65):
                ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
                ws.receive_json()  # landmarks_status
                if i < 45:  # hand still present -- a letter message goes out
                    ws.receive_json()  # letter_prediction/letter_confirmed
                # A "boundary" event (token/utterance) is inserted between
                # landmarks_status and the letter/word messages once
                # no_hand_streak crosses its threshold -- skip past it to
                # reach the word prediction/final_prediction/error this
                # loop actually reads.
                receive_skip_boundary(ws)

    modes_logged = [
        line.split("mode=")[1].split()[0]
        for record in caplog.records
        for line in record.getMessage().splitlines()
        if "conn=" in line and "mode=" in line
    ]
    assert modes_logged[0] == "word", "should start out in word-priority mode"
    letter_index = modes_logged.index("letter") if "letter" in modes_logged else None
    assert letter_index is not None, "should fall back to letter mode once the grace period elapses"
    assert "word" in modes_logged[letter_index:], "should reset back to word mode after the sustained pause"
