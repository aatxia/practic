"""
Continuous-dictation "boundary" events (websocket/protocol.py,
websocket/handler.py) -- a sustained no-hand streak signals "the current
attempt is over", at two configurable thresholds: a short pause finalizes
whatever's mid-flight (a "token" boundary, mainly for an in-progress
fingerspelled word); a longer pause additionally means "run language
post-processing on the whole utterance now" (an "utterance" boundary).
Fires exactly once per threshold crossing, not repeated every frame the
hand stays away -- see ws_token_boundary_no_hand_frames/
ws_utterance_boundary_no_hand_frames in app/core/config.py.

Independent of ML readiness by design (boundary detection only needs real
landmark extraction, not a loaded word/fingerspelling checkpoint) -- both
checkpoint paths are pointed at nonexistent files here to keep these tests
hermetic and fast, same isolation pattern as test_websocket_fingerspelling.py.
"""
import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient
from ml.preprocessing.landmarks import FrameLandmarks

import websocket.handler as ws_handler
from app.main import app

client = TestClient(app)


def _blank_frame_data_url(width: int = 64, height: int = 48) -> str:
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


_HAND = np.zeros((21, 3), dtype=np.float32)


class _HandPresenceExtractor:
    """Reports a hand present for the first `hand_frames` calls, then no
    hand at all -- a controllable pause for exercising the two boundary
    thresholds without needing a real webcam or MediaPipe."""

    def __init__(self, hand_frames: int) -> None:
        self._call = 0
        self._hand_frames = hand_frames

    def extract(self, frame_bgr):
        self._call += 1
        if self._call <= self._hand_frames:
            return FrameLandmarks(right_hand=_HAND)
        return FrameLandmarks()


def _boundary_kinds_received(ws, num_frames: int, frame_data: str) -> list[str]:
    """Sends num_frames frames, returning every 'boundary' message's `kind`
    seen, in arrival order -- draining and discarding every other message
    type (landmarks_status, letter_prediction, prediction/error) since this
    helper only cares about boundary events."""
    kinds = []
    for i in range(num_frames):
        ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
        while True:
            message = ws.receive_json()
            if message["type"] == "boundary":
                kinds.append(message["kind"])
            # landmarks_status is always exactly one message; everything
            # else this frame (an optional boundary, an optional letter
            # message, and exactly one word prediction/error) all arrive
            # before the next frame can be sent -- "error" is always the
            # frame's last message (no checkpoint configured here).
            if message["type"] == "error":
                break
    return kinds


def test_no_boundary_while_a_hand_stays_present(monkeypatch, reset_inference_caches, reset_fingerspelling_caches):
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "does/not/exist.pt")
    monkeypatch.setenv("FINGERSPELLING_CHECKPOINT_PATH", "does/not/exist.pt")
    monkeypatch.setenv("WS_TOKEN_BOUNDARY_NO_HAND_FRAMES", "3")
    monkeypatch.setenv("WS_UTTERANCE_BOUNDARY_NO_HAND_FRAMES", "6")
    # _get_landmark_extractor() is called once per frame in the real
    # handler -- a fresh instance per lambda call would reset _call to 0
    # every time, so the stub must be instantiated once, outside the lambda.
    extractor_stub = _HandPresenceExtractor(hand_frames=20)
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: extractor_stub)

    frame_data = _blank_frame_data_url()
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        kinds = _boundary_kinds_received(ws, num_frames=10, frame_data=frame_data)

    assert kinds == []


def test_token_boundary_fires_exactly_once_at_its_threshold(
    monkeypatch, reset_inference_caches, reset_fingerspelling_caches
):
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "does/not/exist.pt")
    monkeypatch.setenv("FINGERSPELLING_CHECKPOINT_PATH", "does/not/exist.pt")
    monkeypatch.setenv("WS_TOKEN_BOUNDARY_NO_HAND_FRAMES", "3")
    # Set far out of reach so only "token" is exercised by this test.
    monkeypatch.setenv("WS_UTTERANCE_BOUNDARY_NO_HAND_FRAMES", "1000")
    extractor_stub = _HandPresenceExtractor(hand_frames=2)
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: extractor_stub)

    frame_data = _blank_frame_data_url()
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        # No hand from frame index 2 onward; streak hits 3 (the token
        # threshold) at frame index 4 -- a few extra frames of margin to
        # also prove it does NOT refire on every subsequent no-hand frame.
        kinds = _boundary_kinds_received(ws, num_frames=10, frame_data=frame_data)

    assert kinds == ["token"]


def test_utterance_boundary_fires_after_token_once_the_longer_pause_elapses(
    monkeypatch, reset_inference_caches, reset_fingerspelling_caches
):
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "does/not/exist.pt")
    monkeypatch.setenv("FINGERSPELLING_CHECKPOINT_PATH", "does/not/exist.pt")
    monkeypatch.setenv("WS_TOKEN_BOUNDARY_NO_HAND_FRAMES", "3")
    monkeypatch.setenv("WS_UTTERANCE_BOUNDARY_NO_HAND_FRAMES", "6")
    extractor_stub = _HandPresenceExtractor(hand_frames=1)
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: extractor_stub)

    frame_data = _blank_frame_data_url()
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        kinds = _boundary_kinds_received(ws, num_frames=12, frame_data=frame_data)

    # "token" always precedes "utterance" for the same pause -- never the
    # reverse, and each fires exactly once, not on every no-hand frame past
    # its own threshold.
    assert kinds == ["token", "utterance"]


def test_boundary_streak_resets_when_the_hand_reappears(
    monkeypatch, reset_inference_caches, reset_fingerspelling_caches
):
    """A pause that almost reaches the token threshold, then a hand
    reappearing, then a second full pause -- the second pause must count
    from zero, not pick up where the first one left off (the same
    "presence resets the streak" rule no_hand_streak already applies to
    recognition_mode)."""
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "does/not/exist.pt")
    monkeypatch.setenv("FINGERSPELLING_CHECKPOINT_PATH", "does/not/exist.pt")
    monkeypatch.setenv("WS_TOKEN_BOUNDARY_NO_HAND_FRAMES", "3")
    monkeypatch.setenv("WS_UTTERANCE_BOUNDARY_NO_HAND_FRAMES", "1000")

    class _TwoPauses:
        """hand, hand, [no hand x2 -- short of the threshold=3], hand,
        [no hand x4 -- crosses the threshold=3 fresh]."""

        def __init__(self) -> None:
            self._call = 0

        def extract(self, frame_bgr):
            self._call += 1
            no_hand_calls = {3, 4, 7, 8, 9, 10}
            if self._call in no_hand_calls:
                return FrameLandmarks()
            return FrameLandmarks(right_hand=_HAND)

    extractor_stub = _TwoPauses()
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: extractor_stub)

    frame_data = _blank_frame_data_url()
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        kinds = _boundary_kinds_received(ws, num_frames=10, frame_data=frame_data)

    # The first (2-frame) pause never reaches the 3-frame threshold; only
    # the second (4-frame) pause, counted fresh from zero, does.
    assert kinds == ["token"]
