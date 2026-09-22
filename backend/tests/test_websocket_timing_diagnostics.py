"""
ws_debug_log_timing (websocket/handler.py, app/core/config.py) -- real
wall-clock per-stage timing, opt-in and off by default, added for the
continuous-dictation performance investigation (see PROJECT_STATUS.md):
a real performance bug has to be measured on the machine that's actually
slow, not guessed at from this sandbox (which has no camera).

Uses the same deterministic stubs as test_websocket_recognition_priority.py
so behavior can be compared with the flag on vs off -- the instrumentation
must never change what gets predicted/confirmed, only add log lines.
"""
import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient
from ml.fingerspelling.recognizer import LetterPrediction
from ml.inference.recognizer import RecognitionResult
from ml.preprocessing.landmarks import FrameLandmarks

import websocket.handler as ws_handler
from app.main import app
from app.services.inference_service import SignPrediction

client = TestClient(app)


def _blank_frame_data_url(width: int = 64, height: int = 48) -> str:
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


_HAND = np.array([[i * 0.01, i * 0.02, 0.0] for i in range(21)], dtype=np.float32)


class _HandPresenceExtractor:
    def extract(self, frame_bgr):
        return FrameLandmarks(right_hand=_HAND)


class _AlwaysConfirmsWordService:
    is_demo_mode = False
    sequence_length = 32

    def is_ready(self) -> bool:
        return True

    def predict(self, landmark_sequence):
        return SignPrediction(sign="TAK", text="TAK", confidence=0.95, is_final=False)

    def predict_with_alternatives(self, landmark_sequence, k: int = 3):
        result = self.predict(landmark_sequence)
        return result, [RecognitionResult(gloss=result.sign, confidence=result.confidence, is_demo_mode=False)]


class _FixedLetterService:
    def is_ready(self) -> bool:
        return True

    def predict(self, feature_vector):
        return LetterPrediction(letter="А", confidence=0.99)

    def predict_with_alternatives(self, feature_vector, k: int = 3):
        result = self.predict(feature_vector)
        return result, [result]


def _stream_frames(ws, frame_data, n):
    """Sends n frames, draining every message per frame until the terminal
    word prediction/final_prediction/error -- returns the list of terminal
    message dicts, one per frame, for behavior comparison."""
    terminal_messages = []
    for i in range(n):
        ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
        while True:
            message = ws.receive_json()
            if message["type"] in ("prediction", "final_prediction", "error"):
                terminal_messages.append(message)
                break


def test_timing_diagnostics_are_off_by_default(monkeypatch, reset_inference_caches, reset_fingerspelling_caches, caplog):
    import logging

    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("WS_MIN_HAND_MOTION", "-1")
    extractor_stub = _HandPresenceExtractor()
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: extractor_stub)
    monkeypatch.setattr(ws_handler, "get_inference_service", lambda: _AlwaysConfirmsWordService())
    monkeypatch.setattr(ws_handler, "get_fingerspelling_service", lambda: _FixedLetterService())

    frame_data = _blank_frame_data_url()
    with caplog.at_level(logging.INFO, logger="websocket.handler"):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            for i in range(35):
                ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
                while True:
                    message = ws.receive_json()
                    if message["type"] in ("prediction", "final_prediction", "error"):
                        break

    assert not any("[TIMING]" in record.getMessage() for record in caplog.records)


def test_timing_diagnostics_log_every_stage_when_enabled(
    monkeypatch, reset_inference_caches, reset_fingerspelling_caches, caplog
):
    import logging

    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("WS_MIN_HAND_MOTION", "-1")
    monkeypatch.setenv("WS_DEBUG_LOG_TIMING", "true")
    extractor_stub = _HandPresenceExtractor()
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: extractor_stub)
    monkeypatch.setattr(ws_handler, "get_inference_service", lambda: _AlwaysConfirmsWordService())
    monkeypatch.setattr(ws_handler, "get_fingerspelling_service", lambda: _FixedLetterService())

    frame_data = _blank_frame_data_url()
    # 60 frames: the rate limiter (frames_received % 30 == 0) fires on frame
    # 30 (word buffer not yet full, sequence_length=32 -- no word_lstm_
    # inference/total_pipeline line that time) and again on frame 60 (buffer
    # long full by then, so that pass logs every stage).
    with caplog.at_level(logging.INFO, logger="websocket.handler"):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            for i in range(60):
                ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
                while True:
                    message = ws.receive_json()
                    if message["type"] in ("prediction", "final_prediction", "error"):
                        break

    timing_lines = [record.getMessage() for record in caplog.records if "[TIMING]" in record.getMessage()]
    stages_logged = {line.split("stage=")[1].split()[0] for line in timing_lines}
    assert stages_logged == {
        "decode",
        "mediapipe_combined",
        "normalize",
        "feature_vector_build",
        "word_lstm_inference",
        "fingerspelling_inference",
        "total_pipeline",
    }
    # Real, parseable numeric values -- not a placeholder string.
    for line in timing_lines:
        ms_value = float(line.split("ms=")[1])
        assert ms_value >= 0.0


def test_timing_diagnostics_do_not_change_recognition_behavior(
    monkeypatch, reset_inference_caches, reset_fingerspelling_caches
):
    """The exact same stream of frames must confirm the exact same
    predictions whether ws_debug_log_timing is on or off -- instrumentation
    must never alter what gets recognized."""
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("WS_MIN_HAND_MOTION", "-1")
    monkeypatch.setenv("WS_GLOSS_STABILITY_FRAMES", "3")
    frame_data = _blank_frame_data_url()

    def run(timing_enabled: bool):
        if timing_enabled:
            monkeypatch.setenv("WS_DEBUG_LOG_TIMING", "true")
        else:
            monkeypatch.delenv("WS_DEBUG_LOG_TIMING", raising=False)
        extractor_stub = _HandPresenceExtractor()
        monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: extractor_stub)
        monkeypatch.setattr(ws_handler, "get_inference_service", lambda: _AlwaysConfirmsWordService())
        monkeypatch.setattr(ws_handler, "get_fingerspelling_service", lambda: _FixedLetterService())
        results = []
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            for i in range(40):
                ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
                while True:
                    message = ws.receive_json()
                    if message["type"] in ("prediction", "final_prediction", "error"):
                        results.append((message["type"], message.get("gloss"), message.get("confidence")))
                        break
        return results

    from app.core.config import get_settings
    from app.services import fingerspelling_provider, inference_provider

    without_timing = run(timing_enabled=False)
    get_settings.cache_clear()
    inference_provider.reset_inference_service_cache()
    fingerspelling_provider.reset_fingerspelling_service_cache()
    with_timing = run(timing_enabled=True)

    assert without_timing == with_timing
