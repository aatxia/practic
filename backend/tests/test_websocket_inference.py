"""
Phase 9-10: the WebSocket handler's buffering + real inference path, isolated
from Phase 6-7's CV pipeline (which has its own coverage in
ml/tests/test_landmarks.py and backend/tests/test_websocket.py) by stubbing
the landmark extractor. The real MediaPipe detector needs system libraries
(libEGL) not guaranteed to be present in every CI/sandbox -- stubbing it
here keeps this test about the NEW logic (sliding window -> checkpoint ->
PredictionMessage), not a redundant re-test of landmark detection.

Checkpoints trained here are deliberately tagged source="real_webcam" (tiny,
trivial clips, but real video decoded through the real MediaPipe pipeline --
see ml/tests/test_train.py's own single-signer test for the same pattern),
not demo_synthetic: the live WebSocket handler now treats a demo-mode
checkpoint the same as "no checkpoint at all" (an honest not-ready message,
never a fabricated gloss -- see websocket/handler.py), so these buffering/
aggregation tests need a non-demo checkpoint to actually reach the
prediction path they're testing.
"""
import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient
from ml.datasets.annotation import SampleAnnotation, write_annotations
from ml.datasets.synthetic import generate_demo_dataset
from ml.features.feature_vector import FeatureConfig, feature_vector_size
from ml.preprocessing.landmarks import FrameLandmarks
from ml.training.train import main as train_main

import websocket.handler as ws_handler
from app.main import app
from tests.ws_test_helpers import receive_skip_boundary

client = TestClient(app)

SEQUENCE_LENGTH = 32  # configs/model.yaml model.sequence_length


def _write_tiny_clip(path, num_frames=6, fps=30.0, width=64, height=48):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), fill_value=(i * 4) % 256, dtype=np.uint8))
    writer.release()


def _train_tiny_checkpoint(tmp_path):
    """A minimal but real (non-demo) checkpoint: two glosses (TAK/NI, chosen
    because both are real standalone entries in ml/nlp/lexicon.py, same as
    the old demo dataset's glosses) x 6 trivial real .mp4 clips each, one
    signer -- small enough to train in ~1s, real enough that
    source_tags != ["demo_synthetic"]."""
    signer_id = "test_signer"
    raw_dir = tmp_path / "real" / "raw" / signer_id
    raw_dir.mkdir(parents=True)
    annotations = []
    for gloss in ("TAK", "NI"):
        for take in range(6):
            clip_path = raw_dir / f"{gloss}_{take:02d}.mp4"
            _write_tiny_clip(clip_path)
            annotations.append(
                SampleAnnotation(
                    sample_id=f"{signer_id}_{gloss}_{take:02d}",
                    clip_path=clip_path.relative_to(tmp_path).as_posix(),
                    signer_id=signer_id,
                    gloss=gloss,
                    start_frame=0,
                    end_frame=6,
                    fps=30.0,
                    source="real_webcam",
                )
            )
    annotations_path = tmp_path / "annotations" / "test_annotations.jsonl"
    write_annotations(annotations_path, annotations)

    return train_main(
        [
            "--annotations",
            str(annotations_path),
            "--dataset-root",
            str(tmp_path),
            "--config",
            "../configs/model.yaml",
            "--experiment-name",
            "test_run",
            "--output-dir",
            str(tmp_path / "checkpoints"),
            "--epochs",
            "1",
            "--hidden-size",
            "8",
            "--num-layers",
            "1",
            "--device",
            "cpu",
        ]
    )


def _blank_frame_data_url(width: int = 64, height: int = 48) -> str:
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class _StubLandmarkExtractor:
    """An honest 'nothing detected in this frame' result (all fields None),
    same as the real extractor on a blank frame -- just without needing the
    real MediaPipe/EGL runtime for this test."""

    def extract(self, frame_bgr):
        return FrameLandmarks()


class _RightHandOnlyExtractor:
    """A stub reporting a detected right hand and pose, nothing else --
    exercises landmarks_status reporting non-uniform presence, not just
    all-True/all-False."""

    def extract(self, frame_bgr):
        return FrameLandmarks(
            right_hand=np.zeros((21, 3), dtype=np.float32),
            pose=np.zeros((33, 3), dtype=np.float32),
        )


def test_landmarks_status_reflects_actual_per_modality_detection(monkeypatch, reset_inference_caches):
    """Wiring test for the hand/pose-visibility indicator: landmarks_status
    must match ml/preprocessing/normalization.py's `present` dict exactly,
    including when only some modalities are detected -- not collapsed to a
    single boolean, and sent before the checkpoint is even loaded (it's
    independent of ML readiness, see websocket/handler.py)."""
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: _RightHandOnlyExtractor())

    frame_data = _blank_frame_data_url()
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        ws.send_json({"type": "frame", "timestamp": 1, "data": frame_data})
        status = ws.receive_json()

    assert status["type"] == "landmarks_status"
    assert status["left_hand"] is False
    assert status["right_hand"] is True
    assert status["pose"] is True
    assert status["face"] is False
    # Raw (x, y) points for the overlay (frontend/components/Camera), sent
    # alongside the booleans -- None for the modality that wasn't detected,
    # a real per-point list (matching MediaPipe's own landmark counts) for
    # the ones that were.
    assert status["left_hand_points"] is None
    assert status["right_hand_points"] is not None and len(status["right_hand_points"]) == 21
    assert status["pose_points"] is not None and len(status["pose_points"]) == 33


def test_sends_buffering_status_then_a_real_prediction_once_window_fills(
    tmp_path, monkeypatch, reset_inference_caches
):
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("WS_MAX_FPS", "100000")  # avoid rate-limiting a tight send loop
    # The stub extractor reports nothing detected every frame (an honest
    # zero-filled hand feature, identical every time) -- exercising the
    # motion gate (websocket/handler.py, ml/inference/motion_gate.py) isn't
    # this test's concern, only the buffering/prediction wiring is.
    monkeypatch.setenv("WS_MIN_HAND_MOTION", "-1")
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: _StubLandmarkExtractor())

    frame_data = _blank_frame_data_url()

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack

        for i in range(SEQUENCE_LENGTH - 1):
            ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
            ws.receive_json()  # landmarks_status
            response = receive_skip_boundary(ws)
            assert response["type"] == "error"
            assert f"Buffering: {i + 1}/{SEQUENCE_LENGTH}" in response["message"]

        ws.send_json({"type": "frame", "timestamp": SEQUENCE_LENGTH, "data": frame_data})
        ws.receive_json()  # landmarks_status
        response = receive_skip_boundary(ws)

        assert response["type"] == "prediction"
        # Interim (not yet confirmed) predictions show the raw gloss label --
        # composition into a full Ukrainian sentence only happens once the
        # gloss is confirmed (see the final_prediction test below).
        assert response["text"] in {"TAK", "NI"}
        assert response["is_final"] is False
        assert 0.0 <= response["confidence"] <= 1.0


def test_debug_diagnostics_do_not_change_behavior_and_save_a_real_feature_window(
    tmp_path, monkeypatch, reset_inference_caches
):
    """Live-vs-replay diagnostics (ws_debug_log_word_pipeline,
    ws_debug_save_feature_windows_dir -- see app/core/config.py) are
    opt-in and must never change the actual prediction stream, only add
    logging/saved files on the side."""
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    windows_dir = tmp_path / "debug_windows"
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("WS_MIN_HAND_MOTION", "-1")
    monkeypatch.setenv("WS_DEBUG_LOG_WORD_PIPELINE", "true")
    monkeypatch.setenv("WS_DEBUG_SAVE_FEATURE_WINDOWS_DIR", str(windows_dir))
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: _StubLandmarkExtractor())

    frame_data = _blank_frame_data_url()

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        for i in range(SEQUENCE_LENGTH + 2):
            ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
            ws.receive_json()  # landmarks_status
            response = receive_skip_boundary(ws)
            if i >= SEQUENCE_LENGTH - 1:
                assert response["type"] == "prediction"
                assert response["text"] in {"TAK", "NI"}

    saved = list(windows_dir.glob("window_*.npz"))
    assert len(saved) > 0
    data = np.load(saved[0])
    assert data["features"].shape == (SEQUENCE_LENGTH, feature_vector_size(FeatureConfig()))


def test_keeps_predicting_on_the_sliding_window_after_the_first_prediction(
    tmp_path, monkeypatch, reset_inference_caches
):
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("WS_MIN_HAND_MOTION", "-1")  # see the buffering test above
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: _StubLandmarkExtractor())

    frame_data = _blank_frame_data_url()

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        for i in range(SEQUENCE_LENGTH + 2):
            ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
            ws.receive_json()  # landmarks_status
            response = receive_skip_boundary(ws)
            if i >= SEQUENCE_LENGTH - 1:
                assert response["type"] == "prediction"


def test_confirms_a_final_prediction_once_the_gloss_stabilizes(tmp_path, monkeypatch, reset_inference_caches):
    """Phase 11: the same held sign, predicted repeatedly on the sliding
    window, must eventually produce exactly one 'final_prediction' (gloss
    aggregation), not stay interim forever."""
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("WS_GLOSS_STABILITY_FRAMES", "3")
    # A barely-trained tiny checkpoint fed a stub (all-absent) landmark input
    # won't reliably clear a real confidence bar -- 0.0 isolates aggregation
    # behavior (this test) from recognizer confidence calibration (already
    # covered by ml/tests/test_recognizer.py and test_lstm_inference_service.py).
    monkeypatch.setenv("WS_GLOSS_CONFIDENCE_THRESHOLD", "0.0")
    monkeypatch.setenv("WS_MIN_HAND_MOTION", "-1")  # see the buffering test above
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: _StubLandmarkExtractor())

    frame_data = _blank_frame_data_url()
    final_predictions = []

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        for i in range(SEQUENCE_LENGTH + 5):
            ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
            ws.receive_json()  # landmarks_status
            response = ws.receive_json()
            if response["type"] == "final_prediction":
                final_predictions.append(response)

    # Same stub landmarks every frame -> same predicted gloss every frame ->
    # confirmed exactly once (stability_frames=3), never re-confirmed while
    # the "signer" keeps holding it.
    assert len(final_predictions) == 1
    assert final_predictions[0]["is_final"] is True
    # Phase 12: TAK/NI are standalone words in the lexicon, so the confirmed
    # prediction's text is a real composed Ukrainian sentence, not the raw
    # gloss label.
    assert final_predictions[0]["text"] in {"Так.", "Ні."}


def test_confirmed_word_carries_real_ranked_alternatives_for_manual_correction(tmp_path, monkeypatch, reset_inference_caches):
    """A signer correcting a wrong CONFIRMED word needs the model's real
    top-k over its actual trained classes, not a guessed list -- and it
    must never appear on an interim (not yet confirmed) prediction."""
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setenv("WS_GLOSS_STABILITY_FRAMES", "3")
    monkeypatch.setenv("WS_GLOSS_CONFIDENCE_THRESHOLD", "0.0")
    monkeypatch.setenv("WS_MIN_HAND_MOTION", "-1")
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: _StubLandmarkExtractor())

    frame_data = _blank_frame_data_url()
    predictions = []

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        for i in range(SEQUENCE_LENGTH + 5):
            ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
            ws.receive_json()  # landmarks_status
            predictions.append(ws.receive_json())

    interim = [p for p in predictions if p["type"] == "prediction"]
    assert interim, "expected at least one interim prediction before confirmation"
    for message in interim:
        assert message["top_k"] is None

    confirmed = next(p for p in predictions if p["type"] == "final_prediction")
    assert confirmed["top_k"] is not None
    candidate_glosses = {c["gloss"] for c in confirmed["top_k"]}
    assert candidate_glosses == {"TAK", "NI"}
    assert abs(sum(c["confidence"] for c in confirmed["top_k"]) - 1.0) < 0.01
    top_candidate = max(confirmed["top_k"], key=lambda c: c["confidence"])
    assert top_candidate["gloss"] == confirmed["gloss"]


def test_a_demo_only_checkpoint_never_reaches_the_live_translation_stream(
    tmp_path, monkeypatch, reset_inference_caches
):
    """A demo/synthetic-only checkpoint carries no real information about
    the signer's actual gesture -- the handler must treat it the same as
    "no checkpoint at all" (an honest error, never a fabricated/flickering
    gloss), and that error must not leak "demo"/internal-settings wording
    into what the live translation panel would show."""
    generate_demo_dataset(tmp_path, sequence_length=8, samples_per_signer_gloss=4, seed=1)
    checkpoint_path = train_main(
        [
            "--annotations",
            str(tmp_path / "annotations" / "demo_annotations.jsonl"),
            "--dataset-root",
            str(tmp_path),
            "--config",
            "../configs/model.yaml",
            "--experiment-name",
            "test_run",
            "--output-dir",
            str(tmp_path / "checkpoints"),
            "--epochs",
            "1",
            "--hidden-size",
            "8",
            "--num-layers",
            "1",
            "--device",
            "cpu",
        ]
    )
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("WS_MAX_FPS", "100000")
    monkeypatch.setattr(ws_handler, "_get_landmark_extractor", lambda: _StubLandmarkExtractor())

    frame_data = _blank_frame_data_url()

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        for i in range(SEQUENCE_LENGTH + 2):
            ws.send_json({"type": "frame", "timestamp": i, "data": frame_data})
            ws.receive_json()  # landmarks_status
            response = receive_skip_boundary(ws)
            assert response["type"] == "error"
            assert "demo" not in response["message"].lower()
            assert "checkpoint_path" not in response["message"].lower()
