import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import app

client = TestClient(app)


def _blank_frame_data_url(width: int = 64, height: int = 48) -> str:
    """A real, valid JPEG (blank/no hand) as a base64 data URL -- the same
    format the frontend's canvas.toDataURL() produces."""
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def test_websocket_sends_connection_ack_on_connect():
    with client.websocket_connect("/ws") as ws:
        ack = ws.receive_json()
        assert ack == {"type": "connection", "status": "ok"}


def test_valid_frame_gets_an_honest_ml_not_implemented_error_not_a_fake_prediction(
    tmp_path, monkeypatch, reset_inference_caches
):
    """Core 'NO FAKE AI' rule: real landmarks are extracted (Phase 6-7), but no
    trained temporal model exists yet, so a well-formed frame must get an
    explicit error -- never a fabricated sign/text. Points MODEL_CHECKPOINT_PATH
    at a guaranteed-nonexistent file rather than relying on the default
    checkpoint being absent -- that default now points at a real, sometimes-
    present checkpoint (see PROJECT_STATUS.md's real_v1 word classifier)."""
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(tmp_path / "nonexistent.pt"))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # connection ack
        ws.send_json({"type": "frame", "timestamp": 1, "data": _blank_frame_data_url()})

        # landmarks_status precedes the prediction/error for every processed
        # frame (Phase: hand-visibility indicator) -- a blank frame has no
        # hand in it, so this must honestly say so too.
        status = ws.receive_json()
        assert status == {
            "type": "landmarks_status",
            "left_hand": False,
            "right_hand": False,
            "pose": False,
            "face": False,
            "left_hand_points": None,
            "right_hand_points": None,
            "pose_points": None,
        }

        response = ws.receive_json()
        assert response["type"] == "error"
        assert "not available" in response["message"].lower()
        # A blank frame has no hand in it -- must honestly say so, not fabricate one.
        assert "left_hand=False" in response["message"]


def test_frame_with_undecodable_data_returns_a_clear_decode_error():
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "frame", "timestamp": 1, "data": "AAAA"})
        response = ws.receive_json()
        assert response["type"] == "error"
        assert "decode" in response["message"].lower()


def test_invalid_json_returns_error_and_keeps_connection_open():
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_text("this is not json")
        response = ws.receive_json()
        assert response["type"] == "error"
        assert "json" in response["message"].lower()

        # connection must still be usable afterwards
        ws.send_json({"type": "frame", "timestamp": 1, "data": _blank_frame_data_url()})
        ws.receive_json()  # landmarks_status
        follow_up = ws.receive_json()
        assert follow_up["type"] == "error"


def test_unknown_message_type_returns_error():
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "not_a_real_type", "foo": "bar"})
        response = ws.receive_json()
        assert response["type"] == "error"
        assert "unknown" in response["message"].lower()


def test_missing_required_field_returns_error():
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "frame", "timestamp": 1})  # missing 'data'
        response = ws.receive_json()
        assert response["type"] == "error"
        assert "invalid" in response["message"].lower()


def test_second_frame_within_min_interval_is_rate_limited():
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        frame_data = _blank_frame_data_url()
        ws.send_json({"type": "frame", "timestamp": 1, "data": frame_data})
        ws.receive_json()  # landmarks_status
        first = ws.receive_json()
        assert first["type"] == "error"  # ML-not-ready error

        # Sent immediately after -- far faster than any configured max_fps allows.
        ws.send_json({"type": "frame", "timestamp": 2, "data": frame_data})
        second = ws.receive_json()
        assert second["type"] == "error"
        assert "rate" in second["message"].lower()


def test_oversized_message_closes_the_connection():
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        oversized_payload = "A" * 2_200_000  # exceeds default 2 MB WS_MAX_MESSAGE_SIZE_BYTES
        ws.send_json({"type": "frame", "timestamp": 1, "data": oversized_payload})
        response = ws.receive_json()
        assert response["type"] == "error"
        assert "exceeds" in response["message"].lower()

        try:
            ws.receive_json()
            assert False, "expected the server to have closed the connection"
        except WebSocketDisconnect:
            pass
