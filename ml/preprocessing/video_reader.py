"""
video_reader — turns raw input (base64 JPEG from the WebSocket 'frame'
message, or a video file on disk) into numpy arrays that the rest of the
CV pipeline (landmarks.py) can consume.

All frames are returned as BGR uint8 arrays (OpenCV's native convention);
landmarks.py converts to RGB right before handing frames to MediaPipe.
"""
from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


class FrameDecodeError(ValueError):
    """Raised when a frame payload can't be decoded into an image."""


def decode_base64_frame(payload: str) -> np.ndarray:
    """Decode a base64-encoded JPEG (optionally prefixed with a data URL header,
    e.g. 'data:image/jpeg;base64,...' -- exactly what the frontend's
    canvas.toDataURL() produces) into a BGR uint8 numpy array.
    """
    if "," in payload and payload.strip().lower().startswith("data:"):
        payload = payload.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(payload, validate=True)
    except Exception as exc:  # binascii.Error and friends
        raise FrameDecodeError(f"Payload is not valid base64: {exc}") from exc

    buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise FrameDecodeError("Decoded bytes are not a valid image (JPEG/PNG)")

    return image


def read_video_frames(path: str | Path, target_fps: float | None = None) -> Iterator[np.ndarray]:
    """Yield BGR frames from a video file, optionally downsampled to `target_fps`.

    Used by the dataset pipeline (Phase 8) to turn raw UKSL video clips into
    frame sequences for landmark extraction -- the same decode path as the
    live WebSocket frames, so training and inference see identical input.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FrameDecodeError(f"OpenCV could not open video file: {path}")

    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        stride = 1
        if target_fps and source_fps > 0:
            stride = max(1, round(source_fps / target_fps))

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                yield frame
            frame_index += 1
    finally:
        capture.release()
