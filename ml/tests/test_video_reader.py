import base64

import cv2
import numpy as np
import pytest

from ml.preprocessing.video_reader import FrameDecodeError, decode_base64_frame


def _make_jpeg_data_url(width: int = 64, height: int = 48) -> str:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)  # BGR
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def test_decode_base64_frame_with_data_url_prefix():
    data_url = _make_jpeg_data_url(64, 48)
    frame = decode_base64_frame(data_url)
    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8


def test_decode_base64_frame_without_prefix():
    data_url = _make_jpeg_data_url(32, 32)
    raw_b64 = data_url.split(",", 1)[1]
    frame = decode_base64_frame(raw_b64)
    assert frame.shape == (32, 32, 3)


def test_decode_base64_frame_rejects_invalid_base64():
    with pytest.raises(FrameDecodeError, match="not valid base64"):
        decode_base64_frame("not-base64-!!!")


def test_decode_base64_frame_rejects_non_image_bytes():
    garbage_b64 = base64.b64encode(b"this is not an image").decode("ascii")
    with pytest.raises(FrameDecodeError, match="not a valid image"):
        decode_base64_frame(garbage_b64)
