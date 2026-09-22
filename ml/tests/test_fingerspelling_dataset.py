from pathlib import Path

import cv2
import numpy as np

from ml.fingerspelling.dataset import extract_features, scan_image_dataset
from ml.preprocessing.landmarks import FrameLandmarks


def _make_fake_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "USL_alphabet_train"
    (root / "А").mkdir(parents=True)
    (root / "Б").mkdir(parents=True)
    (root / "_excluded_non_ukrainian" / "Ъ").mkdir(parents=True)
    (root / "_excluded_unclear" / "TR_mixed_script").mkdir(parents=True)

    (root / "А" / "1.jpg").write_bytes(b"fake")
    (root / "Б" / "1.jpg").write_bytes(b"fake")
    (root / "Б" / "2.jpg").write_bytes(b"fake")
    (root / "_excluded_non_ukrainian" / "Ъ" / "1.jpg").write_bytes(b"fake")
    (root / "_excluded_unclear" / "TR_mixed_script" / "1.jpg").write_bytes(b"fake")
    return root


def test_scan_image_dataset_skips_underscore_prefixed_folders(tmp_path):
    root = _make_fake_dataset(tmp_path)

    samples = scan_image_dataset(root)

    labels = sorted(s.label for s in samples)
    assert labels == ["А", "Б", "Б"]


def _hand_at(x_offset: float) -> np.ndarray:
    base = np.array([[i * 0.01, i * 0.02, 0.0] for i in range(21)], dtype=np.float32)
    base[:, 0] += x_offset
    return base


class _RightHandOnlyExtractor:
    def __init__(self, hand: np.ndarray):
        self._hand = hand

    def extract(self, frame_bgr):
        return FrameLandmarks(right_hand=self._hand)


class _LeftHandOnlyExtractor:
    def __init__(self, hand: np.ndarray):
        self._hand = hand

    def extract(self, frame_bgr):
        return FrameLandmarks(left_hand=self._hand)


class _BothHandsExtractor:
    def extract(self, frame_bgr):
        return FrameLandmarks(left_hand=_hand_at(0.1), right_hand=_hand_at(0.2))


class _NoHandExtractor:
    def extract(self, frame_bgr):
        return FrameLandmarks()


def test_extract_features_skips_photos_with_no_or_both_hands(tmp_path, monkeypatch):
    root = _make_fake_dataset(tmp_path)
    samples = scan_image_dataset(root)

    monkeypatch.setattr("cv2.imdecode", lambda buf, flags: np.zeros((10, 10, 3), dtype=np.uint8))

    result = extract_features(samples, _BothHandsExtractor())
    assert result.features.shape == (0, 63)
    assert all(reason == "both hands detected, ambiguous for a single-handshape photo" for _, reason in result.skipped)

    result = extract_features(samples, _NoHandExtractor())
    assert result.features.shape == (0, 63)
    assert all(reason == "no hand detected" for _, reason in result.skipped)


def test_extract_features_mirrors_left_hand_to_match_right_hand_space(tmp_path, monkeypatch):
    root = _make_fake_dataset(tmp_path)
    samples = scan_image_dataset(root)[:1]

    monkeypatch.setattr("cv2.imdecode", lambda buf, flags: np.zeros((10, 10, 3), dtype=np.uint8))

    hand = _hand_at(0.1)
    right_result = extract_features(samples, _RightHandOnlyExtractor(hand))

    mirrored_hand = hand.copy()
    mirrored_hand[:, 0] *= -1.0
    left_result = extract_features(samples, _LeftHandOnlyExtractor(mirrored_hand))

    np.testing.assert_allclose(right_result.features, left_result.features, atol=1e-5)
    assert right_result.right_hand_count == 1
    assert left_result.left_hand_count == 1


def test_extract_features_skips_unreadable_image(tmp_path, monkeypatch):
    root = _make_fake_dataset(tmp_path)
    samples = scan_image_dataset(root)[:1]

    monkeypatch.setattr("cv2.imdecode", lambda buf, flags: None)

    result = extract_features(samples, _NoHandExtractor())

    assert result.features.shape == (0, 63)
    assert result.skipped[0][1] == "unreadable image file"


def test_extract_features_reads_a_real_image_at_a_cyrillic_path(tmp_path):
    # Regression test: cv2.imread(str(path)) silently fails (returns None)
    # for non-ASCII paths on Windows -- USL_alphabet_train's real class
    # directories are named with Cyrillic letters, so this isn't
    # hypothetical (it broke a real training run). A real (if tiny) PNG,
    # not fake bytes, so this actually exercises cv2.imdecode end to end.
    root = tmp_path / "USL_alphabet_train"
    class_dir = root / "Я"
    class_dir.mkdir(parents=True)
    image_path = class_dir / "1.jpg"

    real_image = np.full((10, 10, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", real_image)
    assert ok
    image_path.write_bytes(encoded.tobytes())

    samples = scan_image_dataset(root)
    result = extract_features(samples, _NoHandExtractor())

    # No hand in a flat gray image -- honestly skipped for that reason, not
    # "unreadable image file", which is exactly what proves the image WAS
    # successfully decoded despite the Cyrillic path.
    assert result.skipped == [(image_path, "no hand detected")]
