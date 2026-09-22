import numpy as np

from ml.preprocessing.landmarks import FrameLandmarks
from ml.preprocessing.normalization import (
    normalize_face,
    normalize_frame,
    normalize_hand,
    normalize_pose,
)


def _fake_hand(offset: np.ndarray, scale: float) -> np.ndarray:
    """A simple 21-point hand shape (not anatomically real, just distinct
    per-landmark so scale/offset invariance is actually exercised)."""
    base = np.array([[i * 0.01, i * 0.02, 0.0] for i in range(21)], dtype=np.float32)
    return base * scale + offset


def test_normalize_hand_is_invariant_to_translation():
    hand_a = _fake_hand(offset=np.array([0.0, 0.0, 0.0]), scale=1.0)
    hand_b = _fake_hand(offset=np.array([0.3, 0.4, 0.0]), scale=1.0)

    normalized_a = normalize_hand(hand_a)
    normalized_b = normalize_hand(hand_b)

    np.testing.assert_allclose(normalized_a, normalized_b, atol=1e-5)


def test_normalize_hand_is_invariant_to_scale():
    hand_a = _fake_hand(offset=np.array([0.0, 0.0, 0.0]), scale=1.0)
    hand_b = _fake_hand(offset=np.array([0.0, 0.0, 0.0]), scale=2.5)

    normalized_a = normalize_hand(hand_a)
    normalized_b = normalize_hand(hand_b)

    np.testing.assert_allclose(normalized_a, normalized_b, atol=1e-5)


def test_normalize_pose_centers_on_shoulder_midpoint():
    pose = np.zeros((33, 3), dtype=np.float32)
    pose[11] = [0.4, 0.3, 0.0]  # left shoulder
    pose[12] = [0.6, 0.3, 0.0]  # right shoulder

    normalized = normalize_pose(pose)
    shoulder_midpoint = (normalized[11] + normalized[12]) / 2
    np.testing.assert_allclose(shoulder_midpoint, [0.0, 0.0, 0.0], atol=1e-5)


def test_normalize_pose_is_invariant_to_camera_distance():
    """A person standing further from the camera has a smaller shoulder
    width in normalized image coordinates -- normalization must cancel that out."""
    close_pose = np.zeros((33, 3), dtype=np.float32)
    close_pose[11] = [0.3, 0.5, 0.0]
    close_pose[12] = [0.7, 0.5, 0.0]  # wide shoulders = close to camera

    far_pose = np.zeros((33, 3), dtype=np.float32)
    far_pose[11] = [0.45, 0.5, 0.0]
    far_pose[12] = [0.55, 0.5, 0.0]  # narrow shoulders = far from camera

    normalized_close = normalize_pose(close_pose)
    normalized_far = normalize_pose(far_pose)

    np.testing.assert_allclose(normalized_close[11], normalized_far[11], atol=1e-5)
    np.testing.assert_allclose(normalized_close[12], normalized_far[12], atol=1e-5)


def test_normalize_face_centers_on_centroid():
    face = np.random.default_rng(42).uniform(0.4, 0.6, size=(478, 3)).astype(np.float32)
    normalized = normalize_face(face)
    np.testing.assert_allclose(normalized.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-4)


def test_normalize_frame_fills_missing_modalities_with_zeros_and_flags_absence():
    frame = FrameLandmarks(left_hand=None, right_hand=None, pose=None, face=None)
    normalized = normalize_frame(frame)

    assert normalized.left_hand.shape == (21, 3)
    assert np.all(normalized.left_hand == 0.0)
    assert normalized.present == {
        "left_hand": False,
        "right_hand": False,
        "pose": False,
        "face": False,
    }


def test_normalize_frame_marks_present_modalities():
    frame = FrameLandmarks(
        left_hand=_fake_hand(np.zeros(3), 1.0),
        right_hand=None,
        pose=None,
        face=None,
    )
    normalized = normalize_frame(frame)
    assert normalized.present["left_hand"] is True
    assert normalized.present["right_hand"] is False
