import numpy as np

from ml.features.face import FACE_FEATURE_SIZE, face_feature_vector
from ml.features.feature_vector import (
    FeatureConfig,
    build_feature_vector,
    feature_vector_size,
)
from ml.features.hands import HAND_FEATURE_SIZE, hand_feature_vector
from ml.features.pose import POSE_FEATURE_SIZE, pose_feature_vector
from ml.preprocessing.normalization import NormalizedFrame


def test_hand_feature_vector_shape_and_order():
    left = np.full((21, 3), 1.0, dtype=np.float32)
    right = np.full((21, 3), 2.0, dtype=np.float32)
    vector = hand_feature_vector(left, right)

    assert vector.shape == (HAND_FEATURE_SIZE,) == (126,)
    assert np.all(vector[:63] == 1.0)  # left hand first
    assert np.all(vector[63:] == 2.0)  # then right hand


def test_pose_feature_vector_selects_only_upper_body_landmarks():
    pose = np.arange(33 * 3, dtype=np.float32).reshape(33, 3)
    vector = pose_feature_vector(pose)

    assert vector.shape == (POSE_FEATURE_SIZE,) == (24,)
    # first selected landmark is index 11 -> values [33, 34, 35]
    np.testing.assert_array_equal(vector[:3], [33.0, 34.0, 35.0])


def test_face_feature_vector_selects_subset_not_full_mesh():
    face = np.arange(478 * 3, dtype=np.float32).reshape(478, 3)
    vector = face_feature_vector(face)

    assert vector.shape == (FACE_FEATURE_SIZE,) == (72,)
    assert FACE_FEATURE_SIZE < 478 * 3  # a genuine subset, not the whole mesh


def _blank_normalized_frame() -> NormalizedFrame:
    return NormalizedFrame(
        left_hand=np.zeros((21, 3), dtype=np.float32),
        right_hand=np.zeros((21, 3), dtype=np.float32),
        pose=np.zeros((33, 3), dtype=np.float32),
        face=np.zeros((478, 3), dtype=np.float32),
        present={"left_hand": False, "right_hand": False, "pose": False, "face": False},
    )


def test_feature_vector_size_changes_with_config():
    """Section 9: hands-only vs hands+pose vs hands+pose+face must produce
    different (and correctly sized) feature vectors purely via config."""
    hands_only = feature_vector_size(FeatureConfig(hands=True, pose=False, face=False))
    hands_and_pose = feature_vector_size(FeatureConfig(hands=True, pose=True, face=False))
    all_modalities = feature_vector_size(FeatureConfig(hands=True, pose=True, face=True))

    assert hands_only == HAND_FEATURE_SIZE
    assert hands_and_pose == HAND_FEATURE_SIZE + POSE_FEATURE_SIZE
    assert all_modalities == HAND_FEATURE_SIZE + POSE_FEATURE_SIZE + FACE_FEATURE_SIZE
    assert hands_only < hands_and_pose < all_modalities


def test_build_feature_vector_matches_declared_size():
    frame = _blank_normalized_frame()
    config = FeatureConfig(hands=True, pose=True, face=True)

    vector = build_feature_vector(frame, config)

    assert vector.shape == (feature_vector_size(config),)


def test_build_feature_vector_respects_disabled_modalities():
    frame = _blank_normalized_frame()
    config = FeatureConfig(hands=True, pose=False, face=False)

    vector = build_feature_vector(frame, config)

    assert vector.shape == (HAND_FEATURE_SIZE,)


def test_build_feature_vector_empty_config_returns_empty_vector():
    frame = _blank_normalized_frame()
    config = FeatureConfig(hands=False, pose=False, face=False)

    vector = build_feature_vector(frame, config)

    assert vector.shape == (0,)
