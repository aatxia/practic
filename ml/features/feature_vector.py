"""
Combined feature vector builder — assembles the per-frame vector the
temporal model (Phase 9) will consume, from whichever modalities are
enabled in FeatureConfig. This is the single place that implements section 9's
experimental requirement: hands-only / hands+pose / hands+pose+face must all
be selectable via config, without touching model code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.features.face import FACE_FEATURE_SIZE, face_feature_vector
from ml.features.hands import HAND_FEATURE_SIZE, hand_feature_vector
from ml.features.pose import POSE_FEATURE_SIZE, pose_feature_vector
from ml.preprocessing.normalization import NormalizedFrame


@dataclass(frozen=True)
class FeatureConfig:
    hands: bool = True
    pose: bool = True
    face: bool = True


def feature_vector_size(config: FeatureConfig) -> int:
    """Total vector length for a given config -- lets the model/training code
    size its input layer without hardcoding a number that config changes invalidate."""
    size = 0
    if config.hands:
        size += HAND_FEATURE_SIZE
    if config.pose:
        size += POSE_FEATURE_SIZE
    if config.face:
        size += FACE_FEATURE_SIZE
    return size


def build_feature_vector(frame: NormalizedFrame, config: FeatureConfig) -> np.ndarray:
    """Concatenate the enabled modalities' feature vectors, in a fixed
    (hands, pose, face) order, for one frame."""
    parts: list[np.ndarray] = []
    if config.hands:
        parts.append(hand_feature_vector(frame.left_hand, frame.right_hand))
    if config.pose:
        parts.append(pose_feature_vector(frame.pose))
    if config.face:
        parts.append(face_feature_vector(frame.face))

    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)
