"""
Pose feature vector: uses a *selected subset* of MediaPipe's 33 pose
landmarks, not all of them (per master-prompt section 7: "pose: selected
body landmarks"). Signing is an upper-body activity -- legs/feet carry no
linguistic information and would only add noise/dimensionality.

Selected landmarks (MediaPipe Pose topology):
    11, 12 -- left/right shoulder
    13, 14 -- left/right elbow
    15, 16 -- left/right wrist
    23, 24 -- left/right hip (gives the model torso lean/orientation)
"""
from __future__ import annotations

import numpy as np

SELECTED_POSE_LANDMARKS = (11, 12, 13, 14, 15, 16, 23, 24)

POSE_FEATURE_SIZE = len(SELECTED_POSE_LANDMARKS) * 3


def pose_feature_vector(pose: np.ndarray) -> np.ndarray:
    """
    Args:
        pose: (33, 3) normalized pose landmarks (zero-filled if not detected).

    Returns:
        (24,) float32 vector: the 8 selected landmarks, flattened.
    """
    selected = pose[list(SELECTED_POSE_LANDMARKS)]
    return selected.ravel().astype(np.float32)
