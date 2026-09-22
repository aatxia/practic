"""
Hand feature vector: flattens the normalized left+right hand landmarks
(21 points each, 3 coordinates) into one fixed-size vector for the temporal
model (Phase 9). Both hands are always included in the vector -- a hand not
detected in a frame is already zero-filled by normalize_frame(), so the
vector shape never changes frame to frame.
"""
from __future__ import annotations

import numpy as np

from ml.preprocessing.landmarks import NUM_HAND_LANDMARKS

# left hand (21 * 3) + right hand (21 * 3)
HAND_FEATURE_SIZE = NUM_HAND_LANDMARKS * 3 * 2


def hand_feature_vector(left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
    """
    Args:
        left_hand: (21, 3) normalized landmarks (zero-filled if not detected).
        right_hand: (21, 3) normalized landmarks (zero-filled if not detected).

    Returns:
        (126,) float32 vector: left hand flattened, then right hand flattened.
    """
    return np.concatenate([left_hand.ravel(), right_hand.ravel()]).astype(np.float32)
