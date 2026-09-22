"""
Face feature vector: a *selected subset* of MediaPipe FaceMesh's 478
landmarks (per section 7: "face: selected facial landmarks"), chosen for
their relevance to UKSL non-manual grammar markers (section 19) -- eyebrow
raises (questions), eye narrowing (negation/emphasis), and mouth shape --
rather than the full dense mesh, which is mostly redundant for this task.

Selected landmarks (MediaPipe FaceMesh topology, canonical indices):
    Left eyebrow:  70, 63, 105, 66, 107
    Right eyebrow: 336, 296, 334, 293, 300
    Left eye:      33 (outer), 133 (inner), 159 (top), 145 (bottom)
    Right eye:     263 (outer), 362 (inner), 386 (top), 374 (bottom)
    Mouth:         61 (left corner), 291 (right corner), 13 (upper lip
                   center), 14 (lower lip center), 0 (philtrum), 17 (chin top)
"""
from __future__ import annotations

import numpy as np

SELECTED_FACE_LANDMARKS = (
    # left eyebrow
    70, 63, 105, 66, 107,
    # right eyebrow
    336, 296, 334, 293, 300,
    # left eye
    33, 133, 159, 145,
    # right eye
    263, 362, 386, 374,
    # mouth
    61, 291, 13, 14, 0, 17,
)  # fmt: skip

FACE_FEATURE_SIZE = len(SELECTED_FACE_LANDMARKS) * 3


def face_feature_vector(face: np.ndarray) -> np.ndarray:
    """
    Args:
        face: (478, 3) normalized face landmarks (zero-filled if not detected).

    Returns:
        (72,) float32 vector: the 24 selected landmarks, flattened.
    """
    selected = face[list(SELECTED_FACE_LANDMARKS)]
    return selected.ravel().astype(np.float32)
