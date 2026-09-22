"""
motion_gate — guards against a real, confirmed gap in the word-level
classifier: it was only ever trained on the 7 positive word classes
(ml/training/), with no "no gesture" / idle / background class. A live
diagnostic (streaming a genuinely still, duplicated frame through the
real backend/websocket/handler.py pipeline) showed a held-still hand gets
confidently (>0.9) misclassified as one of the 7 known words -- since the
softmax has nowhere else to put "nothing is happening", it always picks
its nearest known class. That is a real "NO FAKE AI" violation in
practice, not a misconfiguration: it looks exactly like a genuine (if
occasionally wrong) recognition, with nothing to tell them apart from
confidence alone.

Retraining with a real negative/background class needs new data
collection, out of scope here. Until then, the classifier is simply
never run on a window with no real hand movement -- gated by how much
the hand-landmark portion of the buffered feature window actually moved,
frame to frame.

Thresholded empirically: a duplicated/held-still frame measures exactly
0.0; every real recorded word clip (data/real/raw/, see PROJECT_STATUS.md)
measures >=1.0 across every 32-frame window sampled from it. 0.4 leaves a
wide margin on both sides for real camera/landmark jitter during an
actually-still hand.
"""
from __future__ import annotations

import numpy as np


def mean_hand_motion(feature_window: list[list[float]], hand_feature_size: int) -> float:
    """Average frame-to-frame L2 distance across `feature_window`,
    restricted to the hand-landmark portion of each feature vector (always
    first -- see ml/features/feature_vector.py's fixed hands/pose/face
    concatenation order). Pose/face motion (head turning, blinking) isn't
    hand motion and shouldn't count as "a sign is happening" the way it
    would bias a whole-vector distance.

    Returns 0.0 for a window with fewer than 2 frames -- no motion to
    measure yet, an honest "not enough data" rather than a guess.
    """
    if len(feature_window) < 2:
        return 0.0
    arr = np.asarray(feature_window, dtype=np.float32)[:, :hand_feature_size]
    diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    return float(diffs.mean())
