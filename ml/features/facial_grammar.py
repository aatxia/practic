"""
facial_grammar — non-manual grammar markers (section 19), Phase 17.

Sign languages carry real grammatical meaning on the face, independent of
the hands: cross-linguistically (УЖМ included), a yes/no question is marked
by raised eyebrows and a wh-question by furrowed/lowered eyebrows, with no
separate manual sign for "?" at all. Phase 12/14's gloss<->text engines
compose punctuation from gloss tokens alone, so without this module a
recognized question would come out looking exactly like a statement.

This module measures one real, well-documented signal -- eyebrow height
relative to the eyes -- from the same MediaPipe FaceMesh landmarks
ml/features/face.py already selects (see its docstring for the canonical
indices). It is a deterministic geometric heuristic, not a trained
classifier: no annotated facial-grammar dataset exists to train or
validate one against (Phase 8 still hasn't found a real УЖМ dataset at
all). The thresholds below are a reasonable, documented starting point,
not empirically calibrated -- tune WS_FACIAL_RAISED_RATIO /
WS_FACIAL_FURROWED_RATIO (.env) once real signing video is available to
validate against.

Because eyebrow-eye proportions vary by face and camera framing, this
calibrates per-connection against the signer's own neutral face (the first
`calibration_frames` frames with a detected face) rather than using one
fixed threshold for everyone -- classification only starts once calibrated;
before that, honestly reports NONE rather than guessing off zero baseline.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

# MediaPipe FaceMesh canonical indices -- the middle landmark of each
# eyebrow group and the top landmark of each eye, from ml/features/face.py's
# SELECTED_FACE_LANDMARKS (70,63,105,66,107 / 336,296,334,293,300 / ...).
LEFT_EYEBROW_CENTER = 105
RIGHT_EYEBROW_CENTER = 334
LEFT_EYE_TOP = 159
RIGHT_EYE_TOP = 386

_EPS = 1e-6


class FacialGrammarMarker(str, Enum):
    NONE = "NONE"
    EYEBROWS_RAISED = "EYEBROWS_RAISED"  # yes/no question marker
    EYEBROWS_FURROWED = "EYEBROWS_FURROWED"  # wh-question marker


def eyebrow_eye_gap(face: np.ndarray) -> float:
    """Average vertical gap between eyebrow-center and eye-top landmarks on
    a normalize_face()-normalized (478, 3) array (ml/preprocessing/
    normalization.py). Image y increases downward, so a larger gap means
    the eyebrows sit farther above the eyes (raised); a smaller/negative
    gap means they've dropped toward the eyes (furrowed)."""
    left_gap = float(face[LEFT_EYE_TOP, 1] - face[LEFT_EYEBROW_CENTER, 1])
    right_gap = float(face[RIGHT_EYE_TOP, 1] - face[RIGHT_EYEBROW_CENTER, 1])
    return (left_gap + right_gap) / 2.0


class BaselineCalibrator:
    """Per-signer eyebrow-eye gap baseline + classifier. Feed it every
    frame's normalized face landmarks via update(); it accumulates a
    neutral-face baseline over the first `calibration_frames` calls, then
    classifies every call after that relative to the ratio thresholds."""

    def __init__(
        self,
        calibration_frames: int = 30,
        raised_ratio: float = 0.25,
        furrowed_ratio: float = 0.25,
    ) -> None:
        if calibration_frames < 1:
            raise ValueError("calibration_frames must be >= 1")
        if raised_ratio <= 0 or furrowed_ratio <= 0:
            raise ValueError("raised_ratio and furrowed_ratio must be > 0")
        self._calibration_frames = calibration_frames
        self._raised_ratio = raised_ratio
        self._furrowed_ratio = furrowed_ratio
        self._samples: list[float] = []
        self._baseline: float | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._baseline is not None

    @property
    def baseline(self) -> float | None:
        return self._baseline

    def update(self, face: np.ndarray) -> FacialGrammarMarker:
        gap = eyebrow_eye_gap(face)

        if self._baseline is None:
            self._samples.append(gap)
            if len(self._samples) >= self._calibration_frames:
                self._baseline = sum(self._samples) / len(self._samples)
            return FacialGrammarMarker.NONE

        scale = self._baseline if abs(self._baseline) > _EPS else _EPS
        deviation_ratio = (gap - self._baseline) / abs(scale)
        if deviation_ratio >= self._raised_ratio:
            return FacialGrammarMarker.EYEBROWS_RAISED
        if deviation_ratio <= -self._furrowed_ratio:
            return FacialGrammarMarker.EYEBROWS_FURROWED
        return FacialGrammarMarker.NONE

    def reset(self) -> None:
        self._samples = []
        self._baseline = None
