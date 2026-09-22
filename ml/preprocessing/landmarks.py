"""
landmarks — real MediaPipe Tasks integration for hands/pose/face detection.

Uses the MediaPipe Tasks API (HandLandmarker, PoseLandmarker, FaceLandmarker),
not the deprecated `mp.solutions` API which current mediapipe releases no
longer ship. Each detector needs its own pretrained `.task` model bundle
(Google-provided, Apache 2.0) -- these are a genuine external ML dependency,
not something this project trains. Run `scripts/download_mediapipe_models.sh`
to fetch them into `models/mediapipe/` before using this module.

Per section 9 (multimodal input / feature toggles), only the detectors
enabled in FeatureToggles are loaded -- e.g. a hands-only experiment never
pays the cost of loading the face model.

No detection is ever fabricated: if a modality isn't detected in a frame
(hand out of view, etc.), the corresponding field is simply None. Downstream
code (normalization.py) is responsible for deciding how to represent that
absence in a fixed-size feature vector -- it is never silently invented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
)

# Hand landmark count is fixed by the MediaPipe Hand model (21 knuckle points).
NUM_HAND_LANDMARKS = 21
# Pose landmark count is fixed by the MediaPipe Pose model (33 body points).
NUM_POSE_LANDMARKS = 33
# Face landmark count depends on the model variant; the standard Face
# Landmarker bundle used here outputs 478 points (incl. iris refinement).
NUM_FACE_LANDMARKS = 478


class ModelNotFoundError(FileNotFoundError):
    """Raised when a required MediaPipe .task model file is missing."""

    def __init__(self, model_path: Path) -> None:
        super().__init__(
            f"MediaPipe model not found at {model_path}. Run "
            "`scripts/download_mediapipe_models.sh` first (see README.md)."
        )


@dataclass(frozen=True)
class FeatureToggles:
    hands: bool = True
    pose: bool = True
    face: bool = True


@dataclass
class FrameLandmarks:
    """Raw (un-normalized) detection results for a single frame.

    Each array is (N, 3) in MediaPipe's normalized image coordinates
    (x, y in [0, 1] relative to frame width/height; z is relative depth).
    None means that modality was not detected in this frame -- not "zero
    confidence", an honest absence.
    """

    left_hand: np.ndarray | None = None
    right_hand: np.ndarray | None = None
    pose: np.ndarray | None = None
    face: np.ndarray | None = None


def _landmarks_to_array(landmark_list) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32)


class LandmarkExtractor:
    """Owns the MediaPipe detector instances for one session.

    Use as a context manager so native detector resources are always
    released deterministically:

        with LandmarkExtractor(model_dir, FeatureToggles(hands=True, pose=True, face=False)) as ex:
            result = ex.extract(frame_bgr)
    """

    def __init__(self, model_dir: str | Path, features: FeatureToggles) -> None:
        self._model_dir = Path(model_dir)
        self._features = features
        self._hand_detector: HandLandmarker | None = None
        self._pose_detector: PoseLandmarker | None = None
        self._face_detector: FaceLandmarker | None = None
        self._load_enabled_detectors()

    def _model_path(self, filename: str) -> Path:
        path = self._model_dir / filename
        if not path.exists():
            raise ModelNotFoundError(path)
        return path

    def _load_enabled_detectors(self) -> None:
        if self._features.hands:
            self._hand_detector = HandLandmarker.create_from_options(
                HandLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(self._model_path("hand_landmarker.task"))
                    ),
                    num_hands=2,
                )
            )
        if self._features.pose:
            self._pose_detector = PoseLandmarker.create_from_options(
                PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(self._model_path("pose_landmarker.task"))
                    ),
                    num_poses=1,
                )
            )
        if self._features.face:
            self._face_detector = FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(self._model_path("face_landmarker.task"))
                    ),
                    num_faces=1,
                )
            )

    def extract(self, frame_bgr: np.ndarray) -> FrameLandmarks:
        """Run every enabled detector on one BGR frame."""
        frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB, MediaPipe's expected order
        mp_image = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))

        left_hand = right_hand = pose = face = None

        if self._hand_detector is not None:
            hand_result = self._hand_detector.detect(mp_image)
            for hand_landmarks, handedness in zip(
                hand_result.hand_landmarks, hand_result.handedness, strict=False
            ):
                array = _landmarks_to_array(hand_landmarks)
                # MediaPipe's "handedness" is from the subject's own perspective
                # (mirrors a selfie camera), matching how a signer sees their hands.
                if handedness[0].category_name == "Left":
                    left_hand = array
                else:
                    right_hand = array

        if self._pose_detector is not None:
            pose_result = self._pose_detector.detect(mp_image)
            if pose_result.pose_landmarks:
                pose = _landmarks_to_array(pose_result.pose_landmarks[0])

        if self._face_detector is not None:
            face_result = self._face_detector.detect(mp_image)
            if face_result.face_landmarks:
                face = _landmarks_to_array(face_result.face_landmarks[0])

        return FrameLandmarks(left_hand=left_hand, right_hand=right_hand, pose=pose, face=face)

    def close(self) -> None:
        for detector in (self._hand_detector, self._pose_detector, self._face_detector):
            if detector is not None:
                detector.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
