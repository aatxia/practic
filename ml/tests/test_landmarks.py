from pathlib import Path

import numpy as np
import pytest

from ml.preprocessing.landmarks import (
    FeatureToggles,
    LandmarkExtractor,
    ModelNotFoundError,
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "mediapipe"

hands_model_available = (MODEL_DIR / "hand_landmarker.task").exists()
face_model_available = (MODEL_DIR / "face_landmarker.task").exists()
pose_model_available = (MODEL_DIR / "pose_landmarker.task").exists()

skip_reason = (
    "MediaPipe model file not present -- run scripts/download_mediapipe_models.sh first"
)


def _blank_frame(width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (120, 140, 160)
    return frame


def test_missing_model_raises_a_clear_error_not_a_silent_failure(tmp_path):
    """Core 'NO FAKE AI' rule applied to preprocessing: a missing model must
    fail loudly with actionable instructions, never pretend to detect."""
    with pytest.raises(ModelNotFoundError, match="download_mediapipe_models.sh"):
        LandmarkExtractor(model_dir=tmp_path, features=FeatureToggles(hands=True))


@pytest.mark.skipif(not hands_model_available, reason=skip_reason)
def test_hand_detector_returns_none_on_a_frame_with_no_hand_not_a_fake_detection():
    with LandmarkExtractor(
        model_dir=MODEL_DIR, features=FeatureToggles(hands=True, pose=False, face=False)
    ) as extractor:
        result = extractor.extract(_blank_frame())
        assert result.left_hand is None
        assert result.right_hand is None


@pytest.mark.skipif(not face_model_available, reason=skip_reason)
def test_face_detector_returns_none_on_a_frame_with_no_face():
    with LandmarkExtractor(
        model_dir=MODEL_DIR, features=FeatureToggles(hands=False, pose=False, face=True)
    ) as extractor:
        result = extractor.extract(_blank_frame())
        assert result.face is None


@pytest.mark.skipif(not pose_model_available, reason=skip_reason)
def test_pose_detector_returns_none_on_a_frame_with_no_person():
    with LandmarkExtractor(
        model_dir=MODEL_DIR, features=FeatureToggles(hands=False, pose=True, face=False)
    ) as extractor:
        result = extractor.extract(_blank_frame())
        assert result.pose is None


@pytest.mark.skipif(not hands_model_available, reason=skip_reason)
def test_only_enabled_feature_toggles_load_their_detector():
    """Section 9: a hands-only experiment must not pay the cost of loading
    the face/pose models."""
    with LandmarkExtractor(
        model_dir=MODEL_DIR, features=FeatureToggles(hands=True, pose=False, face=False)
    ) as extractor:
        assert extractor._hand_detector is not None
        assert extractor._pose_detector is None
        assert extractor._face_detector is None
