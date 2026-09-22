import numpy as np
import pytest

from ml.features.facial_grammar import (
    LEFT_EYE_TOP,
    LEFT_EYEBROW_CENTER,
    RIGHT_EYE_TOP,
    RIGHT_EYEBROW_CENTER,
    BaselineCalibrator,
    FacialGrammarMarker,
    eyebrow_eye_gap,
)


def _face_with_gap(gap: float) -> np.ndarray:
    """A (478, 3) array where every eyebrow/eye landmark used by
    eyebrow_eye_gap is set so the gap comes out to exactly `gap`."""
    face = np.zeros((478, 3), dtype=np.float32)
    face[LEFT_EYEBROW_CENTER, 1] = 0.0
    face[RIGHT_EYEBROW_CENTER, 1] = 0.0
    face[LEFT_EYE_TOP, 1] = gap
    face[RIGHT_EYE_TOP, 1] = gap
    return face


def test_eyebrow_eye_gap_averages_left_and_right():
    face = np.zeros((478, 3), dtype=np.float32)
    face[LEFT_EYEBROW_CENTER, 1] = 0.0
    face[LEFT_EYE_TOP, 1] = 0.10
    face[RIGHT_EYEBROW_CENTER, 1] = 0.0
    face[RIGHT_EYE_TOP, 1] = 0.20

    assert eyebrow_eye_gap(face) == pytest.approx(0.15)


class TestBaselineCalibrator:
    def test_rejects_invalid_calibration_frames(self):
        with pytest.raises(ValueError, match="calibration_frames"):
            BaselineCalibrator(calibration_frames=0)

    def test_rejects_invalid_ratios(self):
        with pytest.raises(ValueError, match="ratio"):
            BaselineCalibrator(raised_ratio=0)
        with pytest.raises(ValueError, match="ratio"):
            BaselineCalibrator(furrowed_ratio=-0.1)

    def test_reports_none_and_stays_uncalibrated_before_enough_frames(self):
        calibrator = BaselineCalibrator(calibration_frames=5)
        for _ in range(4):
            marker = calibrator.update(_face_with_gap(0.10))
            assert marker == FacialGrammarMarker.NONE
            assert calibrator.is_calibrated is False

    def test_becomes_calibrated_after_enough_frames_and_records_the_mean(self):
        calibrator = BaselineCalibrator(calibration_frames=3)
        calibrator.update(_face_with_gap(0.08))
        calibrator.update(_face_with_gap(0.10))
        marker = calibrator.update(_face_with_gap(0.12))

        assert calibrator.is_calibrated is True
        assert calibrator.baseline == pytest.approx(0.10)
        # the calibrating frame itself isn't classified against the baseline
        assert marker == FacialGrammarMarker.NONE

    def test_classifies_raised_eyebrows_once_calibrated(self):
        calibrator = BaselineCalibrator(calibration_frames=2, raised_ratio=0.25)
        calibrator.update(_face_with_gap(0.10))
        calibrator.update(_face_with_gap(0.10))  # baseline = 0.10

        marker = calibrator.update(_face_with_gap(0.20))  # +100% >> 25%
        assert marker == FacialGrammarMarker.EYEBROWS_RAISED

    def test_classifies_furrowed_eyebrows_once_calibrated(self):
        calibrator = BaselineCalibrator(calibration_frames=2, furrowed_ratio=0.25)
        calibrator.update(_face_with_gap(0.10))
        calibrator.update(_face_with_gap(0.10))  # baseline = 0.10

        marker = calibrator.update(_face_with_gap(0.02))  # -80% << -25%
        assert marker == FacialGrammarMarker.EYEBROWS_FURROWED

    def test_small_deviation_within_thresholds_stays_none(self):
        calibrator = BaselineCalibrator(calibration_frames=2, raised_ratio=0.25, furrowed_ratio=0.25)
        calibrator.update(_face_with_gap(0.10))
        calibrator.update(_face_with_gap(0.10))  # baseline = 0.10

        marker = calibrator.update(_face_with_gap(0.105))  # +5%, well within threshold
        assert marker == FacialGrammarMarker.NONE

    def test_reset_forgets_the_baseline(self):
        calibrator = BaselineCalibrator(calibration_frames=2)
        calibrator.update(_face_with_gap(0.10))
        calibrator.update(_face_with_gap(0.10))
        assert calibrator.is_calibrated is True

        calibrator.reset()
        assert calibrator.is_calibrated is False
        assert calibrator.baseline is None
