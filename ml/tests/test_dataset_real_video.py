"""
Real (non-demo) video loading (ml/datasets/dataset.py) -- decodes an actual
video file through the real MediaPipe pipeline, same one the live WebSocket
handler runs per frame. Uses a real (if content-free) .mp4 written with
cv2.VideoWriter, not a mock -- the blank frames won't have a detected hand,
same honesty convention as backend/tests' blank-frame tests: nothing here
claims a hand was seen that wasn't.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from ml.datasets.annotation import SampleAnnotation
from ml.datasets.dataset import load_feature_sequence, load_real_feature_sequence
from ml.features.feature_vector import FeatureConfig, feature_vector_size
from ml.training.dataset import SignSequenceDataset


def _write_test_video(path: Path, num_frames: int, fps: float, width: int = 64, height: int = 48) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=(i * 4) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_sample(clip_path: str, gloss: str = "WE") -> SampleAnnotation:
    return SampleAnnotation(
        sample_id=f"aatxia_{gloss}_00",
        clip_path=clip_path,
        signer_id="aatxia",
        gloss=gloss,
        start_frame=0,
        end_frame=90,
        fps=30.0,
        source="real_webcam",
    )


def test_load_real_feature_sequence_returns_the_right_shape(tmp_path):
    clip_path = tmp_path / "WE_00.mp4"
    _write_test_video(clip_path, num_frames=30, fps=30.0)
    sample = _make_sample("WE_00.mp4")
    feature_config = FeatureConfig(hands=True, pose=True, face=True)

    sequence = load_real_feature_sequence(sample, tmp_path, feature_config, target_fps=30.0)

    assert sequence.shape[1] == feature_vector_size(feature_config)
    assert sequence.dtype == np.float32
    # No real hand in these blank frames -- honestly zero-filled, not fabricated.
    assert sequence.shape[0] > 0


def test_load_real_feature_sequence_resamples_to_target_fps(tmp_path):
    clip_path = tmp_path / "WE_00.mp4"
    _write_test_video(clip_path, num_frames=30, fps=30.0)
    sample = _make_sample("WE_00.mp4")
    feature_config = FeatureConfig(hands=False, pose=False, face=False)

    full_rate = load_real_feature_sequence(sample, tmp_path, feature_config, target_fps=30.0)
    half_rate = load_real_feature_sequence(sample, tmp_path, feature_config, target_fps=15.0)

    # 30fps source resampled to 15fps -- roughly half the frames (same
    # stride logic as ml/preprocessing/video_reader.py's read_video_frames).
    assert half_rate.shape[0] < full_rate.shape[0]
    assert half_rate.shape[0] == pytest.approx(full_rate.shape[0] / 2, abs=2)


def test_load_feature_sequence_requires_feature_config_and_fps_for_real_samples(tmp_path):
    clip_path = tmp_path / "WE_00.mp4"
    _write_test_video(clip_path, num_frames=5, fps=30.0)
    sample = _make_sample("WE_00.mp4")

    with pytest.raises(ValueError, match="requires feature_config and target_fps"):
        load_feature_sequence(sample, tmp_path)


def test_sign_sequence_dataset_caches_real_extraction_across_repeated_access(tmp_path, monkeypatch):
    clip_path = tmp_path / "WE_00.mp4"
    _write_test_video(clip_path, num_frames=10, fps=30.0)
    sample = _make_sample("WE_00.mp4")

    call_count = 0
    real_loader = load_real_feature_sequence

    def counting_loader(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_loader(*args, **kwargs)

    # load_feature_sequence calls load_real_feature_sequence as a plain
    # module-global lookup, so patching the module attribute redirects it
    # too -- lets us count real calls without mocking the whole pipeline.
    monkeypatch.setattr("ml.datasets.dataset.load_real_feature_sequence", counting_loader)

    dataset = SignSequenceDataset(
        [sample],
        tmp_path,
        {"WE": 0},
        sequence_length=8,
        feature_config=FeatureConfig(hands=False, pose=False, face=False),
        target_fps=30.0,
    )

    dataset[0]
    dataset[0]
    dataset[0]

    assert call_count == 1
