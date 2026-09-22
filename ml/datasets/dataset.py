"""
dataset — loads a `SampleAnnotation`'s feature-vector sequence off disk.

Two sources:
  - source="demo_synthetic": reads the precomputed `.npy` written by
    ml/datasets/synthetic.py.
  - anything else (real recorded video, e.g. source="real_webcam" from
    scripts/record_dataset.py): decodes clip_path through the exact same
    real pipeline the live WebSocket handler runs per frame
    (ml/preprocessing/landmarks.py + normalization.py +
    ml/features/feature_vector.py) -- so a real dataset and live inference
    can never silently drift apart on what a "feature vector" means.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.datasets.annotation import SampleAnnotation
from ml.datasets.synthetic import DEMO_SOURCE_TAG
from ml.features.feature_vector import FeatureConfig, build_feature_vector, feature_vector_size
from ml.preprocessing.landmarks import FeatureToggles, LandmarkExtractor
from ml.preprocessing.normalization import normalize_frame
from ml.preprocessing.video_reader import read_video_frames

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEDIAPIPE_MODELS_DIR = REPO_ROOT / "models" / "mediapipe"

# Process-wide (not per-Dataset) -- loading MediaPipe's model bundles is
# one-time work, same reasoning as backend/websocket/handler.py's own
# lazily-cached extractor.
_landmark_extractor: LandmarkExtractor | None = None


def _get_landmark_extractor(models_dir: Path | str) -> LandmarkExtractor:
    global _landmark_extractor
    if _landmark_extractor is None:
        _landmark_extractor = LandmarkExtractor(models_dir, FeatureToggles(hands=True, pose=True, face=True))
    return _landmark_extractor


def load_real_feature_sequence(
    sample: SampleAnnotation,
    dataset_root: Path | str,
    feature_config: FeatureConfig,
    target_fps: float,
    models_dir: Path | str = DEFAULT_MEDIAPIPE_MODELS_DIR,
) -> np.ndarray:
    """Decodes a real recorded clip -- resampled to `target_fps` (a real
    clip's own frame rate is whatever the camera captured at, e.g. 30fps;
    the live WebSocket handler runs at configs/model.yaml's camera.fps,
    currently 12 -- resampling makes the two comparable) -- through the
    real per-frame pipeline. No detection is ever fabricated: a frame with
    no hand/pose/face present contributes a zero-filled feature vector for
    that modality, same honesty convention as the live handler.
    """
    path = Path(dataset_root) / sample.clip_path
    extractor = _get_landmark_extractor(models_dir)

    vectors = [
        build_feature_vector(normalize_frame(extractor.extract(frame)), feature_config)
        for frame in read_video_frames(path, target_fps=target_fps)
    ]
    if not vectors:
        return np.zeros((0, feature_vector_size(feature_config)), dtype=np.float32)
    return np.stack(vectors)


def load_feature_sequence(
    sample: SampleAnnotation,
    dataset_root: Path | str,
    feature_config: FeatureConfig | None = None,
    target_fps: float | None = None,
) -> np.ndarray:
    """Returns the `(sequence_length, feature_size)` array for `sample`.
    `feature_config`/`target_fps` are required for any real (non-demo)
    sample -- there's no honest default frame rate to resample an
    arbitrary real clip to."""
    if sample.source == DEMO_SOURCE_TAG:
        path = Path(dataset_root) / sample.clip_path
        return np.load(path)

    if feature_config is None or target_fps is None:
        raise ValueError(
            f"Loading a real sample (source={sample.source!r}) requires feature_config and "
            "target_fps -- there's no honest default to fall back to."
        )
    return load_real_feature_sequence(sample, dataset_root, feature_config, target_fps)
