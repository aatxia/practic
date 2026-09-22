"""
config — loads configs/model.yaml (section 6/31) into a `TrainingConfig`.
The same file that describes camera/model/features for the running backend's
.env is the single source of truth for what an experiment trains -- no
duplicated hyperparameters hardcoded in train.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ml.features.feature_vector import FeatureConfig

# Only "lstm" (ml/models/lstm.py) exists as of Phase 9. configs/model.yaml's
# comment lists transformer/video_jepa as future options (section 6) -- but
# claiming to support them here would be exactly the fake-readiness this
# project's "no fake AI" rule forbids, so an unimplemented type is a hard error.
SUPPORTED_MODEL_TYPES = ("lstm",)


@dataclass(frozen=True)
class TrainingConfig:
    model_type: str
    sequence_length: int
    feature_config: FeatureConfig
    hidden_size: int
    num_layers: int
    epochs: int
    batch_size: int
    learning_rate: float
    # A real recorded clip's own frame rate is whatever the camera captured
    # at (see ml/datasets/dataset.py's load_real_feature_sequence) -- this
    # is what it gets resampled TO, so a real dataset and the live
    # WebSocket handler (which also runs at this rate, per .env's
    # matching setting) produce comparable feature sequences.
    camera_fps: float


def load_training_config(path: Path | str) -> TrainingConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_type = raw["model"]["type"]
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise NotImplementedError(
            f"model.type={model_type!r} in {path} is not implemented yet (only "
            f"{SUPPORTED_MODEL_TYPES} exist as of Phase 9) -- set model.type to one "
            f"of {SUPPORTED_MODEL_TYPES} in the config, or implement the model first."
        )

    training = raw.get("training", {})

    return TrainingConfig(
        model_type=model_type,
        sequence_length=raw["model"]["sequence_length"],
        feature_config=FeatureConfig(
            hands=raw["features"]["hands"],
            pose=raw["features"]["pose"],
            face=raw["features"]["face"],
        ),
        hidden_size=training.get("hidden_size", 128),
        num_layers=training.get("num_layers", 2),
        epochs=training.get("epochs", 20),
        batch_size=training.get("batch_size", 16),
        learning_rate=training.get("learning_rate", 0.001),
        camera_fps=raw["camera"]["fps"],
    )
