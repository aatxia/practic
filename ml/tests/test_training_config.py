import pytest

from ml.training.config import load_training_config


def test_loads_repo_config():
    config = load_training_config("../configs/model.yaml")

    assert config.model_type == "lstm"
    assert config.sequence_length == 32
    assert config.feature_config.hands is True
    assert config.feature_config.pose is True
    assert config.feature_config.face is True
    assert config.hidden_size > 0
    assert config.num_layers > 0
    assert config.epochs > 0
    assert config.batch_size > 0
    assert config.learning_rate > 0
    assert config.camera_fps > 0


def test_rejects_unimplemented_model_type(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        "model:\n  type: transformer\n  sequence_length: 32\n"
        "features:\n  hands: true\n  pose: true\n  face: true\n",
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        load_training_config(path)


def test_uses_defaults_when_training_section_missing(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        "camera:\n  fps: 12\n"
        "model:\n  type: lstm\n  sequence_length: 16\n"
        "features:\n  hands: true\n  pose: false\n  face: false\n",
        encoding="utf-8",
    )

    config = load_training_config(path)

    assert config.sequence_length == 16
    assert config.hidden_size == 128
    assert config.epochs == 20
    assert config.camera_fps == 12
