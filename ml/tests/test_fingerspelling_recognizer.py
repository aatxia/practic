import numpy as np
import pytest

from ml.fingerspelling.model import INPUT_SIZE
from ml.fingerspelling.recognizer import (
    CheckpointNotFoundError,
    FingerspellingRecognizer,
)
from ml.fingerspelling.train import main as train_main


def _train_tiny_checkpoint(tmp_path):
    """Trains a minimal real checkpoint (two well-separated classes) and
    returns its path -- same spirit as ml/tests/test_recognizer.py's tiny
    LSTM checkpoint, just for a single-frame classifier instead."""
    rng = np.random.default_rng(0)
    classes = ["А", "Б"]
    features = []
    labels = []
    for i, label in enumerate(classes):
        base = np.zeros(INPUT_SIZE, dtype=np.float32)
        base[i] = 5.0
        samples = base + rng.normal(scale=0.1, size=(30, INPUT_SIZE)).astype(np.float32)
        features.append(samples)
        labels.extend([label] * 30)

    features_path = tmp_path / "features.npz"
    np.savez_compressed(features_path, features=np.concatenate(features), labels=np.array(labels, dtype="<U8"))

    output_path = tmp_path / "checkpoints" / "latest.pt"
    train_main(["--features", str(features_path), "--output", str(output_path), "--epochs", "30"])
    return output_path


def test_raises_clear_error_when_checkpoint_missing(tmp_path):
    with pytest.raises(CheckpointNotFoundError, match="No trained fingerspelling checkpoint"):
        FingerspellingRecognizer(tmp_path / "does_not_exist.pt")


def test_loads_checkpoint_and_predicts_a_real_trivially_separable_class(tmp_path):
    checkpoint_path = _train_tiny_checkpoint(tmp_path)

    recognizer = FingerspellingRecognizer(checkpoint_path, device="cpu")

    vector = [0.0] * INPUT_SIZE
    vector[0] = 5.0  # matches class "А"'s cluster center from the tiny training set above

    result = recognizer.predict(vector)

    assert result.letter == "А"
    assert 0.0 <= result.confidence <= 1.0


def test_predict_rejects_a_wrong_sized_vector(tmp_path):
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    recognizer = FingerspellingRecognizer(checkpoint_path, device="cpu")

    with pytest.raises(ValueError, match="63-dim"):
        recognizer.predict([0.0] * 10)


def test_predict_topk_returns_a_ranked_full_distribution(tmp_path):
    """Live fingerspelling diagnostics (websocket/handler.py's [FINGER
    DEBUG] log, scripts/collect_fingerspelling_live.py) need the full
    ranked distribution, not just predict()'s argmax."""
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    recognizer = FingerspellingRecognizer(checkpoint_path, device="cpu")

    vector = [0.0] * INPUT_SIZE
    vector[0] = 5.0  # matches class "А"'s cluster center

    top2 = recognizer.predict_topk(vector, k=2)

    assert len(top2) == 2
    assert top2[0].confidence >= top2[1].confidence
    assert top2[0].letter == recognizer.predict(vector).letter
    assert {r.letter for r in top2} == {"А", "Б"}


def test_predict_topk_rejects_a_wrong_sized_vector(tmp_path):
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    recognizer = FingerspellingRecognizer(checkpoint_path, device="cpu")

    with pytest.raises(ValueError, match="63-dim"):
        recognizer.predict_topk([0.0] * 10)
