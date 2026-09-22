import pytest

from ml.datasets.synthetic import generate_demo_dataset
from ml.inference.recognizer import CheckpointNotFoundError, SignRecognizer
from ml.training.train import main as train_main


def _train_tiny_checkpoint(tmp_path):
    """Trains a minimal checkpoint on the demo dataset and returns its path."""
    generate_demo_dataset(tmp_path, sequence_length=8, samples_per_signer_gloss=4, seed=1)
    annotations_path = tmp_path / "annotations" / "demo_annotations.jsonl"
    checkpoint_dir = tmp_path / "checkpoints"

    return train_main(
        [
            "--annotations",
            str(annotations_path),
            "--dataset-root",
            str(tmp_path),
            "--config",
            "../configs/model.yaml",
            "--experiment-name",
            "test_run",
            "--output-dir",
            str(checkpoint_dir),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--hidden-size",
            "8",
            "--num-layers",
            "1",
            "--device",
            "cpu",
        ]
    )


def test_raises_clear_error_when_checkpoint_missing(tmp_path):
    with pytest.raises(CheckpointNotFoundError, match="No trained checkpoint"):
        SignRecognizer(tmp_path / "does_not_exist.pt")


def test_loads_checkpoint_and_predicts(tmp_path):
    checkpoint_path = _train_tiny_checkpoint(tmp_path)

    recognizer = SignRecognizer(checkpoint_path, device="cpu")

    assert recognizer.is_demo_mode is True
    assert recognizer.source_tags == ["demo_synthetic"]
    assert recognizer.sequence_length == 32  # from configs/model.yaml, not the demo's own 8

    # A window of exactly sequence_length zero-vectors is a valid (if
    # meaningless) input -- just checking predict() runs end to end.
    feature_size = _feature_size(recognizer)
    dummy_sequence = [[0.0] * feature_size for _ in range(recognizer.sequence_length)]
    result = recognizer.predict(dummy_sequence)

    assert result.gloss in {"PRIVIT", "DYAKUYU", "TAK", "NI", "BUD_LASKA"}
    assert 0.0 <= result.confidence <= 1.0
    assert result.is_demo_mode is True


def _feature_size(recognizer: SignRecognizer) -> int:
    from ml.features.feature_vector import feature_vector_size

    return feature_vector_size(recognizer.feature_config)


def test_predict_rejects_wrong_length_input(tmp_path):
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    recognizer = SignRecognizer(checkpoint_path, device="cpu")
    feature_size = _feature_size(recognizer)

    too_short = [[0.0] * feature_size for _ in range(recognizer.sequence_length - 1)]

    with pytest.raises(ValueError, match="Expected exactly"):
        recognizer.predict(too_short)


def test_predict_topk_returns_a_ranked_full_distribution(tmp_path):
    """Live-vs-replay diagnostics (scripts/diagnose_live_capture.py) need
    the full ranked distribution, not just predict()'s argmax -- e.g. to
    see that the expected word was rank 2 at real but sub-threshold
    confidence, not simply absent."""
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    recognizer = SignRecognizer(checkpoint_path, device="cpu")
    feature_size = _feature_size(recognizer)
    dummy_sequence = [[0.0] * feature_size for _ in range(recognizer.sequence_length)]

    top3 = recognizer.predict_topk(dummy_sequence, k=3)

    assert len(top3) == 3
    # Ranked descending by confidence, and matches predict()'s own top-1.
    assert top3[0].confidence >= top3[1].confidence >= top3[2].confidence
    assert top3[0].gloss == recognizer.predict(dummy_sequence).gloss
    assert {r.gloss for r in top3} <= {"PRIVIT", "DYAKUYU", "TAK", "NI", "BUD_LASKA"}


def test_predict_topk_rejects_wrong_length_input(tmp_path):
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    recognizer = SignRecognizer(checkpoint_path, device="cpu")
    feature_size = _feature_size(recognizer)
    too_short = [[0.0] * feature_size for _ in range(recognizer.sequence_length - 1)]

    with pytest.raises(ValueError, match="Expected exactly"):
        recognizer.predict_topk(too_short)
