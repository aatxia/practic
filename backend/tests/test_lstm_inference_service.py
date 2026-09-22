from app.services.inference_service import SignPrediction
from app.services.lstm_inference_service import LSTMSignRecognizer

from ml.datasets.synthetic import generate_demo_dataset
from ml.training.train import main as train_main


def _train_tiny_checkpoint(tmp_path):
    generate_demo_dataset(tmp_path, sequence_length=8, samples_per_signer_gloss=4, seed=1)
    return train_main(
        [
            "--annotations",
            str(tmp_path / "annotations" / "demo_annotations.jsonl"),
            "--dataset-root",
            str(tmp_path),
            "--config",
            "../configs/model.yaml",
            "--experiment-name",
            "test_run",
            "--output-dir",
            str(tmp_path / "checkpoints"),
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


def test_predicts_from_a_full_window_and_tags_demo_text(tmp_path):
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    recognizer = LSTMSignRecognizer(checkpoint_path, device="cpu")

    assert recognizer.is_ready() is True
    assert recognizer.is_demo_mode is True

    feature_size = 222  # hands(126) + pose(24) + face(72), configs/model.yaml default
    window = [[0.0] * feature_size for _ in range(recognizer.sequence_length)]
    prediction = recognizer.predict(window)

    assert isinstance(prediction, SignPrediction)
    assert prediction.text.startswith("[DEMO] ")
    assert prediction.text == f"[DEMO] {prediction.sign}"
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.is_final is False


def test_predict_with_alternatives_matches_predict_and_returns_the_full_ranked_distribution(tmp_path):
    checkpoint_path = _train_tiny_checkpoint(tmp_path)
    recognizer = LSTMSignRecognizer(checkpoint_path, device="cpu")

    feature_size = 222
    window = [[0.0] * feature_size for _ in range(recognizer.sequence_length)]
    prediction, alternatives = recognizer.predict_with_alternatives(window, k=3)

    # Same top-1 predict() itself would return, from the exact same input.
    plain_prediction = recognizer.predict(window)
    assert prediction.sign == plain_prediction.sign
    assert prediction.confidence == plain_prediction.confidence
    assert prediction.text == plain_prediction.text

    # DEMO_GLOSSES has 5 classes (ml/datasets/synthetic.py) -- k=3 returns
    # exactly 3 of them, ranked, never all 5.
    assert len(alternatives) == 3
    assert alternatives[0].gloss == prediction.sign
    assert alternatives[0].confidence == prediction.confidence
    # A real ranked probability distribution (each a real softmax output,
    # summing to at most 1 across all classes), not a fabricated list.
    confidences = [a.confidence for a in alternatives]
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert sum(confidences) <= 1.0 + 1e-6
    assert confidences == sorted(confidences, reverse=True)
