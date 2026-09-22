import numpy as np
import torch

from ml.fingerspelling.model import INPUT_SIZE
from ml.fingerspelling.train import _top_confusions
from ml.fingerspelling.train import main as train_main


def _write_fake_features(path, rng, per_class=40, num_classes=4):
    """Two well-separated clusters per class (in different feature
    dimensions) so a small MLP can actually learn a real class boundary --
    this is checking the training loop plumbing, not real-world accuracy,
    same spirit as ml/tests/test_train.py's tiny checkpoint run."""
    classes = [chr(ord("А") + i) for i in range(num_classes)]
    features = []
    labels = []
    for i, label in enumerate(classes):
        base = np.zeros(INPUT_SIZE, dtype=np.float32)
        base[i] = 5.0
        samples = base + rng.normal(scale=0.1, size=(per_class, INPUT_SIZE)).astype(np.float32)
        features.append(samples)
        labels.extend([label] * per_class)
    np.savez_compressed(path, features=np.concatenate(features), labels=np.array(labels, dtype="<U8"))


def test_train_main_produces_a_checkpoint_and_report_with_reasonable_accuracy(tmp_path):
    rng = np.random.default_rng(0)
    features_path = tmp_path / "features.npz"
    _write_fake_features(features_path, rng)

    output_path = tmp_path / "checkpoints" / "latest.pt"
    train_main(
        [
            "--features",
            str(features_path),
            "--output",
            str(output_path),
            "--epochs",
            "50",
        ]
    )

    assert output_path.exists()
    report_path = output_path.with_suffix(".report.json")
    assert report_path.exists()

    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    # Trivially separable clusters -- the trained model should do well,
    # confirming the train/eval loop itself is correct (real accuracy on
    # the actual USL_alphabet_train data is reported separately and is NOT
    # expected to look like this -- see PROJECT_STATUS.md).
    assert report["test_accuracy"] > 0.8
    # Every per-class entry always carries a confused_with list (empty
    # when the class made no test errors) -- a real diagnostic surfaced
    # by a live run where 89% overall accuracy hid a class the model had
    # essentially never learned (Щ at 0%, mistaken entirely for Ш).
    for entry in report["per_class"].values():
        assert "confused_with" in entry


def test_top_confusions_names_the_exact_wrong_class_a_mistaken_prediction_lands_on():
    index_to_label = {0: "Ш", 1: "Щ", 2: "Ц"}
    # Щ (true=1) mistaken for Ш (pred=0) three times, once for Ц (pred=2);
    # Ц (true=2) always predicted correctly; Ш (true=0) always correct too.
    labels = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2])
    predicted = torch.tensor([0, 0, 0, 0, 0, 2, 2, 2])

    confusions = _top_confusions(predicted, labels, index_to_label)

    assert confusions["Щ"] == [("Ш", 3), ("Ц", 1)]
    assert "Ц" not in confusions  # never mispredicted -- nothing to report
    assert "Ш" not in confusions


def test_top_confusions_is_empty_when_every_prediction_is_correct():
    index_to_label = {0: "А", 1: "Б"}
    labels = torch.tensor([0, 1, 0, 1])
    predicted = torch.tensor([0, 1, 0, 1])

    assert _top_confusions(predicted, labels, index_to_label) == {}
