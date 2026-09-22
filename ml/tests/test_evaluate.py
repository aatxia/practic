import torch

from ml.datasets.synthetic import generate_demo_dataset
from ml.training.evaluate import main
from ml.training.train import main as train_main


def test_reports_per_class_accuracy_for_every_class_on_val_and_test(tmp_path, capsys):
    generate_demo_dataset(tmp_path, sequence_length=8, samples_per_signer_gloss=3, seed=1)
    annotations_path = tmp_path / "annotations" / "demo_annotations.jsonl"
    checkpoint_path = train_main(
        [
            "--annotations",
            str(annotations_path),
            "--dataset-root",
            str(tmp_path),
            "--config",
            "../configs/model.yaml",
            "--experiment-name",
            "eval_test",
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

    main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--annotations",
            str(annotations_path),
            "--dataset-root",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )

    out = capsys.readouterr().out
    assert "[val]" in out
    assert "[test]" in out
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    for label in checkpoint["label_to_index"]:
        assert label in out
