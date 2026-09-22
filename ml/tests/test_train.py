import cv2
import numpy as np
import torch

from ml.datasets.annotation import SampleAnnotation, write_annotations
from ml.datasets.synthetic import generate_demo_dataset
from ml.training.train import main


def test_end_to_end_training_produces_a_loadable_checkpoint(tmp_path):
    # sequence_length matches configs/model.yaml's model.sequence_length (32)
    # so the dataset lines up exactly with no padding/truncation surprises.
    generate_demo_dataset(tmp_path, sequence_length=32, samples_per_signer_gloss=4, seed=1)
    annotations_path = tmp_path / "annotations" / "demo_annotations.jsonl"
    checkpoint_dir = tmp_path / "checkpoints"

    checkpoint_path = main(
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
            "2",
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

    assert checkpoint_path == checkpoint_dir / "test_run" / "latest.pt"
    assert checkpoint_path.exists()

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["model_type"] == "lstm"
    assert checkpoint["demo_mode"] is True
    assert checkpoint["source_tags"] == ["demo_synthetic"]
    assert 0.0 <= checkpoint["val_accuracy"] <= 1.0
    assert checkpoint["sequence_length"] == 32
    assert checkpoint["split_strategy"] == "signer_independent"
    assert checkpoint["seed"] == 42
    assert checkpoint["camera_fps"] == 12
    assert set(checkpoint["label_to_index"].values()) == set(range(len(checkpoint["label_to_index"])))
    assert "model_state_dict" in checkpoint


def test_raises_clearly_when_val_split_would_be_empty(tmp_path):
    generate_demo_dataset(tmp_path, sequence_length=8, samples_per_signer_gloss=1, seed=1)
    annotations_path = tmp_path / "annotations" / "demo_annotations.jsonl"

    try:
        main(
            [
                "--annotations",
                str(annotations_path),
                "--dataset-root",
                str(tmp_path),
                "--config",
                "../configs/model.yaml",
                "--output-dir",
                str(tmp_path / "checkpoints"),
                "--train-ratio",
                "1.0",
                "--val-ratio",
                "0.0",
                "--test-ratio",
                "0.0",
                "--epochs",
                "1",
                "--device",
                "cpu",
            ]
        )
    except ValueError as exc:
        assert "val split is empty" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty val split")


def _write_real_clip(path, num_frames: int, fps: float = 30.0, width: int = 64, height: int = 48) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), fill_value=(i * 4) % 256, dtype=np.uint8))
    writer.release()


def test_falls_back_to_gloss_stratified_split_with_a_single_signer(tmp_path):
    # A single-signer real dataset (like scripts/record_dataset.py's output
    # before a second signer is recorded) has too few signers for
    # signer_independent_split (needs 3 for the default train/val/test
    # ratios) -- train.py must fall back to gloss_stratified_split rather
    # than crash, and must say so in the saved checkpoint.
    signer_id = "aatxia"
    raw_dir = tmp_path / "real" / "raw" / signer_id
    raw_dir.mkdir(parents=True)
    annotations = []
    for gloss in ("WE", "WANT"):
        for take in range(6):
            clip_path = raw_dir / f"{gloss}_{take:02d}.mp4"
            _write_real_clip(clip_path, num_frames=6)
            annotations.append(
                SampleAnnotation(
                    sample_id=f"{signer_id}_{gloss}_{take:02d}",
                    clip_path=clip_path.relative_to(tmp_path).as_posix(),
                    signer_id=signer_id,
                    gloss=gloss,
                    start_frame=0,
                    end_frame=6,
                    fps=30.0,
                    source="real_webcam",
                )
            )
    annotations_path = tmp_path / "annotations" / "aatxia_annotations.jsonl"
    write_annotations(annotations_path, annotations)

    checkpoint_path = main(
        [
            "--annotations",
            str(annotations_path),
            "--dataset-root",
            str(tmp_path),
            "--config",
            "../configs/model.yaml",
            "--experiment-name",
            "real_test_run",
            "--output-dir",
            str(tmp_path / "checkpoints"),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-size",
            "8",
            "--num-layers",
            "1",
            "--device",
            "cpu",
        ]
    )

    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["split_strategy"] == "gloss_stratified (not signer-independent)"
    assert checkpoint["source_tags"] == ["real_webcam"]
    assert checkpoint["demo_mode"] is False
    assert checkpoint["seed"] == 42
