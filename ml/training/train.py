#!/usr/bin/env python3
"""
train — baseline model training loop (Phase 9). Loads annotations, does a
signer-independent split (section 11), trains ml/models/lstm.py on the
train split, evaluates on val, and saves a checkpoint with everything Phase
10 inference will need to reconstruct the model and its label mapping.

Runs identically locally and in Google Colab (GPU) -- no Colab-specific
paths or hacks. From the repo root:

    python -m ml.training.train --annotations data/annotations/demo_annotations.jsonl

In Colab, after cloning the repo and `pip install -r ml/requirements-training.txt`
(torch is already preinstalled with GPU support there -- see that file's
header before running pip install), the same command works via:

    !python -m ml.training.train --annotations data/annotations/demo_annotations.jsonl
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.datasets.annotation import SampleAnnotation, load_annotations
from ml.datasets.split import SplitRatios, gloss_stratified_split, signer_independent_split
from ml.features.feature_vector import feature_vector_size
from ml.models.lstm import LSTMConfig, LSTMSignClassifier
from ml.training.config import TrainingConfig, load_training_config
from ml.training.dataset import SignSequenceDataset


def build_label_vocabulary(samples: list[SampleAnnotation]) -> dict[str, int]:
    """Built from the *entire* annotation file, not just the train split --
    the task's classes are fixed by the dataset, so val/test must map to the
    same indices even if (by chance) a class is thin in one split."""
    labels = sorted({sample.gloss for sample in samples})
    return {label: index for index, label in enumerate(labels)}


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * sequences.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += sequences.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        logits = model(sequences)
        loss = criterion(logits, labels)
        total_loss += loss.item() * sequences.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += sequences.size(0)
    return total_loss / total, correct / total


def save_checkpoint(
    path: Path,
    model: LSTMSignClassifier,
    model_config: LSTMConfig,
    training_config: TrainingConfig,
    label_to_index: dict[str, int],
    source_tags: list[str],
    val_accuracy: float,
    split_strategy: str,
    seed: int,
) -> None:
    """Everything Phase 10 inference needs to reconstruct this exact model
    and interpret its output, plus enough provenance (source_tags, demo_mode,
    split_strategy, seed) that a demo-only checkpoint can never be silently
    mistaken for one trained on real Ukrainian Sign Language data, a
    val_accuracy measured without signer-independence can never be silently
    mistaken for one that generalizes across signers, and the exact train/val/
    test split can always be reconstructed later (e.g. to evaluate the
    reserved test split, which this script itself never touches)."""
    demo_mode = source_tags == ["demo_synthetic"]
    checkpoint = {
        "model_type": training_config.model_type,
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "feature_config": asdict(training_config.feature_config),
        "sequence_length": training_config.sequence_length,
        "label_to_index": label_to_index,
        "source_tags": source_tags,
        "demo_mode": demo_mode,
        "val_accuracy": val_accuracy,
        "split_strategy": split_strategy,
        "seed": seed,
        "camera_fps": training_config.camera_fps,
        "trained_at": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True, help="Path to a JSONL annotation file")
    parser.add_argument("--dataset-root", default="data", help="Root that clip_path in annotations is relative to")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--experiment-name", default="baseline")
    parser.add_argument("--output-dir", default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Overrides configs/model.yaml training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Overrides training.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Overrides training.learning_rate")
    parser.add_argument("--hidden-size", type=int, default=None, help="Overrides training.hidden_size")
    parser.add_argument("--num-layers", type=int, default=None, help="Overrides training.num_layers")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda|cpu (default: auto-detect)")
    args = parser.parse_args(argv)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_config = load_training_config(args.config)
    epochs = args.epochs if args.epochs is not None else training_config.epochs
    batch_size = args.batch_size if args.batch_size is not None else training_config.batch_size
    learning_rate = args.lr if args.lr is not None else training_config.learning_rate
    hidden_size = args.hidden_size if args.hidden_size is not None else training_config.hidden_size
    num_layers = args.num_layers if args.num_layers is not None else training_config.num_layers

    samples = load_annotations(args.annotations)
    label_to_index = build_label_vocabulary(samples)

    ratios = SplitRatios(train=args.train_ratio, val=args.val_ratio, test=args.test_ratio)
    active_splits = [name for name, ratio in (("train", ratios.train), ("val", ratios.val), ("test", ratios.test)) if ratio > 0]
    distinct_signers = {sample.signer_id for sample in samples}
    if len(distinct_signers) >= len(active_splits):
        split = signer_independent_split(samples, ratios=ratios, seed=args.seed)
        split_strategy = "signer_independent"
    else:
        print(
            f"Only {len(distinct_signers)} distinct signer(s) ({sorted(distinct_signers)}) -- not enough for a "
            f"signer-independent {'/'.join(active_splits)} split. Falling back to a per-gloss stratified split: "
            "val/test accuracy will measure whether the model learned these glosses as THIS signer performs "
            "them, not generalization to other signers."
        )
        split = gloss_stratified_split(samples, ratios=ratios, seed=args.seed)
        split_strategy = "gloss_stratified (not signer-independent)"
    by_id = {sample.sample_id: sample for sample in samples}
    train_samples = [by_id[sample_id] for sample_id in split["train"]]
    val_samples = [by_id[sample_id] for sample_id in split["val"]]
    if not val_samples:
        raise ValueError("val split is empty -- adjust --val-ratio or add more signers to the dataset")
    print(
        f"split: train={len(train_samples)} val={len(val_samples)} "
        f"test={len(split['test'])} (test is reserved, not evaluated by this script)"
    )

    train_ds = SignSequenceDataset(
        train_samples,
        args.dataset_root,
        label_to_index,
        training_config.sequence_length,
        feature_config=training_config.feature_config,
        target_fps=training_config.camera_fps,
    )
    val_ds = SignSequenceDataset(
        val_samples,
        args.dataset_root,
        label_to_index,
        training_config.sequence_length,
        feature_config=training_config.feature_config,
        target_fps=training_config.camera_fps,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_size = feature_vector_size(training_config.feature_config)
    model_config = LSTMConfig(
        input_size=input_size, num_classes=len(label_to_index), hidden_size=hidden_size, num_layers=num_layers
    )
    model = LSTMSignClassifier(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    source_tags = sorted({sample.source for sample in samples})
    if source_tags == ["demo_synthetic"]:
        print("=" * 70)
        print("[DEMO MODE] Training on 100% synthetic data (source=demo_synthetic).")
        print("This checkpoint will NOT recognize real Ukrainian Sign Language --")
        print("it only proves the training pipeline works end-to-end.")
        print("=" * 70)

    output_dir = Path(args.output_dir) / args.experiment_name
    checkpoint_path = output_dir / "latest.pt"
    best_val_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch {epoch}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc >= best_val_accuracy:
            best_val_accuracy = val_acc
            save_checkpoint(
                checkpoint_path,
                model,
                model_config,
                training_config,
                label_to_index,
                source_tags,
                val_acc,
                split_strategy,
                args.seed,
            )

    print(f"Training complete. Best val accuracy: {best_val_accuracy:.4f}. Checkpoint: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    main()
