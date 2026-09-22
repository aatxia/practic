#!/usr/bin/env python3
"""
Trains FingerspellingMLP on the real USL_alphabet_train hand-landmark
features (ml/fingerspelling/build_dataset.py's output) -- honest per-class
accuracy included in the printed report, not just one overall number,
because several classes have only a few dozen real samples total and an
aggregate accuracy would hide that.

Usage:
    python -m ml.fingerspelling.train \
        --features ml/fingerspelling/features.npz \
        --output ml/fingerspelling/checkpoints/latest.pt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.fingerspelling.model import FingerspellingMLP
from ml.fingerspelling.split import stratified_split


def _class_weights(train_labels: np.ndarray, num_classes: int, label_to_index: dict[str, int]) -> torch.Tensor:
    """Inverse-sqrt-frequency weights for the loss, so the ~6000-sample
    classes don't drown out the ~24-sample ones during training -- a
    standard mitigation for real class imbalance, not a way to hide it (the
    printed report below still shows the raw per-class counts and
    accuracy). Plain inverse frequency (1/count) was tried first and
    overcorrected badly here: with counts ranging 16-3427 (~200x), it made
    the single largest class (Ж) so cheap to misclassify in the weighted
    loss that the model gave up on it almost entirely (1% test accuracy on
    Ж despite it having the most training data of any class) in exchange
    for the rare classes. sqrt tempers that ratio to ~14x, which is still a
    real correction but doesn't sacrifice the majority class to get it."""
    counts = Counter(train_labels.tolist())
    weights = torch.ones(num_classes, dtype=torch.float32)
    for label, index in label_to_index.items():
        weights[index] = 1.0 / max(counts.get(label, 1), 1) ** 0.5
    return weights / weights.sum() * num_classes


def _evaluate(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(features)
        predicted = logits.argmax(dim=1)
        correct = (predicted == labels)

    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)
    for true_label, is_correct in zip(labels.tolist(), correct.tolist()):
        per_class_total[true_label] += 1
        per_class_correct[true_label] += int(is_correct)

    overall = correct.float().mean().item() if len(labels) else 0.0
    return {
        "overall_accuracy": overall,
        "per_class_correct": per_class_correct,
        "per_class_total": per_class_total,
        "predicted": predicted,
    }


def _top_confusions(
    predicted: torch.Tensor, labels: torch.Tensor, index_to_label: dict[int, str], top_n: int = 3
) -> dict[str, list[tuple[str, int]]]:
    """For every true class with at least one wrong prediction, the top
    `top_n` (wrong_label, count) pairs it was actually mistaken for -- a
    concrete diagnostic for a low per-class accuracy (e.g. is Щ really
    confused with the visually similar Ш, or something else entirely)
    instead of a single opaque accuracy number that can't say why."""
    confusions: dict[int, Counter] = {}
    for true_index, predicted_index in zip(labels.tolist(), predicted.tolist()):
        if true_index != predicted_index:
            confusions.setdefault(true_index, Counter())[predicted_index] += 1
    return {
        index_to_label[true_index]: [
            (index_to_label[predicted_index], count) for predicted_index, count in counter.most_common(top_n)
        ]
        for true_index, counter in confusions.items()
    }


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("ml/fingerspelling/features.npz"))
    parser.add_argument("--output", type=Path, default=Path("ml/fingerspelling/checkpoints/latest.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    data = np.load(args.features)
    features = data["features"]
    labels = data["labels"]

    classes = sorted(set(labels.tolist()))
    label_to_index = {label: i for i, label in enumerate(classes)}
    label_indices = np.array([label_to_index[label] for label in labels])

    split = stratified_split(labels.tolist(), seed=args.seed)
    print(
        f"Split: train={len(split.train)} val={len(split.val)} test={len(split.test)} "
        f"(classes with <{5} samples go entirely to train -- see ml/fingerspelling/split.py)"
    )

    x_train = torch.tensor(features[split.train], dtype=torch.float32)
    y_train = torch.tensor(label_indices[split.train], dtype=torch.long)
    x_val = torch.tensor(features[split.val], dtype=torch.float32)
    y_val = torch.tensor(label_indices[split.val], dtype=torch.long)
    x_test = torch.tensor(features[split.test], dtype=torch.float32)
    y_test = torch.tensor(label_indices[split.test], dtype=torch.long)

    torch.manual_seed(args.seed)
    model = FingerspellingMLP(num_classes=len(classes), hidden_size=args.hidden_size)
    weights = _class_weights(labels[split.train], len(classes), label_to_index)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Mini-batch SGD, not one full-batch gradient step per "epoch": with
    # ~20000 train samples, a single full-batch update per epoch is only a
    # few dozen total weight updates across a whole run -- too few to
    # reliably converge, and (having no other randomness in the loop:
    # FingerspellingMLP has no dropout) sensitive to tiny floating-point
    # differences between machines/torch builds compounding over so few
    # correlated steps. Real per-batch updates converge faster and far more
    # consistently across environments, same reasoning as ml/training/
    # train.py's DataLoader-based loop for the word-level LSTM.
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed)
    )

    best_val_accuracy = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x_batch)
        epoch_loss /= len(x_train)

        val_result = _evaluate(model, x_val, y_val, len(classes)) if len(x_val) else {"overall_accuracy": 0.0}
        if val_result["overall_accuracy"] >= best_val_accuracy:
            best_val_accuracy = val_result["overall_accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch}/{args.epochs} | train_loss={epoch_loss:.4f} | val_acc={val_result['overall_accuracy']:.4f}")

    model.load_state_dict(best_state)
    test_result = _evaluate(model, x_test, y_test, len(classes))

    print(f"\nBest val accuracy: {best_val_accuracy:.4f}")
    print(f"Test overall accuracy: {test_result['overall_accuracy']:.4f} (n={len(x_test)})")
    print("\nPer-class test accuracy (0 total = class had too few samples for a held-out test split):")
    index_to_label = {i: label for label, i in label_to_index.items()}
    test_confusions = _top_confusions(test_result["predicted"], y_test, index_to_label)
    for i in range(len(classes)):
        total = int(test_result["per_class_total"][i].item())
        correct = int(test_result["per_class_correct"][i].item())
        acc = correct / total if total else float("nan")
        train_count = int((y_train == i).sum().item())
        print(f"  {index_to_label[i]}: test_acc={acc:.2f} ({correct}/{total}), train_n={train_count}")

    # A separate, honest look at exactly what a weak class gets mistaken
    # for -- 89% overall can hide a class the model has essentially never
    # learned, and "0.00 accuracy" alone doesn't say whether that's random
    # noise or a real, explainable confusion (e.g. visually similar
    # handshapes) worth fixing before shipping the checkpoint.
    weak_classes = [
        index_to_label[i]
        for i in range(len(classes))
        if test_result["per_class_total"][i] > 0
        and test_result["per_class_correct"][i] / test_result["per_class_total"][i] < 0.85
    ]
    if weak_classes:
        print("\nWeak classes (test_acc < 0.85) -- what they get mistaken for instead:")
        for label in weak_classes:
            pairs = test_confusions.get(label, [])
            pairs_str = ", ".join(f"{other}x{count}" for other, count in pairs) if pairs else "no single dominant wrong class"
            print(f"  {label}: mistaken for {pairs_str}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "label_to_index": label_to_index,
            "hidden_size": args.hidden_size,
            "val_accuracy": best_val_accuracy,
            "test_accuracy": test_result["overall_accuracy"],
            "source": "usl_alphabet_train_real_photos",
        },
        args.output,
    )
    print(f"\nSaved checkpoint to {args.output}")

    report_path = args.output.with_suffix(".report.json")
    report = {
        "classes": classes,
        "train_n": len(split.train),
        "val_n": len(split.val),
        "test_n": len(split.test),
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_result["overall_accuracy"],
        "per_class": {
            index_to_label[i]: {
                "test_correct": int(test_result["per_class_correct"][i].item()),
                "test_total": int(test_result["per_class_total"][i].item()),
                "train_n": int((y_train == i).sum().item()),
                "confused_with": test_confusions.get(index_to_label[i], []),
            }
            for i in range(len(classes))
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report to {report_path}")

    return args.output


if __name__ == "__main__":
    main()
