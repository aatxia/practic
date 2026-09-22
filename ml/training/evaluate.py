#!/usr/bin/env python3
"""
evaluate — loads a checkpoint saved by ml/training/train.py and reports
per-class accuracy on its held-out val and (reserved, never touched by
train.py) test splits, reconstructed with the exact split_strategy/seed
the checkpoint recorded. This is what tells you what a checkpoint
actually does, honestly -- including any weak/noisy classes -- rather
than just the single running val_accuracy number train.py printed
during training.

Usage:
    python -m ml.training.evaluate \
        --checkpoint models/checkpoints/real_v1/latest.pt \
        --annotations data/real/annotations/aatxia_annotations.jsonl \
        --dataset-root .
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.datasets.annotation import load_annotations
from ml.datasets.split import SplitRatios, gloss_stratified_split, signer_independent_split
from ml.features.feature_vector import FeatureConfig
from ml.models.lstm import LSTMConfig, LSTMSignClassifier
from ml.training.dataset import SignSequenceDataset


def reconstruct_split(checkpoint: dict, samples: list, ratios: SplitRatios) -> dict[str, list[str]]:
    strategy = checkpoint["split_strategy"]
    seed = checkpoint["seed"]
    if strategy == "signer_independent":
        return signer_independent_split(samples, ratios=ratios, seed=seed)
    if strategy.startswith("gloss_stratified"):
        return gloss_stratified_split(samples, ratios=ratios, seed=seed)
    raise ValueError(f"unknown split_strategy recorded in checkpoint: {strategy!r}")


@torch.no_grad()
def per_class_accuracy(
    model: LSTMSignClassifier, loader: DataLoader, index_to_label: dict[int, str], device: torch.device
) -> tuple[float, dict[str, tuple[int, int]]]:
    """Returns (overall_accuracy, {label: (correct, total)})."""
    counts: dict[str, list[int]] = {label: [0, 0] for label in index_to_label.values()}
    correct_total, total = 0, 0
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        preds = model(sequences).argmax(dim=1)
        for pred, label in zip(preds.tolist(), labels.tolist(), strict=True):
            label_name = index_to_label[label]
            counts[label_name][1] += 1
            total += 1
            if pred == label:
                counts[label_name][0] += 1
                correct_total += 1
    overall = correct_total / total if total else 0.0
    return overall, {label: (c, n) for label, (c, n) in counts.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    samples = load_annotations(args.annotations)
    ratios = SplitRatios(train=args.train_ratio, val=args.val_ratio, test=args.test_ratio)
    split = reconstruct_split(checkpoint, samples, ratios)
    by_id = {sample.sample_id: sample for sample in samples}

    label_to_index: dict[str, int] = checkpoint["label_to_index"]
    index_to_label = {index: label for label, index in label_to_index.items()}
    feature_config = FeatureConfig(**checkpoint["feature_config"])

    model = LSTMSignClassifier(LSTMConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint: {args.checkpoint}")
    print(
        f"split_strategy={checkpoint['split_strategy']!r}  seed={checkpoint['seed']}  "
        f"demo_mode={checkpoint['demo_mode']}  source_tags={checkpoint['source_tags']}"
    )
    print()

    for split_name in ("val", "test"):
        split_samples = [by_id[sample_id] for sample_id in split[split_name]]
        if not split_samples:
            print(f"[{split_name}] empty, skipping")
            continue
        dataset = SignSequenceDataset(
            split_samples,
            args.dataset_root,
            label_to_index,
            checkpoint["sequence_length"],
            feature_config=feature_config,
            target_fps=checkpoint["camera_fps"],
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        overall, per_class = per_class_accuracy(model, loader, index_to_label, device)

        print(f"[{split_name}] n={len(split_samples)}  overall accuracy={overall:.4f}")
        for label in sorted(per_class):
            correct, n = per_class[label]
            if n == 0:
                continue
            flag = "  <- only a few samples, noisy" if n < 5 else ""
            print(f"    {label:12s} {correct}/{n} = {correct / n:.2f}{flag}")
        print()


if __name__ == "__main__":
    main()
