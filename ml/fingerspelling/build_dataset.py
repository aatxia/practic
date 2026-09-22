#!/usr/bin/env python3
"""
Runs the real MediaPipe hand detector over every photo in
data/USL_alphabet_train/ (excluding the `_excluded_*` folders -- see
ml/fingerspelling/dataset.py's docstring) and saves the resulting
normalized hand-landmark feature vectors + labels to a single .npz file,
so extraction (the slow part -- one real detector call per image) only
has to run once.

Usage:
    python -m ml.fingerspelling.build_dataset \
        --data-root data/USL_alphabet_train \
        --output ml/fingerspelling/features.npz
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ml.fingerspelling.dataset import extract_features, scan_image_dataset
from ml.preprocessing.landmarks import FeatureToggles, LandmarkExtractor


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/USL_alphabet_train"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/mediapipe"))
    parser.add_argument("--output", type=Path, default=Path("ml/fingerspelling/features.npz"))
    args = parser.parse_args(argv)

    samples = scan_image_dataset(args.data_root)
    per_class = Counter(s.label for s in samples)
    print(f"Scanned {len(samples)} images across {len(per_class)} classes.")
    for label, count in sorted(per_class.items()):
        print(f"  {label}: {count}")

    extractor = LandmarkExtractor(args.models_dir, FeatureToggles(hands=True, pose=False, face=False))
    print("\nExtracting hand landmarks (one real detector call per image -- this is the slow part; "
          "progress prints every 2000 images, it is not frozen)...")
    t0 = time.time()
    result = extract_features(samples, extractor, progress_every=2000)
    dt = time.time() - t0

    kept = len(result.labels)
    print(
        f"\nExtracted {kept}/{len(samples)} in {dt:.1f}s "
        f"({dt / max(len(samples), 1) * 1000:.1f} ms/image)."
    )
    print(f"Skipped {len(result.skipped)}: ", Counter(reason for _, reason in result.skipped))
    print(f"Left-hand detections: {result.left_hand_count} (mirrored), right-hand: {result.right_hand_count}")

    kept_per_class = Counter(result.labels)
    print("\nKept per class (after skips):")
    for label in sorted(per_class):
        print(f"  {label}: {kept_per_class.get(label, 0)}/{per_class[label]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=result.features,
        labels=np.array(result.labels, dtype="<U8"),
    )
    print(f"\nSaved {result.features.shape} feature matrix to {args.output}")

    skipped_log = args.output.with_suffix(".skipped.txt")
    with skipped_log.open("w", encoding="utf-8") as f:
        for path, reason in result.skipped:
            f.write(f"{path}\t{reason}\n")
    print(f"Skip log: {skipped_log}")

    return args.output


if __name__ == "__main__":
    main()
