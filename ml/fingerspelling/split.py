"""
fingerspelling.split -- per-class stratified train/val/test split for the
USL_alphabet_train feature set.

This is deliberately NOT ml/datasets/split.py's signer-independent split:
that one exists because sign *recognition from video* would otherwise let a
model cheat by memorizing a signer's individual appearance across train and
test. USL_alphabet_train carries no signer_id at all (no metadata says who
performed which photo), so that safeguard cannot be applied here -- this is
a real, documented limitation of this dataset, not an oversight. A plain
stratified split is the honest thing this data supports; a report built
from it should say so rather than imply signer-independence it doesn't have.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_SAMPLES_FOR_VAL_TEST = 5


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def stratified_split(
    labels: list[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Every class is split independently so even a rare class appears in
    every split it has enough samples for. A class with fewer than
    MIN_SAMPLES_FOR_VAL_TEST samples goes entirely into train -- there's
    too little of it to hold any out without the val/test number for that
    class being statistically meaningless anyway.
    """
    rng = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for label in sorted(set(labels)):
        idx = np.where(labels_arr == label)[0]
        rng.shuffle(idx)
        n = len(idx)

        if n < MIN_SAMPLES_FOR_VAL_TEST:
            train_idx.extend(idx.tolist())
            continue

        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        n_test = min(n_test, n - 2)  # leave at least 2 for train + val
        n_val = min(n_val, n - n_test - 1)  # leave at least 1 for train

        test_idx.extend(idx[:n_test].tolist())
        val_idx.extend(idx[n_test : n_test + n_val].tolist())
        train_idx.extend(idx[n_test + n_val :].tolist())

    return SplitIndices(
        train=np.array(sorted(train_idx)),
        val=np.array(sorted(val_idx)),
        test=np.array(sorted(test_idx)),
    )
