"""
split — signer-independent train/val/test split (section 11).

Hard rule, not a preference: a single `signer_id` must never appear in more
than one split. If the same person's samples land in both train and test,
a model can reach high test accuracy by recognizing *that signer's*
appearance/handshape quirks instead of the sign itself -- the evaluation
would look good and mean nothing. So the unit this splitter assigns is a
whole signer, never an individual sample.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ml.datasets.annotation import SampleAnnotation

_EPS = 1e-3
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        if self.train < 0 or self.val < 0 or self.test < 0:
            raise ValueError("split ratios must be non-negative")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > _EPS:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")


DEFAULT_SPLIT_RATIOS = SplitRatios()


def signer_independent_split(
    samples: list[SampleAnnotation],
    ratios: SplitRatios = DEFAULT_SPLIT_RATIOS,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Assigns every sample to train/val/test by assigning its *signer*,
    never the sample individually. Returns `{"train": [sample_id, ...], ...}`
    (only splits with a non-zero ratio are non-empty).

    Deterministic given `seed`. Uses greedy group balancing: signers are
    visited in a seeded-random order, and each one is assigned whole to
    whichever split is currently furthest below its target sample-count
    share -- this keeps the achieved ratios close to the requested ones
    even though signers can't be split to hit them exactly.
    """
    if not samples:
        raise ValueError("cannot split an empty sample list")

    by_signer: dict[str, list[SampleAnnotation]] = {}
    for sample in samples:
        by_signer.setdefault(sample.signer_id, []).append(sample)

    target_ratio = {"train": ratios.train, "val": ratios.val, "test": ratios.test}
    active_splits = [name for name in SPLIT_NAMES if target_ratio[name] > 0]
    if len(by_signer) < len(active_splits):
        raise ValueError(
            f"need at least {len(active_splits)} distinct signers for a "
            f"{'/'.join(active_splits)} split (signer-independent), got only "
            f"{len(by_signer)}: {sorted(by_signer)}"
        )

    signer_ids = sorted(by_signer)  # deterministic base order before shuffling
    random.Random(seed).shuffle(signer_ids)

    total_samples = len(samples)
    assigned_counts = dict.fromkeys(SPLIT_NAMES, 0)
    signer_assignment: dict[str, str] = {}

    for signer_id in signer_ids:
        best_split = max(active_splits, key=lambda name: target_ratio[name] * total_samples - assigned_counts[name])
        signer_assignment[signer_id] = best_split
        assigned_counts[best_split] += len(by_signer[signer_id])

    result: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for sample in samples:
        result[signer_assignment[sample.signer_id]].append(sample.sample_id)
    for sample_ids in result.values():
        sample_ids.sort()

    return result


MIN_SAMPLES_FOR_VAL_TEST = 5


def gloss_stratified_split(
    samples: list[SampleAnnotation],
    ratios: SplitRatios = DEFAULT_SPLIT_RATIOS,
    seed: int = 42,
    min_samples_for_val_test: int = MIN_SAMPLES_FOR_VAL_TEST,
) -> dict[str, list[str]]:
    """Per-gloss stratified split -- the fallback for datasets with too few
    distinct signers for signer_independent_split (e.g. one person's own
    recorded word dataset, scripts/record_dataset.py's output before more
    signers are added).

    This is NOT signer-independent: the same person's handshape/appearance
    can appear in both train and val/test, so a val/test accuracy measured
    this way tells you "did the model learn to recognize this gloss as this
    specific signer performs it", not generalization to anyone else. That's
    a real, documented limitation of a single-signer dataset -- a report
    built from this split must say so, not imply a rigor it doesn't have.

    A gloss with fewer than `min_samples_for_val_test` samples goes
    entirely into train -- too little to hold any out meaningfully.
    """
    if not samples:
        raise ValueError("cannot split an empty sample list")

    rng = random.Random(seed)
    by_gloss: dict[str, list[SampleAnnotation]] = {}
    for sample in samples:
        by_gloss.setdefault(sample.gloss, []).append(sample)

    result: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for gloss in sorted(by_gloss):
        gloss_samples = list(by_gloss[gloss])
        rng.shuffle(gloss_samples)
        n = len(gloss_samples)

        if n < min_samples_for_val_test:
            result["train"].extend(sample.sample_id for sample in gloss_samples)
            continue

        n_test = max(1, round(n * ratios.test)) if ratios.test > 0 else 0
        n_val = max(1, round(n * ratios.val)) if ratios.val > 0 else 0
        n_test = min(n_test, max(0, n - 2))
        n_val = min(n_val, max(0, n - n_test - 1))

        result["test"].extend(sample.sample_id for sample in gloss_samples[:n_test])
        result["val"].extend(sample.sample_id for sample in gloss_samples[n_test : n_test + n_val])
        result["train"].extend(sample.sample_id for sample in gloss_samples[n_test + n_val :])

    for sample_ids in result.values():
        sample_ids.sort()

    return result
