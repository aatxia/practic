"""
dataset — torch Dataset that turns SampleAnnotations into fixed-length
feature-vector tensors + integer labels, for ml/training/train.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.datasets.annotation import SampleAnnotation
from ml.datasets.dataset import load_feature_sequence
from ml.features.feature_vector import FeatureConfig


def pad_or_truncate(sequence: np.ndarray, sequence_length: int) -> np.ndarray:
    """Fits `sequence` to exactly `sequence_length` frames: truncates from the
    end if longer, zero-pads at the end if shorter. A fixed-length input is
    what LSTMSignClassifier's batching needs; padding with zeros matches what
    normalize_frame() already uses for "no detection", so the model sees a
    consistent signal rather than an arbitrary sentinel."""
    length, feature_size = sequence.shape
    if length == sequence_length:
        return sequence
    if length > sequence_length:
        return sequence[:sequence_length]
    padded = np.zeros((sequence_length, feature_size), dtype=sequence.dtype)
    padded[:length] = sequence
    return padded


class SignSequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[SampleAnnotation],
        dataset_root: Path | str,
        label_to_index: dict[str, int],
        sequence_length: int,
        feature_config: FeatureConfig | None = None,
        target_fps: float | None = None,
    ) -> None:
        self.samples = samples
        self.dataset_root = dataset_root
        self.label_to_index = label_to_index
        self.sequence_length = sequence_length
        # Real (non-demo) samples decode raw video through the full
        # MediaPipe pipeline (ml/datasets/dataset.py) -- far too slow to
        # redo on every __getitem__ call (every batch, every epoch), so
        # each sample's raw sequence is computed once and kept here for
        # the lifetime of this Dataset instance.
        self.feature_config = feature_config
        self.target_fps = target_fps
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        if sample.sample_id not in self._cache:
            self._cache[sample.sample_id] = load_feature_sequence(
                sample, self.dataset_root, self.feature_config, self.target_fps
            )
        sequence = pad_or_truncate(self._cache[sample.sample_id], self.sequence_length)
        label = self.label_to_index[sample.gloss]
        return torch.from_numpy(sequence).float(), torch.tensor(label, dtype=torch.long)
