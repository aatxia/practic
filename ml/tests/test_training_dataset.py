import numpy as np
import torch

from ml.datasets.synthetic import generate_demo_dataset
from ml.training.dataset import SignSequenceDataset, pad_or_truncate


def test_pad_or_truncate_pads_short_sequences_with_zeros():
    sequence = np.ones((5, 4), dtype=np.float32)

    result = pad_or_truncate(sequence, 8)

    assert result.shape == (8, 4)
    np.testing.assert_array_equal(result[:5], sequence)
    np.testing.assert_array_equal(result[5:], np.zeros((3, 4), dtype=np.float32))


def test_pad_or_truncate_truncates_long_sequences():
    sequence = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)

    result = pad_or_truncate(sequence, 4)

    assert result.shape == (4, 3)
    np.testing.assert_array_equal(result, sequence[:4])


def test_pad_or_truncate_leaves_exact_length_unchanged():
    sequence = np.ones((6, 2), dtype=np.float32)

    result = pad_or_truncate(sequence, 6)

    assert result is sequence


def test_dataset_returns_tensor_and_correct_label(tmp_path):
    samples = generate_demo_dataset(tmp_path, sequence_length=16, samples_per_signer_gloss=1, seed=1)
    labels = sorted({s.gloss for s in samples})
    label_to_index = {label: i for i, label in enumerate(labels)}

    dataset = SignSequenceDataset(samples, tmp_path, label_to_index, sequence_length=16)
    sequence_tensor, label_tensor = dataset[0]

    assert isinstance(sequence_tensor, torch.Tensor)
    assert sequence_tensor.dtype == torch.float32
    assert sequence_tensor.shape[0] == 16
    assert label_tensor.item() == label_to_index[samples[0].gloss]
    assert len(dataset) == len(samples)
