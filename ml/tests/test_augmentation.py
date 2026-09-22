import numpy as np

from ml.preprocessing.augmentation import (
    add_gaussian_noise,
    compose,
    rotate_sequence,
    scale_sequence,
    temporal_frame_dropout,
    translate_sequence,
)


def _sample_sequence(t: int = 10, k: int = 21) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(-1.0, 1.0, size=(t, k, 3)).astype(np.float32)


def test_rotate_sequence_preserves_shape_and_changes_values():
    sequence = _sample_sequence()
    rotated = rotate_sequence(sequence, max_degrees=30, rng=np.random.default_rng(1))
    assert rotated.shape == sequence.shape
    assert not np.allclose(rotated, sequence)


def test_rotate_sequence_preserves_vector_norms():
    """A rotation must not change distances between points -- only orientation."""
    sequence = _sample_sequence()
    rotated = rotate_sequence(sequence, max_degrees=45, rng=np.random.default_rng(2))
    original_norms = np.linalg.norm(sequence, axis=-1)
    rotated_norms = np.linalg.norm(rotated, axis=-1)
    np.testing.assert_allclose(original_norms, rotated_norms, atol=1e-4)


def test_scale_sequence_scales_all_points_uniformly():
    sequence = np.ones((5, 3, 3), dtype=np.float32)
    scaled = scale_sequence(sequence, scale_range=(2.0, 2.0), rng=np.random.default_rng(3))
    np.testing.assert_allclose(scaled, sequence * 2.0)


def test_translate_sequence_shifts_xy_but_not_z():
    sequence = np.zeros((4, 2, 3), dtype=np.float32)
    translated = translate_sequence(sequence, max_shift=0.1, rng=np.random.default_rng(4))
    assert np.all(translated[:, :, 2] == 0.0)
    assert not np.all(translated[:, :, 0] == 0.0)


def test_add_gaussian_noise_is_deterministic_with_seeded_rng():
    sequence = _sample_sequence()
    noisy_a = add_gaussian_noise(sequence, sigma=0.02, rng=np.random.default_rng(5))
    noisy_b = add_gaussian_noise(sequence, sigma=0.02, rng=np.random.default_rng(5))
    np.testing.assert_allclose(noisy_a, noisy_b)


def test_temporal_frame_dropout_zeroes_some_frames_entirely():
    sequence = np.ones((20, 3, 3), dtype=np.float32)
    dropped = temporal_frame_dropout(sequence, drop_prob=0.5, rng=np.random.default_rng(6))
    zeroed_frames = np.all(dropped == 0.0, axis=(1, 2))
    assert zeroed_frames.any()
    assert not zeroed_frames.all()


def test_compose_applies_augmentations_in_order():
    sequence = np.ones((3, 2, 3), dtype=np.float32)

    def add_one(seq: np.ndarray) -> np.ndarray:
        return seq + 1.0

    def double(seq: np.ndarray) -> np.ndarray:
        return seq * 2.0

    pipeline = compose(add_one, double)
    result = pipeline(sequence)
    # (1 + 1) * 2 = 4
    np.testing.assert_allclose(result, np.full((3, 2, 3), 4.0, dtype=np.float32))
