"""
augmentation — geometric/temporal augmentations applied to a *sequence* of
normalized landmarks during training (Phase 9+), to make the model robust to
natural variation between signers and recording conditions.

Every function operates on an array of shape (T, K, 3) -- T frames, K
landmarks per frame, 3 coordinates -- and returns an array of the same
shape. All functions accept an optional `numpy.random.Generator` so
augmentations are reproducible in tests and experiments (seed once, reuse).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

Augmentation = Callable[[np.ndarray], np.ndarray]


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def rotate_sequence(
    sequence: np.ndarray, max_degrees: float = 15.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Rotate the whole sequence by one random angle around the z-axis (the
    camera-facing axis) -- simulates the signer facing slightly left/right of
    center. One angle per sequence, not per frame, to preserve the sign's
    internal temporal coherence.
    """
    generator = _rng(rng)
    angle = np.deg2rad(generator.uniform(-max_degrees, max_degrees))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    return sequence @ rotation_matrix.T


def scale_sequence(
    sequence: np.ndarray,
    scale_range: tuple[float, float] = (0.9, 1.1),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Uniformly scale the whole sequence by one random factor -- simulates
    natural variation in how large/small different signers make a sign."""
    generator = _rng(rng)
    factor = generator.uniform(*scale_range)
    return sequence * factor


def translate_sequence(
    sequence: np.ndarray, max_shift: float = 0.05, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Shift the whole sequence by one random (x, y) offset -- simulates
    imperfect normalization / the signer not being perfectly centered."""
    generator = _rng(rng)
    shift = generator.uniform(-max_shift, max_shift, size=3).astype(np.float32)
    shift[2] = 0.0  # leave depth untouched; z-jitter isn't a realistic camera-position effect
    return sequence + shift


def add_gaussian_noise(
    sequence: np.ndarray, sigma: float = 0.01, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Per-coordinate Gaussian jitter -- simulates MediaPipe landmark detection
    noise (never perfectly stable frame-to-frame even for a static hand)."""
    generator = _rng(rng)
    noise = generator.normal(0.0, sigma, size=sequence.shape).astype(np.float32)
    return sequence + noise


def temporal_frame_dropout(
    sequence: np.ndarray, drop_prob: float = 0.1, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Zero out a random subset of frames -- simulates transient detection
    failure (motion blur, brief occlusion), which the temporal model
    (BiLSTM/Transformer, Phase 9) must learn to tolerate."""
    generator = _rng(rng)
    result = sequence.copy()
    drop_mask = generator.random(sequence.shape[0]) < drop_prob
    result[drop_mask] = 0.0
    return result


def compose(*augmentations: Augmentation) -> Augmentation:
    """Chain augmentations into a single callable, applied in order."""

    def _apply(sequence: np.ndarray) -> np.ndarray:
        for augmentation in augmentations:
            sequence = augmentation(sequence)
        return sequence

    return _apply
