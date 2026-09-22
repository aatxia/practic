"""
DEMO MODE ONLY.

Generates a small, fully synthetic "sign" dataset so the dataset pipeline
(annotation format, signer-independent split, sequence loading) and later
the Phase 9 training loop can be exercised end-to-end before a real
Ukrainian Sign Language video dataset is available. As of Phase 8, no
public isolated-sign УЖМ dataset with signer_id labeling could be found
(see PROJECT_STATUS.md, Phase 8) -- this is not a substitute for one and
must never be presented as real training data.

The feature-vector sequences here are seeded random numbers with a
class-dependent offset -- NOT derived from MediaPipe, a camera, or any real
video. Every sample is tagged `source="demo_synthetic"`
(see ml/datasets/annotation.py) so it can never be silently mistaken for
real data, per the project's "no fake AI" rule: this proves the pipeline
plumbing works end-to-end, it says nothing about real-world sign
recognition accuracy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.datasets.annotation import SampleAnnotation, write_annotations
from ml.features.feature_vector import FeatureConfig, feature_vector_size

DEMO_SOURCE_TAG = "demo_synthetic"

# Placeholder gloss labels -- distinct class names to exercise multi-class
# classification. NOT a claim that these are correct/complete UkrSL signs.
DEMO_GLOSSES = ["PRIVIT", "DYAKUYU", "TAK", "NI", "BUD_LASKA"]

# Six distinct fake signers, so a 70/15/15 signer-independent split has
# enough groups to work with (see ml/datasets/split.py).
DEMO_SIGNERS = [f"demo_signer_{i:02d}" for i in range(1, 7)]

DEFAULT_FEATURE_CONFIG = FeatureConfig()


def _synthetic_sequence(
    rng: np.random.Generator, gloss_index: int, sequence_length: int, feature_size: int
) -> np.ndarray:
    """One fake `(sequence_length, feature_size)` feature-vector sequence.
    Each gloss gets a fixed per-dimension offset so classes are separable
    (a sanity-check training run needs *something* learnable), plus
    per-sample Gaussian noise so same-gloss samples still differ."""
    offset = np.sin(np.arange(feature_size, dtype=np.float32) + gloss_index * 7.0) * 0.5
    noise = rng.normal(loc=0.0, scale=0.1, size=(sequence_length, feature_size))
    return (offset + noise).astype(np.float32)


def generate_demo_dataset(
    output_dir: Path | str,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    sequence_length: int = 32,
    samples_per_signer_gloss: int = 3,
    seed: int = 42,
) -> list[SampleAnnotation]:
    """Writes synthetic `(sequence_length, feature_vector_size(config))`
    `.npy` sequences under `<output_dir>/processed/demo/`, and a matching
    `<output_dir>/annotations/demo_annotations.jsonl`. Returns the written
    annotations. Deterministic given `seed`."""
    feature_size = feature_vector_size(config)
    if feature_size == 0:
        raise ValueError("FeatureConfig must enable at least one modality")

    output_dir = Path(output_dir)
    processed_dir = output_dir / "processed" / "demo"
    processed_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    samples: list[SampleAnnotation] = []

    for signer_id in DEMO_SIGNERS:
        for gloss_index, gloss in enumerate(DEMO_GLOSSES):
            for take in range(samples_per_signer_gloss):
                sample_id = f"demo_{signer_id}_{gloss}_{take:02d}"
                sequence = _synthetic_sequence(rng, gloss_index, sequence_length, feature_size)
                npy_path = processed_dir / f"{sample_id}.npy"
                np.save(npy_path, sequence)

                samples.append(
                    SampleAnnotation(
                        sample_id=sample_id,
                        clip_path=str(npy_path.relative_to(output_dir)),
                        signer_id=signer_id,
                        gloss=gloss,
                        start_frame=0,
                        end_frame=sequence_length,
                        fps=12.0,
                        source=DEMO_SOURCE_TAG,
                    )
                )

    annotations_path = output_dir / "annotations" / "demo_annotations.jsonl"
    write_annotations(annotations_path, samples)
    return samples
