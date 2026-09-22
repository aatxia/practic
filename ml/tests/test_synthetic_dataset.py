import numpy as np
import pytest

from ml.datasets.annotation import load_annotations
from ml.datasets.dataset import load_feature_sequence
from ml.datasets.split import signer_independent_split
from ml.datasets.synthetic import (
    DEMO_GLOSSES,
    DEMO_SIGNERS,
    DEMO_SOURCE_TAG,
    generate_demo_dataset,
)
from ml.features.feature_vector import FeatureConfig, feature_vector_size


def test_generates_expected_number_of_samples(tmp_path):
    samples = generate_demo_dataset(tmp_path, samples_per_signer_gloss=2, seed=1)

    assert len(samples) == len(DEMO_SIGNERS) * len(DEMO_GLOSSES) * 2
    assert all(s.source == DEMO_SOURCE_TAG for s in samples)


def test_sequences_match_declared_feature_vector_size(tmp_path):
    config = FeatureConfig(hands=True, pose=False, face=False)

    samples = generate_demo_dataset(tmp_path, config=config, sequence_length=16, samples_per_signer_gloss=1, seed=1)

    for sample in samples:
        sequence = load_feature_sequence(sample, tmp_path)
        assert sequence.shape == (16, feature_vector_size(config))


def test_deterministic_given_same_seed(tmp_path):
    out_a, out_b = tmp_path / "a", tmp_path / "b"

    samples_a = generate_demo_dataset(out_a, samples_per_signer_gloss=1, seed=42)
    samples_b = generate_demo_dataset(out_b, samples_per_signer_gloss=1, seed=42)

    for sample_a, sample_b in zip(samples_a, samples_b, strict=True):
        seq_a = load_feature_sequence(sample_a, out_a)
        seq_b = load_feature_sequence(sample_b, out_b)
        np.testing.assert_array_equal(seq_a, seq_b)


def test_different_seeds_produce_different_sequences(tmp_path):
    out_a, out_b = tmp_path / "a", tmp_path / "b"

    samples_a = generate_demo_dataset(out_a, samples_per_signer_gloss=1, seed=1)
    samples_b = generate_demo_dataset(out_b, samples_per_signer_gloss=1, seed=2)

    seq_a = load_feature_sequence(samples_a[0], out_a)
    seq_b = load_feature_sequence(samples_b[0], out_b)
    assert not np.array_equal(seq_a, seq_b)


def test_rejects_empty_feature_config(tmp_path):
    with pytest.raises(ValueError, match="at least one modality"):
        generate_demo_dataset(tmp_path, config=FeatureConfig(hands=False, pose=False, face=False))


def test_annotations_file_is_loadable_and_matches_returned_samples(tmp_path):
    samples = generate_demo_dataset(tmp_path, samples_per_signer_gloss=1, seed=5)

    loaded = load_annotations(tmp_path / "annotations" / "demo_annotations.jsonl")

    assert loaded == samples


def test_load_feature_sequence_refuses_non_demo_source_without_feature_config_and_fps(tmp_path):
    # Real (non-demo) sources decode actual video through the real pipeline
    # (see ml/tests/test_dataset_real_video.py for that success path) -- but
    # there's no honest default feature_config/target_fps to fall back to,
    # so omitting them is a hard error rather than a silent guess.
    samples = generate_demo_dataset(tmp_path, samples_per_signer_gloss=1, seed=1)
    real_looking = samples[0].__class__(
        **{**samples[0].__dict__, "sample_id": "real_1", "source": "uksl_real"}
    )

    with pytest.raises(ValueError, match="requires feature_config and target_fps"):
        load_feature_sequence(real_looking, tmp_path)


def test_end_to_end_split_respects_signer_independence(tmp_path):
    generate_demo_dataset(tmp_path, samples_per_signer_gloss=3, seed=1)
    samples = load_annotations(tmp_path / "annotations" / "demo_annotations.jsonl")

    split = signer_independent_split(samples, seed=1)

    signer_of = {s.sample_id: s.signer_id for s in samples}
    split_of_signer: dict[str, str] = {}
    for split_name, sample_ids in split.items():
        for sample_id in sample_ids:
            signer = signer_of[sample_id]
            assert split_of_signer.setdefault(signer, split_name) == split_name
    assert all(len(ids) > 0 for ids in split.values())
