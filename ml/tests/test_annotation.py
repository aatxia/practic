import pytest

from ml.datasets.annotation import SampleAnnotation, load_annotations, write_annotations


def _sample(**overrides) -> SampleAnnotation:
    defaults = {
        "sample_id": "s1",
        "clip_path": "processed/demo/s1.npy",
        "signer_id": "signer_01",
        "gloss": "TAK",
        "start_frame": 0,
        "end_frame": 32,
        "fps": 12.0,
        "source": "demo_synthetic",
    }
    defaults.update(overrides)
    return SampleAnnotation(**defaults)


def test_round_trips_through_jsonl(tmp_path):
    samples = [_sample(sample_id="s1"), _sample(sample_id="s2", gloss="NI")]
    path = tmp_path / "annotations.jsonl"

    write_annotations(path, samples)
    loaded = load_annotations(path)

    assert loaded == samples


def test_load_skips_blank_lines(tmp_path):
    path = tmp_path / "annotations.jsonl"
    write_annotations(path, [_sample()])
    with path.open("a", encoding="utf-8") as f:
        f.write("\n\n")

    loaded = load_annotations(path)

    assert len(loaded) == 1


def test_load_raises_with_file_and_line_on_malformed_json(tmp_path):
    path = tmp_path / "annotations.jsonl"
    path.write_text("not json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"annotations\.jsonl:1"):
        load_annotations(path)


def test_load_raises_on_duplicate_sample_id(tmp_path):
    path = tmp_path / "annotations.jsonl"
    write_annotations(path, [_sample(sample_id="dup"), _sample(sample_id="dup")])

    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_annotations(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_id": ""},
        {"signer_id": ""},
        {"start_frame": 10, "end_frame": 10},
        {"start_frame": 10, "end_frame": 5},
        {"fps": 0},
        {"fps": -1},
    ],
)
def test_rejects_invalid_fields(overrides):
    with pytest.raises(ValueError):
        _sample(**overrides)
