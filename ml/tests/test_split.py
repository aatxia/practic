import pytest

from ml.datasets.annotation import SampleAnnotation
from ml.datasets.split import SplitRatios, gloss_stratified_split, signer_independent_split


def _samples(signer_counts: dict[str, int]) -> list[SampleAnnotation]:
    samples = []
    for signer_id, count in signer_counts.items():
        for i in range(count):
            samples.append(
                SampleAnnotation(
                    sample_id=f"{signer_id}_{i}",
                    clip_path=f"processed/demo/{signer_id}_{i}.npy",
                    signer_id=signer_id,
                    gloss="TAK",
                    start_frame=0,
                    end_frame=32,
                    fps=12.0,
                    source="demo_synthetic",
                )
            )
    return samples


def _samples_by_gloss(gloss_counts: dict[str, int], signer_id: str = "aatxia") -> list[SampleAnnotation]:
    samples = []
    for gloss, count in gloss_counts.items():
        for i in range(count):
            samples.append(
                SampleAnnotation(
                    sample_id=f"{signer_id}_{gloss}_{i:02d}",
                    clip_path=f"real/raw/{signer_id}/{gloss}_{i:02d}.mp4",
                    signer_id=signer_id,
                    gloss=gloss,
                    start_frame=0,
                    end_frame=90,
                    fps=30.0,
                    source="real_webcam",
                )
            )
    return samples


def test_no_signer_appears_in_more_than_one_split():
    samples = _samples({f"signer_{i:02d}": 5 for i in range(10)})

    split = signer_independent_split(samples, seed=7)

    signer_of = {s.sample_id: s.signer_id for s in samples}
    split_of_signer: dict[str, str] = {}
    for split_name, sample_ids in split.items():
        for sample_id in sample_ids:
            signer = signer_of[sample_id]
            previous_split = split_of_signer.setdefault(signer, split_name)
            assert previous_split == split_name, (
                f"signer {signer!r} appears in both {previous_split!r} and {split_name!r}"
            )


def test_every_sample_assigned_exactly_once():
    samples = _samples({f"signer_{i:02d}": 3 for i in range(6)})

    split = signer_independent_split(samples, seed=1)

    all_assigned = sorted(sid for ids in split.values() for sid in ids)
    assert all_assigned == sorted(s.sample_id for s in samples)


def test_deterministic_given_same_seed():
    samples = _samples({f"signer_{i:02d}": 4 for i in range(8)})

    first = signer_independent_split(samples, seed=99)
    second = signer_independent_split(samples, seed=99)

    assert first == second


def test_ratios_roughly_respected_with_many_equal_signers():
    samples = _samples({f"signer_{i:02d}": 10 for i in range(20)})

    split = signer_independent_split(samples, ratios=SplitRatios(train=0.7, val=0.15, test=0.15), seed=3)

    total = len(samples)
    assert len(split["train"]) / total == pytest.approx(0.7, abs=0.1)
    assert len(split["val"]) / total == pytest.approx(0.15, abs=0.1)
    assert len(split["test"]) / total == pytest.approx(0.15, abs=0.1)


def test_raises_when_too_few_signers_for_requested_splits():
    samples = _samples({"signer_01": 5, "signer_02": 5})

    with pytest.raises(ValueError, match="need at least 3 distinct signers"):
        signer_independent_split(samples, ratios=SplitRatios(train=0.7, val=0.15, test=0.15))


def test_two_way_split_only_needs_two_signers():
    samples = _samples({"signer_01": 5, "signer_02": 5})

    split = signer_independent_split(samples, ratios=SplitRatios(train=0.5, val=0.0, test=0.5))

    assert split["val"] == []
    assert len(split["train"]) > 0
    assert len(split["test"]) > 0


def test_raises_on_empty_sample_list():
    with pytest.raises(ValueError, match="empty"):
        signer_independent_split([])


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": 0.5, "val": 0.3, "test": 0.3},  # sums to 1.1
        {"train": 0.5, "val": 0.1, "test": 0.1},  # sums to 0.7
        {"train": -0.1, "val": 0.6, "test": 0.5},  # negative
    ],
)
def test_split_ratios_validates_sum_and_sign(ratios):
    with pytest.raises(ValueError):
        SplitRatios(**ratios)


def test_gloss_stratified_split_every_sample_assigned_exactly_once():
    samples = _samples_by_gloss({"WE": 15, "WANT": 15, "HAVE": 15, "I": 9})

    split = gloss_stratified_split(samples, seed=1)

    all_assigned = sorted(sid for ids in split.values() for sid in ids)
    assert all_assigned == sorted(s.sample_id for s in samples)


def test_gloss_stratified_split_every_split_gets_every_well_represented_gloss():
    samples = _samples_by_gloss({"WE": 15, "WANT": 15, "HAVE": 15})

    split = gloss_stratified_split(samples, seed=1)

    gloss_of = {s.sample_id: s.gloss for s in samples}
    for split_name in ("train", "val", "test"):
        glosses_in_split = {gloss_of[sid] for sid in split[split_name]}
        assert glosses_in_split == {"WE", "WANT", "HAVE"}


def test_gloss_stratified_split_puts_thin_class_entirely_in_train():
    samples = _samples_by_gloss({"WE": 15, "I": 3})

    split = gloss_stratified_split(samples, seed=1)

    gloss_of = {s.sample_id: s.gloss for s in samples}
    i_sample_ids = {sid for sid, gloss in gloss_of.items() if gloss == "I"}
    assert i_sample_ids <= set(split["train"])
    assert not (i_sample_ids & set(split["val"]))
    assert not (i_sample_ids & set(split["test"]))


def test_gloss_stratified_split_deterministic_given_same_seed():
    samples = _samples_by_gloss({"WE": 15, "WANT": 15})

    first = gloss_stratified_split(samples, seed=5)
    second = gloss_stratified_split(samples, seed=5)

    assert first == second


def test_gloss_stratified_split_works_with_a_single_signer():
    samples = _samples_by_gloss({"WE": 15, "WANT": 15, "HAVE": 15}, signer_id="aatxia")

    split = gloss_stratified_split(samples, seed=1)

    assert len(split["train"]) > 0
    assert len(split["val"]) > 0
    assert len(split["test"]) > 0


def test_gloss_stratified_split_raises_on_empty_sample_list():
    with pytest.raises(ValueError, match="empty"):
        gloss_stratified_split([])
