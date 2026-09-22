import numpy as np

from ml.fingerspelling.split import stratified_split


def test_every_split_is_disjoint_and_covers_all_samples():
    labels = ["А"] * 20 + ["Б"] * 20 + ["В"] * 20

    split = stratified_split(labels, seed=1)

    all_idx = np.concatenate([split.train, split.val, split.test])
    assert sorted(all_idx.tolist()) == list(range(60))
    assert set(split.train.tolist()) & set(split.val.tolist()) == set()
    assert set(split.train.tolist()) & set(split.test.tolist()) == set()
    assert set(split.val.tolist()) & set(split.test.tolist()) == set()


def test_tiny_classes_go_entirely_to_train_not_split_at_all():
    labels = ["Я"] * 3 + ["Б"] * 30  # Я has fewer than MIN_SAMPLES_FOR_VAL_TEST

    split = stratified_split(labels, seed=1)

    labels_arr = np.array(labels)
    ya_indices = set(np.where(labels_arr == "Я")[0].tolist())
    assert ya_indices.issubset(set(split.train.tolist()))
    assert not (ya_indices & set(split.val.tolist()))
    assert not (ya_indices & set(split.test.tolist()))


def test_every_class_with_enough_samples_appears_in_every_split():
    labels = ["А"] * 50 + ["Б"] * 50

    split = stratified_split(labels, seed=1)
    labels_arr = np.array(labels)

    for label in ("А", "Б"):
        idx = set(np.where(labels_arr == label)[0].tolist())
        assert idx & set(split.train.tolist())
        assert idx & set(split.val.tolist())
        assert idx & set(split.test.tolist())


def test_deterministic_given_seed():
    labels = ["А"] * 40 + ["Б"] * 40

    split_a = stratified_split(labels, seed=7)
    split_b = stratified_split(labels, seed=7)

    np.testing.assert_array_equal(split_a.train, split_b.train)
    np.testing.assert_array_equal(split_a.val, split_b.val)
    np.testing.assert_array_equal(split_a.test, split_b.test)
