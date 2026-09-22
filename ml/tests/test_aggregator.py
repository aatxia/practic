import pytest

from ml.inference.aggregator import GlossSequenceAggregator


def test_does_not_confirm_before_stability_threshold_reached():
    agg = GlossSequenceAggregator(stability_frames=5, confidence_threshold=0.5)

    for _ in range(4):
        confirmed = agg.update("TAK", confidence=0.9)
        assert confirmed is False

    assert agg.sequence == []


def test_confirms_exactly_on_the_stability_frame():
    agg = GlossSequenceAggregator(stability_frames=3, confidence_threshold=0.5)

    assert agg.update("TAK", confidence=0.9) is False
    assert agg.update("TAK", confidence=0.9) is False
    assert agg.update("TAK", confidence=0.9) is True

    assert agg.sequence == ["TAK"]


def test_does_not_reconfirm_the_same_held_sign():
    agg = GlossSequenceAggregator(stability_frames=2, confidence_threshold=0.5)

    agg.update("TAK", confidence=0.9)
    confirmed = agg.update("TAK", confidence=0.9)
    assert confirmed is True

    # Signer keeps holding the same sign for many more frames.
    for _ in range(20):
        assert agg.update("TAK", confidence=0.9) is False

    assert agg.sequence == ["TAK"]


def test_confirms_a_new_sign_after_the_gloss_changes():
    agg = GlossSequenceAggregator(stability_frames=2, confidence_threshold=0.5)

    agg.update("TAK", confidence=0.9)
    agg.update("TAK", confidence=0.9)
    agg.update("NI", confidence=0.9)
    confirmed = agg.update("NI", confidence=0.9)

    assert confirmed is True
    assert agg.sequence == ["TAK", "NI"]


def test_low_confidence_predictions_never_count_toward_stability():
    agg = GlossSequenceAggregator(stability_frames=2, confidence_threshold=0.5)

    for _ in range(10):
        assert agg.update("TAK", confidence=0.1) is False

    assert agg.sequence == []


def test_low_confidence_frame_interrupts_an_in_progress_streak():
    agg = GlossSequenceAggregator(stability_frames=3, confidence_threshold=0.5)

    agg.update("TAK", confidence=0.9)
    agg.update("TAK", confidence=0.9)
    agg.update("TAK", confidence=0.1)  # breaks the streak before confirmation
    assert agg.update("TAK", confidence=0.9) is False  # streak restarts at 1
    assert agg.update("TAK", confidence=0.9) is False  # streak at 2
    assert agg.update("TAK", confidence=0.9) is True  # streak at 3 -- confirmed

    assert agg.sequence == ["TAK"]


def test_reset_clears_sequence_and_streak_state():
    agg = GlossSequenceAggregator(stability_frames=2, confidence_threshold=0.5)
    agg.update("TAK", confidence=0.9)
    agg.update("TAK", confidence=0.9)
    assert agg.sequence == ["TAK"]

    agg.reset()

    assert agg.sequence == []
    # A sign confirmed before reset can be confirmed again afterwards.
    agg.update("TAK", confidence=0.9)
    assert agg.update("TAK", confidence=0.9) is True
    assert agg.sequence == ["TAK"]


def test_rejects_non_positive_stability_frames():
    with pytest.raises(ValueError, match="stability_frames must be >= 1"):
        GlossSequenceAggregator(stability_frames=0)
