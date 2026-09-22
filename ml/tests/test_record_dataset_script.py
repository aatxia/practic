import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.nlp.lexicon import NEGATION_GLOSS, TENSE_PAST_GLOSS
from scripts.record_dataset import (
    DEFAULT_WORDS,
    build_annotation,
    display_label,
    next_take_index,
)


def test_display_label_covers_every_default_word():
    for gloss in DEFAULT_WORDS:
        label = display_label(gloss)
        assert label and label != ""


def test_display_label_uses_real_lexicon_forms():
    assert display_label("I") == "Я"
    assert display_label("WANT") == "хотіти"
    assert display_label("WATER") == "вода"
    assert display_label("TODAY") == "сьогодні"


def test_display_label_explains_grammar_markers_not_just_prints_the_code():
    assert display_label(NEGATION_GLOSS) != NEGATION_GLOSS
    assert display_label(TENSE_PAST_GLOSS) != TENSE_PAST_GLOSS


def test_display_label_falls_back_to_the_raw_code_for_an_unknown_gloss():
    assert display_label("SOME_NEW_WORD") == "SOME_NEW_WORD"


def test_next_take_index_is_zero_for_a_fresh_signer_and_gloss(tmp_path):
    assert next_take_index(tmp_path, "oksana", "WANT") == 0


def test_next_take_index_skips_existing_takes(tmp_path):
    clip_dir = tmp_path / "raw" / "oksana"
    clip_dir.mkdir(parents=True)
    (clip_dir / "WANT_00.mp4").write_bytes(b"")
    (clip_dir / "WANT_01.mp4").write_bytes(b"")

    assert next_take_index(tmp_path, "oksana", "WANT") == 2


def test_next_take_index_reuses_a_gap_left_by_a_discarded_retake(tmp_path):
    clip_dir = tmp_path / "raw" / "oksana"
    clip_dir.mkdir(parents=True)
    (clip_dir / "WANT_00.mp4").write_bytes(b"")
    (clip_dir / "WANT_02.mp4").write_bytes(b"")  # 01 was discarded and deleted

    assert next_take_index(tmp_path, "oksana", "WANT") == 1


def test_next_take_index_is_independent_per_gloss_and_signer(tmp_path):
    clip_dir = tmp_path / "raw" / "oksana"
    clip_dir.mkdir(parents=True)
    (clip_dir / "WANT_00.mp4").write_bytes(b"")

    assert next_take_index(tmp_path, "oksana", "LIKE") == 0
    assert next_take_index(tmp_path, "inna", "WANT") == 0


def test_build_annotation_matches_the_real_pipeline_schema(tmp_path):
    clip_path = tmp_path / "raw" / "oksana" / "WANT_00.mp4"

    annotation = build_annotation(clip_path, "oksana", "WANT", take=0, frame_count=36, fps=12.0)

    assert annotation.sample_id == "oksana_WANT_00"
    assert annotation.clip_path == str(clip_path)
    assert annotation.signer_id == "oksana"
    assert annotation.gloss == "WANT"
    assert annotation.start_frame == 0
    assert annotation.end_frame == 36
    assert annotation.fps == 12.0
    assert annotation.source == "real_webcam"
