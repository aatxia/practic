import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from scripts.collect_dynamic_signs import (
    MIN_HAND_DETECTION_RATE,
    choose_word,
    count_existing_takes,
    verify_landmarks,
)

from ml.nlp.vocabulary import (
    PROPOSED_NEW_DYNAMIC_WORDS,
    TARGET_DYNAMIC_VOCABULARY,
    TRAINED_DYNAMIC_WORDS,
)
from ml.preprocessing.landmarks import FeatureToggles, LandmarkExtractor

MEDIAPIPE_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "mediapipe"


def test_count_existing_takes_is_zero_for_a_fresh_signer_and_gloss(tmp_path):
    assert count_existing_takes(tmp_path, "oksana", "YOU") == 0


def test_count_existing_takes_counts_real_files_on_disk(tmp_path):
    clip_dir = tmp_path / "raw" / "oksana"
    clip_dir.mkdir(parents=True)
    (clip_dir / "YOU_00.mp4").write_bytes(b"")
    (clip_dir / "YOU_01.mp4").write_bytes(b"")
    (clip_dir / "HELP_00.mp4").write_bytes(b"")

    assert count_existing_takes(tmp_path, "oksana", "YOU") == 2
    assert count_existing_takes(tmp_path, "oksana", "HELP") == 1
    assert count_existing_takes(tmp_path, "oksana", "FOOD") == 0


def test_new_target_vocabulary_is_the_7_trained_plus_8_proposed_words():
    assert len(TRAINED_DYNAMIC_WORDS) == 7
    assert len(PROPOSED_NEW_DYNAMIC_WORDS) == 8
    assert set(PROPOSED_NEW_DYNAMIC_WORDS) == {"YOU", "NEED", "HELP", "FOOD", "EAT", "DRINK", "GO", "HOME"}
    assert len(TARGET_DYNAMIC_VOCABULARY) == 15
    assert len(set(TARGET_DYNAMIC_VOCABULARY)) == 15  # no accidental duplicate between the two tuples


def test_choose_word_returns_none_on_an_empty_answer(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert choose_word(tmp_path, "oksana", ["YOU", "HELP"], target=20) is None


def test_choose_word_returns_the_selected_gloss(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert choose_word(tmp_path, "oksana", ["YOU", "HELP", "FOOD"], target=20) == "HELP"


def test_choose_word_reprompts_on_an_out_of_range_number(tmp_path, monkeypatch):
    answers = iter(["99", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert choose_word(tmp_path, "oksana", ["YOU", "HELP"], target=20) == "YOU"


def test_choose_word_reprompts_on_a_non_numeric_answer(tmp_path, monkeypatch):
    answers = iter(["not a number", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert choose_word(tmp_path, "oksana", ["YOU", "HELP"], target=20) == "HELP"


def _write_clip(path: Path, frames: list[np.ndarray], fps: float = 12.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_verify_landmarks_reports_zero_for_a_blank_clip_with_no_real_hand(tmp_path):
    """An honest floor: a clip that never shows a hand (e.g. camera pointed
    at a blank wall) must measure at or near 0%, confirming the check
    would actually catch a genuinely broken capture -- not just pass
    everything unconditionally."""
    frames = [np.full((48, 64, 3), 60, dtype=np.uint8) for _ in range(10)]
    clip_path = tmp_path / "BLANK_00.mp4"
    _write_clip(clip_path, frames)

    extractor = LandmarkExtractor(model_dir=str(MEDIAPIPE_MODELS_DIR), features=FeatureToggles(hands=True, pose=False, face=False))
    try:
        rate, frame_count = verify_landmarks(clip_path, extractor)
    finally:
        extractor.close()

    assert frame_count == 10
    assert rate < MIN_HAND_DETECTION_RATE


def test_verify_landmarks_detects_a_real_hand_in_a_real_training_photo_repeated(tmp_path):
    """The inverse check, against real data: a clip built from a real
    photo of a held letter (a real hand, unlike the synthetic blank frame
    above) must measure a high detection rate, well above the threshold."""
    photo_path = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "USL_alphabet_train"
        / "А"
    )
    real_photo = next(photo_path.glob("*.jpg"))
    frame = cv2.imread(str(real_photo))
    assert frame is not None
    clip_path = tmp_path / "REAL_00.mp4"
    _write_clip(clip_path, [frame] * 10, fps=12.0)

    extractor = LandmarkExtractor(model_dir=str(MEDIAPIPE_MODELS_DIR), features=FeatureToggles(hands=True, pose=False, face=False))
    try:
        rate, frame_count = verify_landmarks(clip_path, extractor)
    finally:
        extractor.close()

    assert frame_count == 10
    assert rate > MIN_HAND_DETECTION_RATE


def test_verify_landmarks_calibration_threshold_passes_every_already_trained_real_v1_clip():
    """The exact real-world calibration check this threshold was derived
    from (see MIN_HAND_DETECTION_RATE's docstring) -- every one of these
    real clips already trained a working real_v1 checkpoint, so the
    landmark-verification step must never flag any of them as low quality."""
    raw_dir = Path(__file__).resolve().parent.parent.parent / "data" / "real" / "raw" / "aatxia"
    sample_clips = ["WANT_00.mp4", "HAVE_00.mp4", "WATER_00.mp4", "WE_03.mp4", "I_00.mp4"]

    extractor = LandmarkExtractor(model_dir=str(MEDIAPIPE_MODELS_DIR), features=FeatureToggles(hands=True, pose=False, face=False))
    try:
        for clip_name in sample_clips:
            clip_path = raw_dir / clip_name
            if not clip_path.exists():
                continue
            rate, _ = verify_landmarks(clip_path, extractor)
            assert rate >= MIN_HAND_DETECTION_RATE, f"{clip_name} scored {rate:.0%}, below the threshold"
    finally:
        extractor.close()


def test_main_exits_gracefully_on_ctrl_c_at_the_word_menu(tmp_path, monkeypatch, capsys):
    """Without a handler, Ctrl+C at the "Номер слова:" prompt raises
    KeyboardInterrupt as an unhandled traceback instead of exiting
    cleanly like every other interactive script in this project does
    (see scripts/record_dataset.py's own "Stopped early" message)."""
    from scripts.collect_dynamic_signs import main

    def _raise_keyboard_interrupt(_prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise_keyboard_interrupt)

    main(["--signer-id", "test_signer", "--output-dir", str(tmp_path), "--words", "YOU"])

    captured = capsys.readouterr()
    assert "Stopped early" in captured.out
    assert "Traceback" not in captured.out
