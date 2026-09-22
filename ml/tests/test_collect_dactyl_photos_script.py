import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.collect_dactyl_photos import (
    _ALPHABET_ORDER,
    _should_keep_capturing,
    choose_letter,
    count_existing_photos,
    existing_letters,
    missing_letters,
    next_photo_path,
)

from ml.nlp.fingerspelling import UKRAINIAN_ALPHABET

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "USL_alphabet_train"


def test_alphabet_order_literal_matches_the_real_ukrainian_alphabet_set():
    """Guards against the menu-ordering literal drifting out of sync with
    ml.nlp.fingerspelling.UKRAINIAN_ALPHABET, the real source of truth."""
    assert set(_ALPHABET_ORDER) == UKRAINIAN_ALPHABET
    assert len(_ALPHABET_ORDER) == len(UKRAINIAN_ALPHABET)


def test_existing_letters_is_empty_for_a_fresh_directory(tmp_path):
    assert existing_letters(tmp_path / "does_not_exist") == set()


def test_existing_letters_skips_excluded_prefixed_directories(tmp_path):
    (tmp_path / "А").mkdir()
    (tmp_path / "Б").mkdir()
    (tmp_path / "_excluded_non_ukrainian").mkdir()
    (tmp_path / "_excluded_unclear").mkdir()

    assert existing_letters(tmp_path) == {"А", "Б"}


def test_missing_letters_excludes_letters_that_already_have_a_directory(tmp_path):
    (tmp_path / "А").mkdir()
    (tmp_path / "Б").mkdir()

    result = missing_letters(tmp_path)
    assert "А" not in result
    assert "Б" not in result
    assert "В" in result
    assert len(result) == len(UKRAINIAN_ALPHABET) - 2


def test_missing_letters_is_in_real_alphabet_order(tmp_path):
    result = missing_letters(tmp_path)
    assert result == [char.upper() for char in _ALPHABET_ORDER]


def test_missing_letters_against_the_real_dataset_matches_the_known_gap():
    """A real check against the actual project data (same spirit as
    test_collect_dynamic_signs_script.py's calibration test): confirms the
    7 letters this script was written to fill in are exactly the ones
    still missing, not a guessed or stale list."""
    assert missing_letters(DATA_ROOT) == ["Ґ", "Є", "І", "Ї", "К", "Л", "Н"]


def test_count_existing_photos_is_zero_for_a_fresh_letter(tmp_path):
    assert count_existing_photos(tmp_path, "К") == 0


def test_count_existing_photos_counts_real_files_on_disk(tmp_path):
    letter_dir = tmp_path / "К"
    letter_dir.mkdir()
    (letter_dir / "IMG_20240101_000000.jpg").write_bytes(b"")
    (letter_dir / "IMG_20240101_000001.jpg").write_bytes(b"")

    assert count_existing_photos(tmp_path, "К") == 2
    assert count_existing_photos(tmp_path, "Л") == 0


def test_choose_letter_returns_none_on_an_empty_answer(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert choose_letter(tmp_path, ["К", "Л"], target=20) is None


def test_choose_letter_returns_the_selected_letter(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert choose_letter(tmp_path, ["К", "Л", "Н"], target=20) == "Л"


def test_choose_letter_reprompts_on_an_out_of_range_number(tmp_path, monkeypatch):
    answers = iter(["99", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert choose_letter(tmp_path, ["К", "Л"], target=20) == "К"


def test_should_keep_capturing_normally_stops_once_target_is_reached():
    assert _should_keep_capturing(existing=19, target=20, started_above_target=False) is True
    assert _should_keep_capturing(existing=20, target=20, started_above_target=False) is False


def test_should_keep_capturing_never_auto_stops_when_the_session_started_already_at_or_above_target():
    """Without this, re-shooting a letter with existing photos already at
    or above target (e.g. Ь with 1500 unusable existing photos -- see the
    module docstring's ml/fingerspelling/dataset.py reference) would exit
    the capture loop before the camera window even opened, because
    `existing < target` was already false before the first photo. Once a
    session starts above target, only q should end it -- the on-disk
    count no longer means "done"."""
    assert _should_keep_capturing(existing=1500, target=20, started_above_target=True) is True
    assert _should_keep_capturing(existing=2000, target=20, started_above_target=True) is True


def test_choose_letter_reprompts_on_a_non_numeric_answer(tmp_path, monkeypatch):
    answers = iter(["not a number", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert choose_letter(tmp_path, ["К", "Л"], target=20) == "Л"


class _FixedNow(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.datetime(2024, 3, 15, 10, 30, 45)


def test_next_photo_path_follows_the_existing_img_timestamp_convention(tmp_path, monkeypatch):
    from scripts import collect_dactyl_photos as module

    monkeypatch.setattr(module.datetime, "datetime", _FixedNow)

    path = next_photo_path(tmp_path, "К")

    assert path == tmp_path / "К" / "IMG_20240315_103045.jpg"
    assert path.parent.exists()


def test_next_photo_path_never_overwrites_an_existing_file(tmp_path, monkeypatch):
    from scripts import collect_dactyl_photos as module

    monkeypatch.setattr(module.datetime, "datetime", _FixedNow)
    letter_dir = tmp_path / "К"
    letter_dir.mkdir()
    (letter_dir / "IMG_20240315_103045.jpg").write_bytes(b"existing")

    path = next_photo_path(tmp_path, "К")

    assert path == letter_dir / "IMG_20240315_103045_1.jpg"


def test_next_photo_path_handles_two_back_to_back_collisions(tmp_path, monkeypatch):
    from scripts import collect_dactyl_photos as module

    monkeypatch.setattr(module.datetime, "datetime", _FixedNow)
    letter_dir = tmp_path / "К"
    letter_dir.mkdir()
    (letter_dir / "IMG_20240315_103045.jpg").write_bytes(b"existing")
    (letter_dir / "IMG_20240315_103045_1.jpg").write_bytes(b"existing")

    path = next_photo_path(tmp_path, "К")

    assert path == letter_dir / "IMG_20240315_103045_2.jpg"


def test_main_exits_gracefully_on_ctrl_c_at_the_letter_menu(tmp_path, monkeypatch, capsys):
    """Mirrors the same fix in collect_dynamic_signs.py: Ctrl+C at the
    "Номер літери:" prompt must exit cleanly, not raise an unhandled
    KeyboardInterrupt traceback."""
    from scripts.collect_dactyl_photos import main

    def _raise_keyboard_interrupt(_prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise_keyboard_interrupt)

    main(["--output-dir", str(tmp_path), "--letters", "К"])

    captured = capsys.readouterr()
    assert "Stopped early" in captured.out
    assert "Traceback" not in captured.out


def test_main_reports_nothing_to_do_when_no_letters_are_missing(tmp_path, capsys):
    from scripts.collect_dactyl_photos import main

    for char in _ALPHABET_ORDER:
        (tmp_path / char.upper()).mkdir()

    main(["--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert "Немає літер без фото" in captured.out
