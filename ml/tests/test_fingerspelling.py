import pytest

from ml.nlp.fingerspelling import (
    UnspellableCharacterError,
    despell,
    is_fingerspell_gloss,
    spell_word,
)


def test_spell_word_produces_one_gloss_per_letter():
    assert spell_word("так") == ["FS_Т", "FS_А", "FS_К"]


def test_spell_word_is_case_insensitive_on_input():
    assert spell_word("Так") == spell_word("так") == spell_word("ТАК")


def test_spell_word_covers_every_letter_of_the_ukrainian_alphabet():
    alphabet = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
    assert spell_word(alphabet) == [f"FS_{letter.upper()}" for letter in alphabet]


def test_spell_word_rejects_latin_script():
    with pytest.raises(UnspellableCharacterError, match="'h'"):
        spell_word("hello")


def test_spell_word_rejects_digits():
    with pytest.raises(UnspellableCharacterError, match="'5'"):
        spell_word("5")


def test_spell_word_rejects_apostrophe():
    with pytest.raises(UnspellableCharacterError, match="'"):
        spell_word("комп'ютер")


def test_spell_word_rejects_empty_word():
    with pytest.raises(ValueError, match="must not be empty"):
        spell_word("")


def test_is_fingerspell_gloss_true_for_a_spelled_letter():
    assert is_fingerspell_gloss("FS_А") is True
    assert is_fingerspell_gloss("FS_Я") is True


def test_is_fingerspell_gloss_false_for_a_lexicon_gloss():
    assert is_fingerspell_gloss("TAK") is False
    assert is_fingerspell_gloss("WATER") is False
    assert is_fingerspell_gloss("FS_") is False
    assert is_fingerspell_gloss("FS_AB") is False  # not a single letter


def test_despell_reverses_spell_word():
    word = "оксана"
    assert despell(spell_word(word)) == word


def test_despell_rejects_a_non_fingerspell_gloss():
    with pytest.raises(ValueError, match="TAK"):
        despell(["FS_Т", "TAK"])


def test_despell_rejects_empty_list():
    with pytest.raises(ValueError, match="must not be empty"):
        despell([])
