import pytest

from app.services.translation_service import RuleBasedTranslationService


def test_gloss_to_text_delegates_to_the_real_rule_based_engine():
    service = RuleBasedTranslationService()

    assert service.gloss_to_text(["I", "WANT", "WATER"]) == "Я хочу води."
    assert service.gloss_to_text(["TAK"]) == "Так."


def test_gloss_to_text_raises_clearly_on_unknown_gloss():
    service = RuleBasedTranslationService()

    with pytest.raises(ValueError, match="WATERMELON"):
        service.gloss_to_text(["I", "WANT", "WATERMELON"])


def test_text_to_gloss_delegates_to_the_real_rule_based_engine():
    service = RuleBasedTranslationService()

    assert service.text_to_gloss("Я хочу води.") == ["I", "WANT", "WATER"]
    assert service.text_to_gloss("Так.") == ["TAK"]


def test_text_to_gloss_fingerspells_a_word_outside_the_lexicon():
    # Phase 16: "кавун" isn't a lexicon noun, but every letter is a
    # Ukrainian dactyl letter, so it's spelled rather than rejected.
    service = RuleBasedTranslationService()

    assert service.text_to_gloss("Я хочу кавун.") == [
        "I",
        "WANT",
        "FS_К",
        "FS_А",
        "FS_В",
        "FS_У",
        "FS_Н",
    ]


def test_text_to_gloss_raises_clearly_on_a_word_with_no_dactyl_handshape():
    service = RuleBasedTranslationService()

    with pytest.raises(ValueError, match="pizza"):
        service.text_to_gloss("Я хочу pizza.")
